"""Evaluation entrypoint for ForecastBench backtest harness."""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import os
import re
import time
from pathlib import Path
from typing import Callable, Protocol

from cutoff import CutoffEnvironment
from fetch_data import Question, QuestionSet, Resolution, ResolvedQuestion, load_data, join_resolved_questions
from score import ScoringResult, score_forecasts


class Forecaster(Protocol):
    def __call__(self, question: Question) -> float: ...


class AsyncForecaster(Protocol):
    async def __call__(
        self, question: Question, cutoff: CutoffEnvironment | None = None
    ) -> float: ...


def split_held_out(
    question_sets: list[QuestionSet],
    n_held_out: int = 2,
) -> tuple[list[QuestionSet], list[QuestionSet]]:
    """Split question sets into iteration and held-out sets by forecast_due_date.

    The most recent n_held_out sets (by date descending) go to held-out.
    """
    if n_held_out < 0:
        raise ValueError(f"n_held_out must be non-negative, got {n_held_out}")
    if n_held_out >= len(question_sets):
        return [], list(question_sets)

    sorted_qs = sorted(question_sets, key=lambda qs: qs.forecast_due_date)
    split_point = len(sorted_qs) - n_held_out
    iteration_set = sorted_qs[:split_point]
    held_out_set = sorted_qs[split_point:]
    return iteration_set, held_out_set


def run_eval(
    forecaster: Callable[[Question], float],
    n_held_out: int = 2,
) -> ScoringResult:
    """Run the full evaluation pipeline."""
    question_sets, resolved = load_data()
    iteration_set, _held_out = split_held_out(question_sets, n_held_out)

    resolutions_by_id = {q.id: q for q in resolved}
    iteration_resolved = join_resolved_questions(
        iteration_set,
        {q_id: Resolution(id=q_id, outcome=r.outcome, resolution_date=r.resolution_date)
         for q_id, r in resolutions_by_id.items()},
    )

    forecasts: dict[str, float] = {}
    for q in iteration_resolved:
        question = Question(
            id=q.id,
            source=q.source,
            question=q.question,
            background=q.background,
            resolution_criteria=q.resolution_criteria,
            freeze_datetime=q.freeze_datetime,
            freeze_datetime_value=q.freeze_datetime_value,
            resolution_dates=q.resolution_dates,
            url=q.url,
            combination_of=q.combination_of,
        )
        forecasts[q.id] = forecaster(question)

    result = score_forecasts(forecasts, iteration_resolved)
    _print_results(result)
    return result


def _print_results(result: ScoringResult) -> None:
    print("=" * 50)
    print("ForecastBench Backtest Results")
    print("=" * 50)
    print(f"Dataset:  Brier={result.dataset_brier:.4f}  Index={result.dataset_index:.1f}%  (n={result.n_dataset})")
    print(f"Market:   Brier={result.market_brier:.4f}  Index={result.market_index:.1f}%  (n={result.n_market})")
    print(f"Overall:  Brier={result.overall_brier:.4f}  Index={result.overall_index:.1f}%")
    print(f"Missing forecasts (defaulted to 0.5): {result.n_missing}")
    print("=" * 50)


def _model_slug(model: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_-]", "_", model)


def _cache_dir(model: str) -> Path:
    return Path(".cache") / "forecasts" / _model_slug(model)


def _load_cached_forecast(model: str, question_id: str) -> float | None:
    path = _cache_dir(model) / f"{question_id}.json"
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text())
        return float(data["probability"])
    except (json.JSONDecodeError, KeyError, ValueError):
        return None


def _save_cached_forecast(model: str, question_id: str, probability: float) -> None:
    cache = _cache_dir(model)
    cache.mkdir(parents=True, exist_ok=True)
    payload = {
        "probability": probability,
        "model": model,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    (cache / f"{question_id}.json").write_text(json.dumps(payload))


def _format_duration(seconds: float) -> str:
    if not math.isfinite(seconds):
        return "--"
    if seconds < 60:
        return f"{int(seconds)}s"
    minutes = int(seconds) // 60
    secs = int(seconds) % 60
    if minutes < 60:
        return f"{minutes}m{secs:02d}s"
    hours = minutes // 60
    mins = minutes % 60
    return f"{hours}h{mins:02d}m"


def _resolved_to_question(q: ResolvedQuestion) -> Question:
    return Question(
        id=q.id,
        source=q.source,
        question=q.question,
        background=q.background,
        resolution_criteria=q.resolution_criteria,
        freeze_datetime=q.freeze_datetime,
        freeze_datetime_value=q.freeze_datetime_value,
        resolution_dates=q.resolution_dates,
        url=q.url,
        combination_of=q.combination_of,
    )


async def run_baseline_eval(
    forecaster: AsyncForecaster,
    model: str = "",
    n_held_out: int = 2,
) -> ScoringResult:
    """Run async evaluation with concurrency, caching, and progress logging."""
    try:
        concurrency = max(1, int(os.environ.get("FORECAST_CONCURRENCY", "10")))
    except (ValueError, TypeError):
        concurrency = 10
    if not model:
        model = os.environ.get("FORECAST_MODEL", "claude-sonnet-4-20250514")

    question_sets, resolved = load_data()
    iteration_set, _held_out = split_held_out(question_sets, n_held_out)

    resolutions_by_id = {q.id: q for q in resolved}
    iteration_resolved = join_resolved_questions(
        iteration_set,
        {q_id: Resolution(id=q_id, outcome=r.outcome, resolution_date=r.resolution_date)
         for q_id, r in resolutions_by_id.items()},
    )

    forecasts: dict[str, float] = {}
    to_forecast: list[ResolvedQuestion] = []

    for q in iteration_resolved:
        cached = _load_cached_forecast(model, q.id)
        if cached is not None:
            forecasts[q.id] = cached
        else:
            to_forecast.append(q)

    total = len(iteration_resolved)
    done = len(forecasts)
    if done > 0:
        print(f"Loaded {done} cached forecasts, {len(to_forecast)} remaining")

    semaphore = asyncio.Semaphore(concurrency)
    start_time = time.monotonic()

    async def forecast_one(q: ResolvedQuestion) -> tuple[str, float]:
        async with semaphore:
            question = _resolved_to_question(q)
            cutoff: CutoffEnvironment | None = None
            if q.freeze_datetime:
                cutoff = CutoffEnvironment(freeze_datetime=q.freeze_datetime)
            prob = await forecaster(question, cutoff)
            _save_cached_forecast(model, q.id, prob)
            return q.id, prob

    completed = 0
    log_interval = 50
    tasks = [asyncio.create_task(forecast_one(q)) for q in to_forecast]

    for coro in asyncio.as_completed(tasks):
        q_id, prob = await coro
        forecasts[q_id] = prob
        done += 1
        completed += 1

        if completed % log_interval == 0 or completed == len(to_forecast):
            elapsed = time.monotonic() - start_time
            pct = done / total * 100 if total > 0 else 100.0
            if completed > 0:
                eta = elapsed / completed * (len(to_forecast) - completed)
            else:
                eta = 0.0
            print(
                f"[{done}/{total}] {pct:.1f}%"
                f" — elapsed {_format_duration(elapsed)}"
                f" — ETA {_format_duration(eta)}"
            )

    result = score_forecasts(forecasts, iteration_resolved)
    _print_results(result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="ForecastBench evaluation")
    parser.add_argument("--baseline", action="store_true", help="Run baseline LLM agent")
    parser.add_argument("--analyze", action="store_true", help="Run error analysis after scoring")
    args = parser.parse_args()

    if args.baseline:
        from baseline_agent import aforecast
        model = os.environ.get("FORECAST_MODEL", "claude-sonnet-4-20250514")
        asyncio.run(run_baseline_eval(aforecast, model=model))

        if args.analyze:
            from analyze import run_analysis, save_analysis

            question_sets, resolved = load_data()
            iteration_set, _ = split_held_out(question_sets)
            resolutions_by_id = {q.id: q for q in resolved}
            iteration_resolved = join_resolved_questions(
                iteration_set,
                {q_id: Resolution(id=q_id, outcome=r.outcome, resolution_date=r.resolution_date)
                 for q_id, r in resolutions_by_id.items()},
            )

            forecasts: dict[str, float] = {}
            for q in iteration_resolved:
                cached = _load_cached_forecast(model, q.id)
                if cached is not None:
                    forecasts[q.id] = cached

            analysis = run_analysis(forecasts, iteration_resolved)
            output_dir = Path("results")
            save_analysis(analysis, output_dir)
            print(f"Analysis saved to {output_dir}/analysis.json and {output_dir}/analysis.md")
    else:
        from dummy_forecaster import forecast
        run_eval(forecast)


if __name__ == "__main__":
    main()
