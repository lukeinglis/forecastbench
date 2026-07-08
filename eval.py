"""Evaluation entrypoint for ForecastBench backtest harness."""

from __future__ import annotations

import asyncio
import inspect
import json
import os
import re
from pathlib import Path
from typing import Protocol, Union

os.environ.setdefault("LITELLM_LOG", "ERROR")
import litellm  # noqa: E402

litellm.suppress_debug_info = True

from fetch_data import Question, QuestionSet, Resolution, ResolvedQuestion, load_data, join_resolved_questions  # noqa: E402
from score import ScoringResult, score_forecasts  # noqa: E402

CACHE_DIR = Path(".cache/forecasts")


MARKET_SOURCES = {"metaculus", "polymarket", "manifold", "infer"}


class SyncForecaster(Protocol):
    def __call__(self, question: Question) -> float: ...


class AsyncForecaster(Protocol):
    async def __call__(self, question: Question) -> float: ...


Forecaster = Union[SyncForecaster, AsyncForecaster]


def _has_multi_horizon(question: Question) -> bool:
    if question.source.lower() in MARKET_SOURCES:
        return False
    rd = question.resolution_dates
    return isinstance(rd, list) and len(rd) > 0


def _expand_resolved_for_horizons(
    resolved: list[ResolvedQuestion],
) -> list[ResolvedQuestion]:
    expanded: list[ResolvedQuestion] = []
    for rq in resolved:
        if rq.source.lower() in MARKET_SOURCES:
            expanded.append(rq)
            continue
        rd = rq.resolution_dates
        if not isinstance(rd, list) or len(rd) == 0:
            expanded.append(rq)
            continue
        for date_str in rd:
            composite_id = f"{rq.id}_{date_str}"
            expanded.append(
                ResolvedQuestion(
                    id=composite_id,
                    source=rq.source,
                    question=rq.question,
                    background=rq.background,
                    resolution_criteria=rq.resolution_criteria,
                    freeze_datetime=rq.freeze_datetime,
                    freeze_datetime_value=rq.freeze_datetime_value,
                    resolution_dates=rq.resolution_dates,
                    url=rq.url,
                    combination_of=rq.combination_of,
                    source_intro=rq.source_intro,
                    freeze_datetime_value_explanation=rq.freeze_datetime_value_explanation,
                    market_info_open_datetime=rq.market_info_open_datetime,
                    market_info_close_datetime=rq.market_info_close_datetime,
                    market_info_resolution_criteria=rq.market_info_resolution_criteria,
                    outcome=rq.outcome,
                    resolution_date=date_str,
                    forecast_due_date=rq.forecast_due_date,
                    question_set=rq.question_set,
                )
            )
    return expanded


def is_async_forecaster(forecaster: Forecaster) -> bool:
    return inspect.iscoroutinefunction(forecaster)


def _model_slug() -> str:
    raw = os.getenv("FORECAST_MODEL", "default")
    return re.sub(r"[^\w\-.]", "_", raw)


def _cache_path_for(model_slug: str, question_id: str) -> Path:
    safe_qid = re.sub(r"[^\w\-.]", "_", question_id)
    return CACHE_DIR / model_slug / f"{safe_qid}.json"


def _read_cache(model_slug: str, question_id: str) -> float | None:
    path = _cache_path_for(model_slug, question_id)
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text())
        return float(data["probability"])
    except (json.JSONDecodeError, KeyError, ValueError, TypeError):
        return None


def _write_cache(model_slug: str, question_id: str, probability: float) -> None:
    path = _cache_path_for(model_slug, question_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({
        "probability": probability,
        "model": model_slug,
        "question_id": question_id,
    }))


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


def _build_question(q: Question | ResolvedQuestion) -> Question:
    """Build a Question from a ResolvedQuestion or Question-like object."""
    return Question(
        id=q.id,
        source=q.source,
        question=q.question,
        background=getattr(q, "background", ""),
        resolution_criteria=getattr(q, "resolution_criteria", ""),
        freeze_datetime=getattr(q, "freeze_datetime", None),
        freeze_datetime_value=getattr(q, "freeze_datetime_value", None),
        resolution_dates=getattr(q, "resolution_dates", None),
        url=getattr(q, "url", None),
        combination_of=getattr(q, "combination_of", None),
        source_intro=getattr(q, "source_intro", None),
        freeze_datetime_value_explanation=getattr(q, "freeze_datetime_value_explanation", None),
        market_info_open_datetime=getattr(q, "market_info_open_datetime", None),
        market_info_close_datetime=getattr(q, "market_info_close_datetime", None),
        market_info_resolution_criteria=getattr(q, "market_info_resolution_criteria", None),
    )


async def run_eval(
    forecaster: Forecaster,
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

    questions = [_build_question(q) for q in iteration_resolved]
    model_slug = _model_slug()

    if is_async_forecaster(forecaster):
        forecasts = await _run_async(forecaster, questions, model_slug)  # type: ignore[arg-type]
    else:
        forecasts = _run_sync(forecaster, questions, model_slug)  # type: ignore[arg-type]

    expanded_resolved = _expand_resolved_for_horizons(iteration_resolved)
    result = score_forecasts(forecasts, expanded_resolved)
    _print_results(result)
    return result


def _run_sync(
    forecaster: SyncForecaster,
    questions: list[Question],
    model_slug: str,
) -> dict[str, float]:
    forecasts: dict[str, float] = {}
    for q in questions:
        if _has_multi_horizon(q):
            for date_str in q.resolution_dates:
                composite_key = f"{q.id}_{date_str}"
                cached = _read_cache(model_slug, composite_key)
                if cached is not None:
                    forecasts[composite_key] = cached
                    continue
                prob = forecaster(q, resolution_date=date_str)  # type: ignore[call-arg]
                forecasts[composite_key] = prob
                _write_cache(model_slug, composite_key, prob)
        else:
            cached = _read_cache(model_slug, q.id)
            if cached is not None:
                forecasts[q.id] = cached
                continue
            prob = forecaster(q)
            forecasts[q.id] = prob
            _write_cache(model_slug, q.id, prob)
    return forecasts


async def _run_async(
    forecaster: AsyncForecaster,
    questions: list[Question],
    model_slug: str,
) -> dict[str, float]:
    from tqdm.asyncio import tqdm_asyncio

    concurrency = max(1, int(os.getenv("FORECAST_CONCURRENCY", "10")))
    semaphore = asyncio.Semaphore(concurrency)

    async def _forecast_one(
        q: Question,
        cache_key: str,
        resolution_date: str | None = None,
    ) -> tuple[str, float]:
        cached = _read_cache(model_slug, cache_key)
        if cached is not None:
            return cache_key, cached
        async with semaphore:
            try:
                if resolution_date is not None:
                    prob = await forecaster(q, resolution_date=resolution_date)  # type: ignore[call-arg]
                else:
                    prob = await forecaster(q)
            except Exception:
                return cache_key, 0.5
        _write_cache(model_slug, cache_key, prob)
        return cache_key, prob

    tasks = []
    for q in questions:
        if _has_multi_horizon(q):
            for date_str in q.resolution_dates:
                composite_key = f"{q.id}_{date_str}"
                tasks.append(_forecast_one(q, composite_key, resolution_date=date_str))
        else:
            tasks.append(_forecast_one(q, q.id))

    if tasks:
        results = await tqdm_asyncio.gather(*tasks, desc="Forecasting")
    else:
        results = []

    return {qid: prob for qid, prob in results}


def _print_results(result: ScoringResult) -> None:
    print("=" * 50)
    print("ForecastBench Backtest Results")
    print("=" * 50)
    print(f"Dataset:  Brier={result.dataset_brier:.4f}  Index={result.dataset_index:.1f}%  (n={result.n_dataset})")
    print(f"Market:   Brier={result.market_brier:.4f}  Index={result.market_index:.1f}%  (n={result.n_market})")
    print(f"Overall:  Brier={result.overall_brier:.4f}  Index={result.overall_index:.1f}%")
    print(f"Missing forecasts (defaulted to 0.5): {result.n_missing}")
    print("=" * 50)


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="ForecastBench evaluation")
    parser.add_argument(
        "--agent",
        choices=["dummy", "baseline"],
        default="dummy",
        help="Forecaster agent to use (default: dummy)",
    )
    args = parser.parse_args()

    if args.agent == "baseline":
        from baseline_agent import aforecast
        forecaster: Forecaster = aforecast
    else:
        from dummy_forecaster import forecast
        forecaster = forecast

    result = asyncio.run(run_eval(forecaster))

    if args.agent != "dummy":
        _run_analysis(result)


def _run_analysis(result: ScoringResult) -> None:
    from analyze import (
        analyze_by_source,
        analyze_calibration,
        analyze_biases,
        print_analysis,
        save_analysis,
    )
    from fetch_data import load_data

    question_sets, resolved = load_data()
    iteration_set, _ = split_held_out(question_sets)

    resolutions_by_id = {q.id: q for q in resolved}
    from fetch_data import Resolution, join_resolved_questions as _join

    iteration_resolved = _join(
        iteration_set,
        {q_id: Resolution(id=q_id, outcome=r.outcome, resolution_date=r.resolution_date)
         for q_id, r in resolutions_by_id.items()},
    )

    model_slug = _model_slug()
    cache_dir = Path(f".cache/forecasts/{model_slug}")
    forecasts: dict[str, float] = {}
    if cache_dir.exists():
        for p in cache_dir.glob("*.json"):
            cached = _read_cache(model_slug, p.stem)
            if cached is not None:
                forecasts[p.stem] = cached

    analysis = {
        "by_source": analyze_by_source(forecasts, iteration_resolved),
        "calibration": analyze_calibration(forecasts, iteration_resolved),
        "biases": analyze_biases(forecasts, iteration_resolved),
    }

    print_analysis(analysis)

    analysis_path = Path(f".cache/analysis/{model_slug}/analysis.json")
    save_analysis(analysis, analysis_path)
    print(f"\nAnalysis saved to {analysis_path}")


if __name__ == "__main__":
    main()
