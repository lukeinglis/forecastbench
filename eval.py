"""Evaluation entrypoint for ForecastBench backtest harness."""

from __future__ import annotations

import asyncio
import inspect
import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, NamedTuple, Protocol, Union

os.environ.setdefault("LITELLM_LOG", "ERROR")
import litellm  # noqa: E402

litellm.suppress_debug_info = True

from fetch_data import MARKET_SOURCES, Question, QuestionSet, Resolution, ResolvedQuestion, load_data, join_resolved_questions  # noqa: E402
from score import ScoringResult, score_forecasts  # noqa: E402

CACHE_DIR = Path(".cache/forecasts")
RESULTS_DIR = Path("results")


class SyncForecaster(Protocol):
    def __call__(self, question: Question, resolution_date: str | None = None) -> float: ...


class AsyncForecaster(Protocol):
    async def __call__(self, question: Question, resolution_date: str | None = None) -> float: ...


Forecaster = Union[SyncForecaster, AsyncForecaster]

MultiForecaster = Any


class EvalResult(NamedTuple):
    scoring: ScoringResult
    forecasts: dict[str, float]
    resolved: list[ResolvedQuestion]
    model_slug: str


def _has_multi_horizon(question: Question) -> bool:
    if question.source.lower() in MARKET_SOURCES:
        return False
    rd = question.resolution_dates
    return isinstance(rd, list) and any(d for d in rd)


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


def save_result(
    result: ScoringResult,
    forecasts: dict[str, float],
    model_slug: str,
    question_sets_used: list[str],
    n_held_out: int,
) -> Path:
    """Save run result to results/{timestamp}_{model_slug}.json."""
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    payload = {
        "timestamp": timestamp,
        "model_slug": model_slug,
        "scoring_result": {
            "dataset_brier": result.dataset_brier,
            "dataset_index": result.dataset_index,
            "market_brier": result.market_brier,
            "market_index": result.market_index,
            "overall_brier": result.overall_brier,
            "overall_index": result.overall_index,
            "n_dataset": result.n_dataset,
            "n_market": result.n_market,
            "n_missing": result.n_missing,
            "difficulty_adjusted": result.difficulty_adjusted,
        },
        "forecasts": forecasts,
        "metadata": {
            "n_questions": result.n_dataset + result.n_market,
            "n_held_out": n_held_out,
            "question_sets_used": question_sets_used,
        },
    }
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    path = RESULTS_DIR / f"{timestamp}_{model_slug}.json"
    path.write_text(json.dumps(payload, indent=2))
    return path


def load_previous_results(results_dir: Path | None = None) -> list[dict[str, object]]:
    """Load all previously saved results for building peer pools."""
    if results_dir is None:
        results_dir = RESULTS_DIR
    if not results_dir.exists():
        return []
    results: list[dict[str, object]] = []
    for p in sorted(results_dir.glob("*.json")):
        try:
            data = json.loads(p.read_text())
            results.append(data)
        except (json.JSONDecodeError, KeyError):
            continue
    return results


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
        forecast_due_date=getattr(q, "forecast_due_date", None),
    )


async def run_eval(
    forecaster: Forecaster,
    n_held_out: int = 2,
    raw: bool = False,
    multi_forecaster: MultiForecaster | None = None,
    async_multi_forecaster: MultiForecaster | None = None,
) -> EvalResult:
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
        forecasts = await _run_async(forecaster, questions, model_slug, async_multi_forecaster=async_multi_forecaster)  # type: ignore[arg-type]
    else:
        forecasts = _run_sync(forecaster, questions, model_slug, multi_forecaster=multi_forecaster)  # type: ignore[arg-type]

    expanded_resolved = _expand_resolved_for_horizons(iteration_resolved)

    all_forecasts: dict[str, dict[str, float]] | None = None
    if not raw:
        previous = load_previous_results()
        if len(previous) >= 2:
            all_forecasts = {}
            for prev in previous:
                slug = prev["model_slug"]
                all_forecasts[str(slug)] = prev["forecasts"]  # type: ignore[assignment]
            print(f"Difficulty adjustment: using {len(all_forecasts)} peer forecasters from results/")
        else:
            print(f"Difficulty adjustment: skipped (need 2+ results, found {len(previous)})")

    result = score_forecasts(
        forecasts, expanded_resolved,
        difficulty_adjusted=not raw,
        all_forecasts=all_forecasts,
    )
    _print_results(result)

    question_sets_used = [qs.forecast_due_date for qs in iteration_set]
    result_path = save_result(result, forecasts, model_slug, question_sets_used, n_held_out)
    print(f"Results saved to {result_path}")

    return EvalResult(scoring=result, forecasts=forecasts, resolved=iteration_resolved, model_slug=model_slug)


def _run_sync(
    forecaster: SyncForecaster,
    questions: list[Question],
    model_slug: str,
    multi_forecaster: MultiForecaster | None = None,
) -> dict[str, float]:
    forecasts: dict[str, float] = {}
    for q in questions:
        if _has_multi_horizon(q) and multi_forecaster is not None:
            dates = [d for d in q.resolution_dates if d]
            uncached_dates: list[str] = []
            for date_str in dates:
                composite_key = f"{q.id}_{date_str}"
                cached = _read_cache(model_slug, composite_key)
                if cached is not None:
                    forecasts[composite_key] = cached
                else:
                    uncached_dates.append(date_str)
            if not uncached_dates:
                continue
            try:
                probs = multi_forecaster(q, resolution_dates=uncached_dates)
            except Exception:
                probs = [0.5] * len(uncached_dates)
            for date_str, prob in zip(uncached_dates, probs):
                composite_key = f"{q.id}_{date_str}"
                forecasts[composite_key] = prob
                _write_cache(model_slug, composite_key, prob)
        elif _has_multi_horizon(q):
            dates = [d for d in q.resolution_dates if d]
            for date_str in dates:
                composite_key = f"{q.id}_{date_str}"
                cached = _read_cache(model_slug, composite_key)
                if cached is not None:
                    forecasts[composite_key] = cached
                    continue
                try:
                    prob = forecaster(q, resolution_date=date_str)
                except Exception:
                    prob = 0.5
                forecasts[composite_key] = prob
                _write_cache(model_slug, composite_key, prob)
        else:
            cached = _read_cache(model_slug, q.id)
            if cached is not None:
                forecasts[q.id] = cached
                continue
            try:
                prob = forecaster(q)
            except Exception:
                prob = 0.5
            forecasts[q.id] = prob
            _write_cache(model_slug, q.id, prob)
    return forecasts


async def _run_async(
    forecaster: AsyncForecaster,
    questions: list[Question],
    model_slug: str,
    async_multi_forecaster: MultiForecaster | None = None,
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
                    prob = await forecaster(q, resolution_date=resolution_date)
                else:
                    prob = await forecaster(q)
            except Exception:
                return cache_key, 0.5
        _write_cache(model_slug, cache_key, prob)
        return cache_key, prob

    async def _forecast_multi_horizon(
        q: Question,
        dates: list[str],
    ) -> list[tuple[str, float]]:
        cached_results: dict[str, float] = {}
        uncached_dates: list[str] = []
        for date_str in dates:
            composite_key = f"{q.id}_{date_str}"
            cached = _read_cache(model_slug, composite_key)
            if cached is not None:
                cached_results[composite_key] = cached
            else:
                uncached_dates.append(date_str)

        if not uncached_dates:
            return list(cached_results.items())

        assert async_multi_forecaster is not None
        async with semaphore:
            try:
                probs = await async_multi_forecaster(q, resolution_dates=uncached_dates)
            except Exception:
                probs = [0.5] * len(uncached_dates)

        results = list(cached_results.items())
        for date_str, prob in zip(uncached_dates, probs):
            composite_key = f"{q.id}_{date_str}"
            _write_cache(model_slug, composite_key, prob)
            results.append((composite_key, prob))
        return results

    single_tasks = []
    multi_tasks = []
    for q in questions:
        if _has_multi_horizon(q) and async_multi_forecaster is not None:
            dates = [d for d in q.resolution_dates if d]
            multi_tasks.append(_forecast_multi_horizon(q, dates))
        elif _has_multi_horizon(q):
            dates = [d for d in q.resolution_dates if d]
            for date_str in dates:
                composite_key = f"{q.id}_{date_str}"
                single_tasks.append(_forecast_one(q, composite_key, resolution_date=date_str))
        else:
            single_tasks.append(_forecast_one(q, q.id))

    all_results: list[tuple[str, float]] = []
    if single_tasks:
        single_results = await tqdm_asyncio.gather(*single_tasks, desc="Forecasting (market)")
        all_results.extend(single_results)
    if multi_tasks:
        multi_results = await tqdm_asyncio.gather(*multi_tasks, desc="Forecasting (dataset)")
        for result_list in multi_results:
            all_results.extend(result_list)

    return {qid: prob for qid, prob in all_results}


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
    parser.add_argument(
        "--raw",
        action="store_true",
        help="Disable difficulty adjustment, use raw Brier scores",
    )
    args = parser.parse_args()

    if args.agent == "baseline":
        from baseline_agent import aforecast, aforecast_multi
        forecaster: Forecaster = aforecast
        eval_result = asyncio.run(run_eval(forecaster, raw=args.raw, async_multi_forecaster=aforecast_multi))
    else:
        from dummy_forecaster import forecast
        forecaster = forecast
        eval_result = asyncio.run(run_eval(forecaster, raw=args.raw))

    if args.agent != "dummy":
        _run_analysis(eval_result.forecasts, eval_result.resolved, eval_result.model_slug)


def _run_analysis(
    forecasts: dict[str, float],
    resolved: list[ResolvedQuestion],
    model_slug: str,
) -> None:
    from analyze import (
        analyze_by_source,
        analyze_calibration,
        analyze_biases,
        print_analysis,
        save_analysis,
    )

    analysis = {
        "by_source": analyze_by_source(forecasts, resolved),
        "calibration": analyze_calibration(forecasts, resolved),
        "biases": analyze_biases(forecasts, resolved),
    }

    print_analysis(analysis)

    analysis_path = Path(f".cache/analysis/{model_slug}/analysis.json")
    save_analysis(analysis, analysis_path)
    print(f"\nAnalysis saved to {analysis_path}")


if __name__ == "__main__":
    main()
