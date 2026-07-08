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

from fetch_data import Question, QuestionSet, Resolution, load_data, join_resolved_questions  # noqa: E402
from score import ScoringResult, score_forecasts  # noqa: E402

CACHE_DIR = Path(".cache/forecasts")


class SyncForecaster(Protocol):
    def __call__(self, question: Question) -> float: ...


class AsyncForecaster(Protocol):
    async def __call__(self, question: Question) -> float: ...


Forecaster = Union[SyncForecaster, AsyncForecaster]


def is_async_forecaster(forecaster: Forecaster) -> bool:
    return inspect.iscoroutinefunction(forecaster)


def _model_slug() -> str:
    raw = os.getenv("FORECASTER_MODEL", "default")
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


def _build_question(q: Question | object) -> Question:
    """Build a Question from a ResolvedQuestion or Question-like object."""
    return Question(
        id=q.id,  # type: ignore[attr-defined]
        source=q.source,  # type: ignore[attr-defined]
        question=q.question,  # type: ignore[attr-defined]
        background=getattr(q, "background", ""),
        resolution_criteria=getattr(q, "resolution_criteria", ""),
        freeze_datetime=getattr(q, "freeze_datetime", None),
        freeze_datetime_value=getattr(q, "freeze_datetime_value", None),
        resolution_dates=getattr(q, "resolution_dates", None),
        url=getattr(q, "url", None),
        combination_of=getattr(q, "combination_of", None),
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

    result = score_forecasts(forecasts, iteration_resolved)
    _print_results(result)
    return result


def _run_sync(
    forecaster: SyncForecaster,
    questions: list[Question],
    model_slug: str,
) -> dict[str, float]:
    forecasts: dict[str, float] = {}
    for q in questions:
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

    async def _forecast_one(q: Question) -> tuple[str, float]:
        cached = _read_cache(model_slug, q.id)
        if cached is not None:
            return q.id, cached
        async with semaphore:
            try:
                prob = await forecaster(q)
            except Exception:
                return q.id, 0.5
        _write_cache(model_slug, q.id, prob)
        return q.id, prob

    tasks = [_forecast_one(q) for q in questions]
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
    from dummy_forecaster import forecast
    asyncio.run(run_eval(forecast))


if __name__ == "__main__":
    main()
