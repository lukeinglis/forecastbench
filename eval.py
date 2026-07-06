"""Evaluation entrypoint for ForecastBench backtest harness."""

from __future__ import annotations

from typing import Callable, Protocol

from fetch_data import Question, QuestionSet, Resolution, load_data, join_resolved_questions
from score import ScoringResult, score_forecasts
from cutoff import CutoffEnvironment


class Forecaster(Protocol):
    def __call__(self, question: Question) -> float: ...


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


def run_baseline_eval(
    model: str | None = None,
    n_held_out: int = 2,
) -> "RunResult":  # noqa: F821
    """Run baseline LLM agent evaluation with cutoff enforcement and error analysis."""
    import baseline_agent
    from analyze import RunResult, analyze_run, save_results, print_summary

    env = CutoffEnvironment()
    question_sets, resolved = load_data()
    iteration_set, _held_out = split_held_out(question_sets, n_held_out)

    resolutions_by_id = {q.id: q for q in resolved}
    iteration_resolved = join_resolved_questions(
        iteration_set,
        {q_id: Resolution(id=q_id, outcome=r.outcome, resolution_date=r.resolution_date)
         for q_id, r in resolutions_by_id.items()},
    )

    used_model = model or baseline_agent.DEFAULT_MODEL
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
        context = env.prepare_context(question, q.forecast_due_date)
        today_date = env.get_today_date(context)
        forecasts[q.id] = baseline_agent.forecast(question, model=model, today_date=today_date)

    result: RunResult = analyze_run(forecasts, iteration_resolved, used_model)
    filepath = save_results(result)
    print(f"\nResults saved to {filepath}")
    print_summary(result)

    scoring = score_forecasts(forecasts, iteration_resolved)
    _print_results(scoring)

    return result


def main() -> None:
    from dummy_forecaster import forecast
    run_eval(forecast)


if __name__ == "__main__":
    main()
