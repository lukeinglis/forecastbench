"""Scoring wrapper — re-exports from forecastbench-parity with equal-weight overall formula fix."""

from forecastbench_parity.score import (
    AdjustmentResult,
    ScoringResult,
    _build_market_effects,
    _estimate_difficulty_effects_ols,
    _is_market_question,
    _validate_forecast,
    _validate_outcome,
    adjust_for_difficulty,
    bootstrap_ci,
    brier_index,
    brier_score,
    brier_skill_score,
    mean_brier_score,
    murphy_decomposition,
)
from forecastbench_parity.score import score_forecasts as _parity_score_forecasts

from fetch_data import ResolvedQuestion


def score_forecasts(
    forecasts: dict[str, float],
    resolved: list[ResolvedQuestion],
    *,
    difficulty_adjusted: bool = True,
    all_forecasts: dict[str, dict[str, float]] | None = None,
    market_weight: float = 1.0,
    market_forecasts: dict[str, float] | None = None,
) -> ScoringResult:
    """Score forecasts with equal-weight overall formula: (dataset + market) / 2."""
    result = _parity_score_forecasts(
        forecasts, resolved,
        difficulty_adjusted=difficulty_adjusted,
        all_forecasts=all_forecasts,
        market_weight=market_weight,
        market_forecasts=market_forecasts,
    )
    if result.n_dataset > 0 and result.n_market > 0:
        overall_bs = (result.dataset_brier + result.market_brier) / 2.0
    elif result.n_dataset > 0:
        overall_bs = result.dataset_brier
    elif result.n_market > 0:
        overall_bs = result.market_brier
    else:
        overall_bs = 0.0

    return ScoringResult(
        dataset_brier=result.dataset_brier,
        dataset_index=result.dataset_index,
        market_brier=result.market_brier,
        market_index=result.market_index,
        overall_brier=overall_bs,
        overall_index=brier_index(overall_bs),
        n_dataset=result.n_dataset,
        n_market=result.n_market,
        n_missing=result.n_missing,
        difficulty_adjusted=result.difficulty_adjusted,
        question_effects=result.question_effects,
    )


__all__ = [
    "AdjustmentResult",
    "ScoringResult",
    "_build_market_effects",
    "_estimate_difficulty_effects_ols",
    "_is_market_question",
    "_validate_forecast",
    "_validate_outcome",
    "adjust_for_difficulty",
    "bootstrap_ci",
    "brier_index",
    "brier_score",
    "brier_skill_score",
    "mean_brier_score",
    "murphy_decomposition",
    "score_forecasts",
]
