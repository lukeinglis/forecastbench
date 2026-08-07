"""Scoring — re-exports from forecastbench-parity."""

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
    score_forecasts,
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
