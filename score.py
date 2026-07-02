"""Brier score and Brier Index scoring for ForecastBench."""

from __future__ import annotations

import math
from dataclasses import dataclass

from fetch_data import ResolvedQuestion


@dataclass
class ScoringResult:
    dataset_brier: float
    dataset_index: float
    market_brier: float
    market_index: float
    overall_brier: float
    overall_index: float
    n_dataset: int
    n_market: int
    n_missing: int


def _validate_forecast(forecast: float) -> float:
    if math.isnan(forecast) or math.isinf(forecast):
        raise ValueError(f"Forecast must be finite, got {forecast}")
    if forecast < 0.0 or forecast > 1.0:
        raise ValueError(f"Forecast must be in [0, 1], got {forecast}")
    return forecast


def _validate_outcome(outcome: int) -> int:
    if outcome not in (0, 1):
        raise ValueError(f"Outcome must be 0 or 1, got {outcome}")
    return outcome


def brier_score(forecast: float, outcome: int) -> float:
    """Single-question Brier score = (forecast - outcome)^2. Range [0, 1]."""
    _validate_forecast(forecast)
    _validate_outcome(outcome)
    return (forecast - outcome) ** 2


def mean_brier_score(pairs: list[tuple[float, int]]) -> float:
    """Arithmetic mean of individual Brier scores."""
    if not pairs:
        raise ValueError("Cannot compute mean Brier score of empty list")
    total = sum(brier_score(f, o) for f, o in pairs)
    return total / len(pairs)


def brier_index(mean_bs: float) -> float:
    """Transform mean Brier score to Brier Index. Applied AFTER averaging, never per-question."""
    return (1.0 - math.sqrt(mean_bs)) * 100.0


def _is_market_question(q: ResolvedQuestion) -> bool:
    source_lower = q.source.lower()
    return any(s in source_lower for s in ("metaculus", "polymarket", "manifold", "infer"))


def score_forecasts(
    forecasts: dict[str, float],
    resolved: list[ResolvedQuestion],
) -> ScoringResult:
    """Score forecasts against resolved questions.

    Missing forecasts default to 0.5 per ForecastBench rules.
    """
    if not resolved:
        raise ValueError("No resolved questions to score")

    dataset_pairs: list[tuple[float, int]] = []
    market_pairs: list[tuple[float, int]] = []
    n_missing = 0

    for q in resolved:
        if q.id in forecasts:
            f = forecasts[q.id]
        else:
            f = 0.5
            n_missing += 1

        _validate_forecast(f)
        _validate_outcome(q.outcome)

        if _is_market_question(q):
            market_pairs.append((f, q.outcome))
        else:
            dataset_pairs.append((f, q.outcome))

    ds_brier = mean_brier_score(dataset_pairs) if dataset_pairs else 0.0
    ds_index = brier_index(ds_brier) if dataset_pairs else 0.0
    mk_brier = mean_brier_score(market_pairs) if market_pairs else 0.0
    mk_index = brier_index(mk_brier) if market_pairs else 0.0

    components = []
    if dataset_pairs:
        components.append(ds_brier)
    if market_pairs:
        components.append(mk_brier)

    if components:
        overall_bs = sum(components) / len(components)
    else:
        overall_bs = 0.0

    return ScoringResult(
        dataset_brier=ds_brier,
        dataset_index=ds_index,
        market_brier=mk_brier,
        market_index=mk_index,
        overall_brier=overall_bs,
        overall_index=brier_index(overall_bs),
        n_dataset=len(dataset_pairs),
        n_market=len(market_pairs),
        n_missing=n_missing,
    )
