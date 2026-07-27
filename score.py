"""Brier score and Brier Index scoring for ForecastBench.

Implements the difficulty-adjusted Brier score from:
  Kucinskas, Bastani & Karger, "ForecastBench: Updated Ranking Methodology"
  https://www.forecastbench.org/assets/pdfs/forecastbench_updated_methodology.pdf
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import NamedTuple
from uuid import uuid4

from fetch_data import MARKET_SOURCES, ResolvedQuestion
from logging_config import get_logger

logger = get_logger("score")


class AdjustmentResult(NamedTuple):
    adjusted_scores: dict[str, dict[str, float]]
    question_effects: dict[str, float]


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
    difficulty_adjusted: bool = False
    question_effects: dict[str, float] = field(default_factory=dict)


def _validate_forecast(forecast: float) -> float:
    if math.isnan(forecast) or math.isinf(forecast):
        logger.warning("invalid_forecast", value=forecast, reason="not_finite")
        raise ValueError(f"Forecast must be finite, got {forecast}")
    if forecast < 0.0 or forecast > 1.0:
        logger.warning("invalid_forecast", value=forecast, reason="out_of_range")
        raise ValueError(f"Forecast must be in [0, 1], got {forecast}")
    return forecast


def _validate_outcome(outcome: int) -> int:
    if outcome not in (0, 1):
        logger.warning("invalid_outcome", value=outcome)
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
    if mean_bs < 0:
        raise ValueError(f"mean_bs must be non-negative, got {mean_bs}")
    index_value = (1.0 - math.sqrt(mean_bs)) * 100.0
    logger.debug("brier_index", mean_brier=round(mean_bs, 6), index_value=round(index_value, 2))
    return index_value


def brier_skill_score(forecaster_brier: float, reference_brier: float = 0.25) -> float:
    """Brier Skill Score: improvement over reference forecaster.

    BSS = 1 - (forecaster_brier / reference_brier)
    Positive = better than reference, 0 = same, negative = worse.
    Reference defaults to 0.25 (always-0.5 forecaster).
    """
    if reference_brier == 0:
        return 0.0
    return 1.0 - (forecaster_brier / reference_brier)


def bootstrap_ci(
    pairs: list[tuple[float, int]],
    n_replicates: int = 1000,
    ci: float = 0.95,
    seed: int = 42,
) -> tuple[float, float]:
    """Bootstrap confidence interval for mean Brier score.

    Returns (lower, upper) bounds of the CI.
    """
    import random

    rng = random.Random(seed)
    n = len(pairs)
    if n == 0:
        return (0.0, 0.0)
    means = []
    for _ in range(n_replicates):
        sample = rng.choices(pairs, k=n)
        mean_bs = sum((f - o) ** 2 for f, o in sample) / n
        means.append(mean_bs)
    means.sort()
    alpha = (1 - ci) / 2
    lo = means[int(alpha * n_replicates)]
    hi = means[int((1 - alpha) * n_replicates)]
    return (lo, hi)


def murphy_decomposition(
    pairs: list[tuple[float, int]],
    n_bins: int = 10,
) -> dict[str, float]:
    """Decompose the Brier score into reliability, resolution, and uncertainty.

    Uses Murphy's additive partition: Brier = REL - RES + UNC.

    Reference:
        Murphy, A. H. (1973). 'A New Vector Partition of the Probability
        Score.' Journal of Applied Meteorology, 12(4), 595-600.

    Args:
        pairs: List of (forecast_probability, binary_outcome) tuples.
        n_bins: Number of equally-spaced bins in [0, 1]. Default 10.

    Returns:
        Dict with keys: reliability, resolution, uncertainty, brier_check.
    """
    if not pairs:
        raise ValueError("Cannot decompose empty list of pairs")

    for f, o in pairs:
        _validate_forecast(f)
        _validate_outcome(o)

    n = len(pairs)
    base_rate = sum(o for _, o in pairs) / n

    bin_width = 1.0 / n_bins
    bins: list[tuple[float, float, list[tuple[float, int]]]] = []
    for i in range(n_bins):
        low = i * bin_width
        high = (i + 1) * bin_width
        in_bin = [
            (f, o) for f, o in pairs
            if low <= f < high or (i == n_bins - 1 and f == high)
        ]
        if in_bin:
            bins.append((low, high, in_bin))

    reliability = 0.0
    resolution = 0.0
    for _, _, bin_pairs in bins:
        n_k = len(bin_pairs)
        f_k = sum(f for f, _ in bin_pairs) / n_k
        o_k = sum(o for _, o in bin_pairs) / n_k
        reliability += n_k * (f_k - o_k) ** 2
        resolution += n_k * (o_k - base_rate) ** 2

    reliability /= n
    resolution /= n
    uncertainty = base_rate * (1.0 - base_rate)
    brier_check = reliability - resolution + uncertainty

    logger.info(
        "murphy_decomposition",
        reliability=round(reliability, 6),
        resolution=round(resolution, 6),
        uncertainty=round(uncertainty, 6),
        brier_check=round(brier_check, 6),
        n_bins_used=len(bins),
    )

    return {
        "reliability": reliability,
        "resolution": resolution,
        "uncertainty": uncertainty,
        "brier_check": brier_check,
    }


def _is_market_question(q: ResolvedQuestion) -> bool:
    source_lower = q.source.lower()
    return any(s in source_lower for s in MARKET_SOURCES)


def _estimate_difficulty_effects_ols(
    all_forecasts: dict[str, dict[str, float]],
    outcomes: dict[str, int],
    question_ids: list[str],
) -> dict[str, float]:
    """Estimate question-difficulty fixed effects via OLS on Brier scores.

    Uses the two-way fixed-effects model: b_{i,j} = α_i + γ_j + ε_{i,j}
    The OLS estimate of γ_j equals the mean Brier score on question j
    across all forecasters who answered it, minus the grand mean.
    With demeaned data (forecaster effects removed), γ̂_j = mean_j(b_{i,j}) - grand_mean.
    """
    if not question_ids or not all_forecasts:
        return {}

    question_scores: dict[str, list[float]] = {qid: [] for qid in question_ids}
    for forecaster_id, fcast_map in all_forecasts.items():
        for qid in question_ids:
            if qid in fcast_map and qid in outcomes:
                bs = (fcast_map[qid] - outcomes[qid]) ** 2
                question_scores[qid].append(bs)

    question_means: dict[str, float] = {}
    for qid, scores in question_scores.items():
        if scores:
            question_means[qid] = sum(scores) / len(scores)

    if not question_means:
        return {}

    all_scores = [s for scores in question_scores.values() for s in scores]
    grand_mean = sum(all_scores) / len(all_scores) if all_scores else 0.0
    return {qid: mean - grand_mean for qid, mean in question_means.items()}


def _build_market_effects(
    all_forecasts: dict[str, dict[str, float]],
    outcomes: dict[str, int],
    market_qids: list[str],
    market_weight: float,
    market_forecasts: dict[str, float] | None,
) -> dict[str, float]:
    """Compute difficulty effects for market questions using weighted estimator."""
    if not market_qids:
        return {}
    ols_market = _estimate_difficulty_effects_ols(all_forecasts, outcomes, market_qids)
    if market_weight == 0.0:
        return ols_market
    if market_weight == 1.0 and market_forecasts:
        raw: dict[str, float] = {}
        for qid in market_qids:
            if qid in outcomes and qid in market_forecasts:
                raw[qid] = (market_forecasts[qid] - outcomes[qid]) ** 2
        if not raw:
            return {}
        raw_mean = sum(raw.values()) / len(raw)
        return {qid: bs - raw_mean for qid, bs in raw.items()}
    if market_forecasts:
        raw_mkt: dict[str, float] = {}
        effects: dict[str, float] = {}
        for qid in market_qids:
            ols_val = ols_market.get(qid, 0.0)
            if qid in outcomes and qid in market_forecasts:
                mkt_bs = (market_forecasts[qid] - outcomes[qid]) ** 2
                raw_mkt[qid] = mkt_bs
            else:
                effects[qid] = ols_val
        if raw_mkt:
            mkt_mean = sum(raw_mkt.values()) / len(raw_mkt)
            for qid, mkt_bs in raw_mkt.items():
                ols_val = ols_market.get(qid, 0.0)
                centered_mkt = mkt_bs - mkt_mean
                effects[qid] = market_weight * centered_mkt + (1.0 - market_weight) * ols_val
        return effects
    return ols_market


def adjust_for_difficulty(
    all_forecasts: dict[str, dict[str, float]],
    resolved: list[ResolvedQuestion],
    market_weight: float = 1.0,
    market_forecasts: dict[str, float] | None = None,
) -> AdjustmentResult:
    """Apply difficulty-adjusted Brier scoring per ForecastBench methodology.

    Args:
        all_forecasts: {forecaster_id: {question_id: forecast}} for all forecasters.
        resolved: Resolved questions with outcomes.
        market_weight: Weight on market-based difficulty for market questions (w_mkt).
            ForecastBench uses 1.0 in practice.
        market_forecasts: {question_id: market_probability} for market questions.
            Required when market_weight > 0 and market questions exist.

    Returns:
        AdjustmentResult with adjusted_scores and question_effects.
    """
    if not all_forecasts or not resolved:
        return AdjustmentResult(adjusted_scores={}, question_effects={})

    logger.info(
        "difficulty_adjustment_start",
        n_forecasters=len(all_forecasts),
        n_questions=len(resolved),
        market_weight=market_weight,
    )

    outcomes = {q.id: q.outcome for q in resolved}
    dataset_qids = [q.id for q in resolved if not _is_market_question(q)]
    market_qids = [q.id for q in resolved if _is_market_question(q)]

    dataset_effects = _estimate_difficulty_effects_ols(
        all_forecasts, outcomes, dataset_qids,
    )
    market_effects = _build_market_effects(
        all_forecasts, outcomes, market_qids, market_weight, market_forecasts,
    )

    all_effects = {**dataset_effects, **market_effects}

    raw_brier: dict[str, dict[str, float]] = {}
    for fid, fcast_map in all_forecasts.items():
        raw_brier[fid] = {}
        for qid in fcast_map:
            if qid in outcomes:
                raw_brier[fid][qid] = (fcast_map[qid] - outcomes[qid]) ** 2

    unscaled: dict[str, dict[str, float]] = {}
    for fid, scores in raw_brier.items():
        unscaled[fid] = {}
        for qid, bs in scores.items():
            effect = all_effects.get(qid, 0.0)
            unscaled[fid][qid] = bs - effect

    constant_half_scores: list[float] = []
    for q in resolved:
        bs_half = (0.5 - q.outcome) ** 2
        effect = all_effects.get(q.id, 0.0)
        constant_half_scores.append(bs_half - effect)

    if constant_half_scores:
        mean_half_unscaled = sum(constant_half_scores) / len(constant_half_scores)
    else:
        mean_half_unscaled = 0.0

    shift = 0.25 - mean_half_unscaled

    adjusted: dict[str, dict[str, float]] = {}
    for fid, scores in unscaled.items():
        adjusted[fid] = {}
        for qid, val in scores.items():
            adjusted[fid][qid] = max(0.0, min(1.0, val + shift))

    return AdjustmentResult(adjusted_scores=adjusted, question_effects=all_effects)


def score_forecasts(
    forecasts: dict[str, float],
    resolved: list[ResolvedQuestion],
    *,
    difficulty_adjusted: bool = True,
    all_forecasts: dict[str, dict[str, float]] | None = None,
    market_weight: float = 1.0,
    market_forecasts: dict[str, float] | None = None,
) -> ScoringResult:
    """Score forecasts against resolved questions.

    Missing forecasts default to 0.5 per ForecastBench rules.
    When difficulty_adjusted=True and all_forecasts is provided,
    applies the ForecastBench two-way fixed-effects adjustment.
    """
    logger.info("scoring_start", n_questions=len(resolved), difficulty_adjusted=difficulty_adjusted)

    if not resolved:
        raise ValueError("No resolved questions to score")

    n_missing = 0
    complete_forecasts: dict[str, float] = {}
    for q in resolved:
        if q.id in forecasts:
            complete_forecasts[q.id] = forecasts[q.id]
        else:
            complete_forecasts[q.id] = 0.5
            n_missing += 1

    for f in complete_forecasts.values():
        _validate_forecast(f)
    for q in resolved:
        _validate_outcome(q.outcome)

    question_effects: dict[str, float] = {}

    if difficulty_adjusted and all_forecasts and len(all_forecasts) > 1:
        forecaster_id = f"_target_{uuid4().hex}"
        pool = dict(all_forecasts)
        pool[forecaster_id] = complete_forecasts
        adj_result = adjust_for_difficulty(
            pool, resolved,
            market_weight=market_weight,
            market_forecasts=market_forecasts,
        )
        target_adjusted = adj_result.adjusted_scores.get(forecaster_id, {})
        question_effects = adj_result.question_effects

        dataset_scores: list[float] = []
        market_scores: list[float] = []
        for q in resolved:
            adj = target_adjusted.get(q.id)
            if adj is None:
                adj = brier_score(complete_forecasts[q.id], q.outcome)
            if _is_market_question(q):
                market_scores.append(adj)
            else:
                dataset_scores.append(adj)

        ds_brier = (sum(dataset_scores) / len(dataset_scores)) if dataset_scores else 0.0
        mk_brier = (sum(market_scores) / len(market_scores)) if market_scores else 0.0
    else:
        dataset_pairs: list[tuple[float, int]] = []
        market_pairs: list[tuple[float, int]] = []

        for q in resolved:
            f = complete_forecasts[q.id]
            if _is_market_question(q):
                market_pairs.append((f, q.outcome))
            else:
                dataset_pairs.append((f, q.outcome))

        ds_brier = mean_brier_score(dataset_pairs) if dataset_pairs else 0.0
        mk_brier = mean_brier_score(market_pairs) if market_pairs else 0.0

    n_dataset = len([q for q in resolved if not _is_market_question(q)])
    n_market = len([q for q in resolved if _is_market_question(q)])

    ds_index = brier_index(ds_brier) if n_dataset > 0 else 0.0
    mk_index = brier_index(mk_brier) if n_market > 0 else 0.0

    total_questions = n_dataset + n_market
    if total_questions > 0:
        overall_bs = (ds_brier * n_dataset + mk_brier * n_market) / total_questions
    else:
        overall_bs = 0.0

    result = ScoringResult(
        dataset_brier=ds_brier,
        dataset_index=ds_index,
        market_brier=mk_brier,
        market_index=mk_index,
        overall_brier=overall_bs,
        overall_index=brier_index(overall_bs),
        n_dataset=n_dataset,
        n_market=n_market,
        n_missing=n_missing,
        difficulty_adjusted=difficulty_adjusted and all_forecasts is not None and len(all_forecasts or {}) > 1,
        question_effects=question_effects,
    )

    logger.info(
        "scoring_complete",
        question_count=total_questions,
        overall_brier=round(result.overall_brier, 6),
        overall_index=round(result.overall_index, 2),
        dataset_brier=round(result.dataset_brier, 6),
        dataset_index=round(result.dataset_index, 2),
        market_brier=round(result.market_brier, 6),
        market_index=round(result.market_index, 2),
        n_dataset=n_dataset,
        n_market=n_market,
        n_missing=n_missing,
        difficulty_adjusted=result.difficulty_adjusted,
    )

    return result
