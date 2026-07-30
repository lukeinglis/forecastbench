"""Statistical baseline forecasts for timeseries questions (fred, dbnomics, yfinance)."""

from __future__ import annotations

import math
import re
from datetime import date

import numpy as np
from scipy.stats import norm

from fetch_data import Question

TIMESERIES_SOURCES = frozenset({"fred", "dbnomics", "yfinance"})

MIN_DATAPOINTS_FOR_RANDOM_WALK = 5

_THRESHOLD_PATTERNS = [
    re.compile(r"(?:exceed|surpass|rise above|go above|be above|above|greater than|more than)\s+([\d,.]+)\s*%?", re.I),
    re.compile(r"(?:fall below|drop below|go below|be below|below|less than|under)\s+([\d,.]+)\s*%?", re.I),
    re.compile(r"(?:reach|hit|cross)\s+([\d,.]+)\s*%?", re.I),
]


def extract_threshold(question_text: str, resolution_criteria: str) -> tuple[float, str] | None:
    """Parse question text and resolution criteria for numeric threshold and direction."""
    combined = question_text + " " + resolution_criteria

    for i, pattern in enumerate(_THRESHOLD_PATTERNS):
        m = pattern.search(combined)
        if m:
            try:
                threshold = float(m.group(1).replace(",", ""))
            except ValueError:
                continue
            direction = "below" if i == 1 else "above"
            return (threshold, direction)

    return None


def compute_naive_forecast(freeze_value: float, threshold: float, direction: str) -> float:
    """Naive forecast based on current distance from threshold (fallback)."""
    if direction == "above":
        if freeze_value >= threshold:
            ratio = min((freeze_value - threshold) / max(abs(threshold), 1e-6), 1.0)
            return 0.5 + 0.3 * ratio
        else:
            ratio = min((threshold - freeze_value) / max(abs(threshold), 1e-6), 1.0)
            return 0.5 - 0.3 * ratio
    else:
        if freeze_value <= threshold:
            ratio = min((threshold - freeze_value) / max(abs(threshold), 1e-6), 1.0)
            return 0.5 + 0.3 * ratio
        else:
            ratio = min((freeze_value - threshold) / max(abs(threshold), 1e-6), 1.0)
            return 0.5 - 0.3 * ratio


def compute_random_walk_forecast(
    historical_data: dict[str, float],
    freeze_value: float,
    threshold: float,
    direction: str,
    horizon_days: int,
) -> float | None:
    """Compute threshold-crossing probability via random walk + log-normal CDF.

    Uses historical log returns to estimate drift (mu) and volatility (sigma),
    then projects forward assuming geometric Brownian motion:
      log(future) ~ Normal(log(current) + mu*days, sigma*sqrt(days))

    Returns None if insufficient data (<5 points) to estimate parameters.
    """
    sorted_dates = sorted(historical_data.keys())
    values = [historical_data[d] for d in sorted_dates]

    if len(values) < MIN_DATAPOINTS_FOR_RANDOM_WALK:
        return None

    positive_values = [v for v in values if v > 0]
    if len(positive_values) < MIN_DATAPOINTS_FOR_RANDOM_WALK:
        return None

    log_returns: list[float] = []
    for i in range(1, len(values)):
        if values[i] > 0 and values[i - 1] > 0:
            log_returns.append(math.log(values[i] / values[i - 1]))

    if len(log_returns) < 2:
        return None

    mu = float(np.mean(log_returns))
    sigma = float(np.std(log_returns, ddof=1))

    if sigma < 1e-12:
        projected = freeze_value * math.exp(mu * horizon_days)
        if direction == "above":
            return 0.98 if projected > threshold else 0.02
        else:
            return 0.98 if projected < threshold else 0.02

    if freeze_value <= 0 or threshold <= 0:
        return None

    log_current = math.log(freeze_value)
    projected_log_mean = log_current + mu * horizon_days
    projected_log_std = sigma * math.sqrt(horizon_days)

    log_threshold = math.log(threshold)

    if direction == "above":
        prob = 1.0 - float(norm.cdf(log_threshold, loc=projected_log_mean, scale=projected_log_std))
    else:
        prob = float(norm.cdf(log_threshold, loc=projected_log_mean, scale=projected_log_std))

    return float(np.clip(prob, 0.02, 0.98))


def _compute_horizon_days(freeze_datetime: str, resolution_date: str) -> int | None:
    """Compute days between freeze date and resolution date."""
    try:
        freeze = date.fromisoformat(str(freeze_datetime)[:10])
        resolve = date.fromisoformat(str(resolution_date)[:10])
        delta = (resolve - freeze).days
        return max(delta, 1)
    except (ValueError, TypeError):
        return None


def get_statistical_forecast(
    question: Question,
    historical_data: dict[str, float] | None,
    horizon_days: int,
) -> float | None:
    """Compute a statistical forecast for a timeseries question.

    Tries random walk + CDF first; falls back to naive heuristic if insufficient data.
    Returns None if the question isn't a parseable timeseries threshold question.
    """
    if question.source.lower() not in TIMESERIES_SOURCES:
        return None
    if question.freeze_datetime_value is None:
        return None

    try:
        freeze_value = float(question.freeze_datetime_value)
    except (TypeError, ValueError):
        return None

    result = extract_threshold(question.question, question.resolution_criteria or "")
    if result is None:
        return None
    threshold, direction = result

    if historical_data and len(historical_data) >= MIN_DATAPOINTS_FOR_RANDOM_WALK:
        rw_forecast = compute_random_walk_forecast(
            historical_data, freeze_value, threshold, direction, horizon_days
        )
        if rw_forecast is not None:
            return rw_forecast

    return compute_naive_forecast(freeze_value, threshold, direction)


def format_statistical_context(forecast: float, method: str = "naive") -> str:
    """Format statistical baseline as a prompt snippet."""
    return (
        f"Statistical baseline ({method} method): {forecast:.0%} probability\n"
        f"Note: This is a simple statistical estimate. Use your judgment to adjust."
    )


def get_statistical_context(question: Question) -> str | None:
    """Compute statistical context for a timeseries question.

    Only applies to fred, dbnomics, yfinance sources.
    Uses naive method (no historical data). For random walk, use get_statistical_forecast().
    """
    if question.source.lower() not in TIMESERIES_SOURCES:
        return None
    if question.freeze_datetime_value is None:
        return None
    try:
        freeze_value = float(question.freeze_datetime_value)
    except (TypeError, ValueError):
        return None

    result = extract_threshold(question.question, question.resolution_criteria or "")
    if result is None:
        return None
    threshold, direction = result
    forecast = compute_naive_forecast(freeze_value, threshold, direction)
    return format_statistical_context(forecast)
