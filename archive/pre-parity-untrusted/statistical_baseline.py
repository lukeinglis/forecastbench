"""Statistical baseline forecasts for timeseries questions (fred, dbnomics, yfinance)."""

from __future__ import annotations

import re

from fetch_data import Question

TIMESERIES_SOURCES = frozenset({"fred", "dbnomics", "yfinance"})

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
    """Naive forecast based on current distance from threshold."""
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


def format_statistical_context(forecast: float, method: str = "naive") -> str:
    """Format statistical baseline as a prompt snippet."""
    return (
        f"Statistical baseline ({method} method): {forecast:.0%} probability\n"
        f"Note: This is a simple statistical estimate. Use your judgment to adjust."
    )


def get_statistical_context(question: Question) -> str | None:
    """Compute statistical context for a timeseries question.

    Only applies to fred, dbnomics, yfinance sources.
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
