"""Tests for statistical_baseline module."""

from __future__ import annotations

import pytest

from fetch_data import Question
from statistical_baseline import (
    compute_naive_forecast,
    extract_threshold,
    format_statistical_context,
    get_statistical_context,
)


def _make_question(source: str = "fred", **overrides: object) -> Question:
    defaults: dict[str, object] = dict(
        id="test-q",
        source=source,
        question="Will X exceed 100 by the resolution date?",
        resolution_criteria="Resolves YES if value exceeds 100.",
        freeze_datetime="2025-03-01",
        freeze_datetime_value=100.0,
        resolution_dates=["2025-04-01"],
    )
    defaults.update(overrides)
    return Question(**defaults)  # type: ignore[arg-type]


class TestExtractThreshold:
    @pytest.mark.parametrize(
        "question,criteria,expected_value,expected_dir",
        [
            ("Will X exceed 5.0 by date?", "", 5.0, "above"),
            ("Will X fall below 100?", "", 100.0, "below"),
            ("Will X be above 3.5%?", "", 3.5, "above"),
            ("Will X drop below 2,500 by next month?", "", 2500.0, "below"),
            ("Will X surpass 42.7?", "", 42.7, "above"),
            ("Will it be greater than 10?", "", 10.0, "above"),
            ("", "Value must be less than 50", 50.0, "below"),
        ],
    )
    def test_parses_common_patterns(
        self,
        question: str,
        criteria: str,
        expected_value: float,
        expected_dir: str,
    ) -> None:
        result = extract_threshold(question, criteria)
        assert result is not None
        value, direction = result
        assert value == pytest.approx(expected_value)
        assert direction == expected_dir

    def test_returns_none_for_unparseable(self) -> None:
        result = extract_threshold(
            "What will the weather be like tomorrow?",
            "Based on observations.",
        )
        assert result is None


class TestComputeNaiveForecast:
    def test_value_above_threshold_returns_above_half(self) -> None:
        prob = compute_naive_forecast(freeze_value=110.0, threshold=100.0, direction="above")
        assert prob > 0.5

    def test_value_below_threshold_returns_below_half(self) -> None:
        prob = compute_naive_forecast(freeze_value=90.0, threshold=100.0, direction="above")
        assert prob < 0.5

    def test_value_at_threshold_returns_half(self) -> None:
        prob = compute_naive_forecast(freeze_value=100.0, threshold=100.0, direction="above")
        assert prob == pytest.approx(0.5)

    def test_direction_below_inverts(self) -> None:
        prob_above = compute_naive_forecast(freeze_value=110.0, threshold=100.0, direction="above")
        prob_below = compute_naive_forecast(freeze_value=110.0, threshold=100.0, direction="below")
        assert prob_above > 0.5
        assert prob_below < 0.5


class TestGetStatisticalContext:
    def test_returns_none_for_market_source(self) -> None:
        q = _make_question(source="metaculus")
        assert get_statistical_context(q) is None

    def test_returns_none_without_freeze_value(self) -> None:
        q = _make_question(source="fred", freeze_datetime_value=None)
        assert get_statistical_context(q) is None

    def test_returns_string_for_fred_source(self) -> None:
        q = _make_question(
            source="fred",
            question="Will the rate exceed 5.0?",
            freeze_datetime_value=4.5,
        )
        result = get_statistical_context(q)
        assert result is not None
        assert "Statistical baseline" in result

    def test_returns_string_for_dbnomics(self) -> None:
        q = _make_question(
            source="dbnomics",
            question="Will GDP exceed 1000?",
            freeze_datetime_value=950.0,
        )
        result = get_statistical_context(q)
        assert result is not None

    def test_returns_string_for_yfinance(self) -> None:
        q = _make_question(
            source="yfinance",
            question="Will price exceed 200?",
            freeze_datetime_value=190.0,
        )
        result = get_statistical_context(q)
        assert result is not None

    def test_returns_none_for_no_threshold(self) -> None:
        q = _make_question(
            source="fred",
            question="What will the weather be like?",
            resolution_criteria="Based on observations.",
            freeze_datetime_value=42.0,
        )
        assert get_statistical_context(q) is None


class TestFormatStatisticalContext:
    def test_produces_expected_format(self) -> None:
        result = format_statistical_context(0.65, "naive")
        assert "Statistical baseline (naive method): 65% probability" in result
        assert "simple statistical estimate" in result
