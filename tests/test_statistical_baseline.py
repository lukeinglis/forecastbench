"""Tests for statistical_baseline module."""

from __future__ import annotations

import math

import numpy as np
import pytest

from fetch_data import Question
from statistical_baseline import (
    MIN_DATAPOINTS_FOR_RANDOM_WALK,
    compute_naive_forecast,
    compute_random_walk_forecast,
    extract_threshold,
    format_statistical_context,
    get_statistical_context,
    get_statistical_forecast,
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


def _make_historical_data(
    n: int = 60,
    start: float = 100.0,
    daily_return: float = 0.001,
    volatility: float = 0.01,
    seed: int = 42,
) -> dict[str, float]:
    """Generate synthetic daily price data with known drift and volatility."""
    rng = np.random.default_rng(seed)
    prices = [start]
    for _ in range(n - 1):
        log_ret = daily_return + volatility * rng.standard_normal()
        prices.append(prices[-1] * math.exp(log_ret))

    from datetime import date, timedelta

    base = date(2025, 1, 1)
    return {str(base + timedelta(days=i)): float(p) for i, p in enumerate(prices)}


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
            ("Will the index reach 5000?", "", 5000.0, "above"),
            ("Will GDP rise above 20?", "", 20.0, "above"),
            ("Will price go below 150?", "", 150.0, "below"),
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

    def test_threshold_in_resolution_criteria_only(self) -> None:
        result = extract_threshold("What happens next?", "Resolves YES if value is above 42 units.")
        assert result is not None
        assert result[0] == pytest.approx(42.0)
        assert result[1] == "above"


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


class TestComputeRandomWalkForecast:
    def test_returns_none_for_insufficient_data(self) -> None:
        data = {"2025-01-01": 100.0, "2025-01-02": 101.0}
        result = compute_random_walk_forecast(data, 101.0, 110.0, "above", 30)
        assert result is None

    def test_returns_none_for_negative_values(self) -> None:
        data = _make_historical_data(n=10, start=-5.0)
        for k in data:
            data[k] = -abs(data[k])
        result = compute_random_walk_forecast(data, -5.0, 10.0, "above", 30)
        assert result is None

    def test_positive_drift_above_threshold_high_prob(self) -> None:
        data = _make_historical_data(n=60, start=100.0, daily_return=0.005, volatility=0.005)
        last_value = list(data.values())[-1]
        threshold = last_value * 1.05
        result = compute_random_walk_forecast(data, last_value, threshold, "above", 90)
        assert result is not None
        assert result > 0.3

    def test_negative_drift_below_threshold_high_prob(self) -> None:
        data = _make_historical_data(n=60, start=100.0, daily_return=-0.005, volatility=0.005)
        last_value = list(data.values())[-1]
        threshold = last_value * 0.95
        result = compute_random_walk_forecast(data, last_value, threshold, "below", 90)
        assert result is not None
        assert result > 0.3

    def test_threshold_already_crossed_above(self) -> None:
        data = _make_historical_data(n=60, start=100.0, daily_return=0.001, volatility=0.01)
        last_value = list(data.values())[-1]
        threshold = last_value * 0.5
        result = compute_random_walk_forecast(data, last_value, threshold, "above", 30)
        assert result is not None
        assert result > 0.9

    def test_threshold_already_crossed_below(self) -> None:
        data = _make_historical_data(n=60, start=100.0, daily_return=0.001, volatility=0.01)
        last_value = list(data.values())[-1]
        threshold = last_value * 2.0
        result = compute_random_walk_forecast(data, last_value, threshold, "below", 30)
        assert result is not None
        assert result > 0.9

    def test_result_clipped_to_bounds(self) -> None:
        data = _make_historical_data(n=60, start=100.0, daily_return=0.0, volatility=0.001)
        last_value = list(data.values())[-1]
        result = compute_random_walk_forecast(data, last_value, last_value * 0.01, "above", 30)
        assert result is not None
        assert result <= 0.98
        assert result >= 0.02

    def test_zero_volatility_above(self) -> None:
        dates = [f"2025-01-{d:02d}" for d in range(1, 11)]
        data = {d: 100.0 for d in dates}
        result = compute_random_walk_forecast(data, 100.0, 110.0, "above", 30)
        assert result is not None
        assert result == pytest.approx(0.02)

    def test_zero_volatility_below(self) -> None:
        dates = [f"2025-01-{d:02d}" for d in range(1, 11)]
        data = {d: 100.0 for d in dates}
        result = compute_random_walk_forecast(data, 100.0, 110.0, "below", 30)
        assert result is not None
        assert result == pytest.approx(0.98)

    def test_multi_horizon_monotonicity_above(self) -> None:
        data = _make_historical_data(n=60, start=100.0, daily_return=0.001, volatility=0.02)
        last_value = list(data.values())[-1]
        threshold = last_value * 1.1

        horizons = [7, 30, 60, 90, 180]
        probs = []
        for h in horizons:
            p = compute_random_walk_forecast(data, last_value, threshold, "above", h)
            assert p is not None
            probs.append(p)

        for i in range(len(probs) - 1):
            assert probs[i] <= probs[i + 1] + 1e-6, (
                f"P(cross by {horizons[i]}d)={probs[i]:.4f} > "
                f"P(cross by {horizons[i+1]}d)={probs[i+1]:.4f}"
            )

    def test_multi_horizon_monotonicity_below(self) -> None:
        data = _make_historical_data(n=60, start=100.0, daily_return=-0.001, volatility=0.02)
        last_value = list(data.values())[-1]
        threshold = last_value * 0.9

        horizons = [7, 30, 60, 90, 180]
        probs = []
        for h in horizons:
            p = compute_random_walk_forecast(data, last_value, threshold, "below", h)
            assert p is not None
            probs.append(p)

        for i in range(len(probs) - 1):
            assert probs[i] <= probs[i + 1] + 1e-6, (
                f"P(cross by {horizons[i]}d)={probs[i]:.4f} > "
                f"P(cross by {horizons[i+1]}d)={probs[i+1]:.4f}"
            )

    def test_exactly_min_datapoints(self) -> None:
        dates = [f"2025-01-{d:02d}" for d in range(1, MIN_DATAPOINTS_FOR_RANDOM_WALK + 1)]
        data = {d: 100.0 + i * 0.5 for i, d in enumerate(dates)}
        result = compute_random_walk_forecast(data, 102.0, 110.0, "above", 30)
        assert result is not None

    def test_one_below_min_datapoints_returns_none(self) -> None:
        dates = [f"2025-01-{d:02d}" for d in range(1, MIN_DATAPOINTS_FOR_RANDOM_WALK)]
        data = {d: 100.0 + i * 0.5 for i, d in enumerate(dates)}
        result = compute_random_walk_forecast(data, 102.0, 110.0, "above", 30)
        assert result is None


class TestGetStatisticalForecast:
    def test_returns_none_for_market_source(self) -> None:
        q = _make_question(source="metaculus")
        assert get_statistical_forecast(q, None, 30) is None

    def test_returns_none_without_freeze_value(self) -> None:
        q = _make_question(source="fred", freeze_datetime_value=None)
        assert get_statistical_forecast(q, None, 30) is None

    def test_returns_none_for_unparseable_threshold(self) -> None:
        q = _make_question(
            source="fred",
            question="What will the weather be?",
            resolution_criteria="Based on observations.",
            freeze_datetime_value=42.0,
        )
        assert get_statistical_forecast(q, None, 30) is None

    def test_uses_random_walk_when_data_available(self) -> None:
        data = _make_historical_data(n=60, start=100.0, daily_return=0.001, volatility=0.01)
        q = _make_question(
            source="fred",
            question="Will the rate exceed 150?",
            freeze_datetime_value=100.0,
        )
        result = get_statistical_forecast(q, data, 30)
        assert result is not None
        naive = compute_naive_forecast(100.0, 150.0, "above")
        assert result != pytest.approx(naive, abs=0.01)

    def test_falls_back_to_naive_without_data(self) -> None:
        q = _make_question(
            source="fred",
            question="Will the rate exceed 150?",
            freeze_datetime_value=100.0,
        )
        result = get_statistical_forecast(q, None, 30)
        assert result is not None
        naive = compute_naive_forecast(100.0, 150.0, "above")
        assert result == pytest.approx(naive)

    def test_falls_back_to_naive_with_short_data(self) -> None:
        data = {"2025-01-01": 100.0, "2025-01-02": 101.0}
        q = _make_question(
            source="fred",
            question="Will the rate exceed 150?",
            freeze_datetime_value=100.0,
        )
        result = get_statistical_forecast(q, data, 30)
        assert result is not None
        naive = compute_naive_forecast(100.0, 150.0, "above")
        assert result == pytest.approx(naive)

    def test_works_for_all_timeseries_sources(self) -> None:
        data = _make_historical_data(n=20, start=50.0, daily_return=0.002, volatility=0.01)
        for source in ["fred", "dbnomics", "yfinance"]:
            q = _make_question(
                source=source,
                question="Will value exceed 60?",
                freeze_datetime_value=50.0,
            )
            result = get_statistical_forecast(q, data, 30)
            assert result is not None
            assert 0.02 <= result <= 0.98


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
