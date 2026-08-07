"""Tests for Murphy decomposition and calibration metrics."""

from __future__ import annotations

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from analyze import calibration_metrics, analyze_decomposition
from fetch_data import ResolvedQuestion
from score import murphy_decomposition, mean_brier_score


def _rq(qid: str, outcome: int) -> ResolvedQuestion:
    return ResolvedQuestion(
        id=qid, source="acled", question=f"Q {qid}",
        outcome=outcome, forecast_due_date="2024-01-01",
    )


class TestMurphyDecomposition:
    def test_identity_rel_minus_res_plus_unc_equals_brier(self) -> None:
        pairs = [(0.9, 1), (0.1, 0), (0.7, 1), (0.3, 0), (0.5, 1)]
        result = murphy_decomposition(pairs)
        brier = mean_brier_score(pairs)
        reconstructed = result["reliability"] - result["resolution"] + result["uncertainty"]
        assert abs(reconstructed - brier) < 1e-10
        assert abs(result["brier_check"] - brier) < 1e-10

    def test_perfect_calibration_reliability_zero(self) -> None:
        """When forecasts exactly match observed frequencies in each bin, reliability = 0."""
        pairs = [(0.0, 0)] * 10 + [(1.0, 1)] * 10
        result = murphy_decomposition(pairs)
        assert abs(result["reliability"]) < 1e-10

    def test_constant_forecaster_resolution_zero(self) -> None:
        """A forecaster predicting 0.5 for everything has resolution = 0."""
        pairs = [(0.5, 1)] * 5 + [(0.5, 0)] * 5
        result = murphy_decomposition(pairs)
        assert abs(result["resolution"]) < 1e-10

    def test_uncertainty_is_base_rate_entropy(self) -> None:
        pairs = [(0.5, 1)] * 3 + [(0.5, 0)] * 7
        result = murphy_decomposition(pairs)
        base_rate = 3 / 10
        expected_unc = base_rate * (1.0 - base_rate)
        assert abs(result["uncertainty"] - expected_unc) < 1e-10

    def test_all_outcomes_one(self) -> None:
        pairs = [(0.8, 1), (0.9, 1), (0.7, 1)]
        result = murphy_decomposition(pairs)
        assert abs(result["uncertainty"]) < 1e-10
        reconstructed = result["reliability"] - result["resolution"] + result["uncertainty"]
        brier = mean_brier_score(pairs)
        assert abs(reconstructed - brier) < 0.01

    def test_all_outcomes_zero(self) -> None:
        pairs = [(0.2, 0), (0.1, 0), (0.3, 0)]
        result = murphy_decomposition(pairs)
        assert abs(result["uncertainty"]) < 1e-10
        reconstructed = result["reliability"] - result["resolution"] + result["uncertainty"]
        brier = mean_brier_score(pairs)
        # Binning approximation: identity holds within bin-mean tolerance
        assert abs(reconstructed - brier) < 0.01

    def test_single_pair(self) -> None:
        pairs = [(0.7, 1)]
        result = murphy_decomposition(pairs)
        brier = mean_brier_score(pairs)
        reconstructed = result["reliability"] - result["resolution"] + result["uncertainty"]
        assert abs(reconstructed - brier) < 1e-10  # single pair, one bin, exact

    def test_empty_raises(self) -> None:
        with pytest.raises(ValueError, match="empty"):
            murphy_decomposition([])

    def test_all_same_forecast(self) -> None:
        pairs = [(0.6, 1), (0.6, 0), (0.6, 1), (0.6, 0)]
        result = murphy_decomposition(pairs)
        assert abs(result["resolution"]) < 1e-10
        reconstructed = result["reliability"] - result["resolution"] + result["uncertainty"]
        brier = mean_brier_score(pairs)
        assert abs(reconstructed - brier) < 1e-10  # all in one bin, exact

    def test_reliability_nonnegative(self) -> None:
        pairs = [(0.3, 1), (0.7, 0), (0.5, 1), (0.1, 0)]
        result = murphy_decomposition(pairs)
        assert result["reliability"] >= -1e-10

    def test_resolution_nonnegative(self) -> None:
        pairs = [(0.9, 1), (0.1, 0), (0.5, 1), (0.5, 0)]
        result = murphy_decomposition(pairs)
        assert result["resolution"] >= -1e-10

    def test_known_values(self) -> None:
        """Hand-computed example: 4 pairs in 2 bins.

        Bin [0, 0.1): pairs (0.1,0),(0.1,0) -> f_k=0.1, o_k=0.0
        Bin [0.9, 1.0]: pairs (0.9,1),(0.9,1) -> f_k=0.9, o_k=1.0
        base_rate = 0.5
        REL = (2*(0.1-0)^2 + 2*(0.9-1)^2)/4 = 0.01
        RES = (2*(0-0.5)^2 + 2*(1-0.5)^2)/4 = 0.25
        UNC = 0.5*0.5 = 0.25
        """
        pairs = [(0.1, 0), (0.1, 0), (0.9, 1), (0.9, 1)]
        result = murphy_decomposition(pairs)
        assert abs(result["reliability"] - 0.01) < 1e-10
        assert abs(result["uncertainty"] - 0.25) < 1e-10
        assert abs(result["resolution"] - 0.25) < 1e-10
        assert abs(result["brier_check"] - 0.01) < 1e-10

    @given(
        pairs=st.lists(
            st.tuples(
                st.floats(min_value=0.0, max_value=1.0),
                st.sampled_from([0, 1]),
            ),
            min_size=10,
            max_size=50,
        ),
    )
    @settings(max_examples=50)
    def test_identity_holds_property(self, pairs: list[tuple[float, int]]) -> None:
        result = murphy_decomposition(pairs)
        brier = mean_brier_score(pairs)
        reconstructed = result["reliability"] - result["resolution"] + result["uncertainty"]
        # Binning approximation: within-bin variance causes deviations with small samples
        assert abs(reconstructed - brier) < 0.05


class TestCalibrationMetrics:
    def test_perfect_calibration_ece_zero(self) -> None:
        pairs = [(0.0, 0)] * 10 + [(1.0, 1)] * 10
        result = calibration_metrics(pairs)
        assert abs(result["ece"]) < 1e-10

    def test_perfect_calibration_mce_zero(self) -> None:
        pairs = [(0.0, 0)] * 10 + [(1.0, 1)] * 10
        result = calibration_metrics(pairs)
        assert abs(result["mce"]) < 1e-10

    def test_worst_calibration(self) -> None:
        pairs = [(0.0, 1)] * 10 + [(1.0, 0)] * 10
        result = calibration_metrics(pairs)
        assert abs(result["ece"] - 1.0) < 1e-10
        assert abs(result["mce"] - 1.0) < 1e-10

    def test_ece_weighted_average(self) -> None:
        """ECE should be a weighted average of per-bin gaps."""
        pairs = [(0.1, 0)] * 5 + [(0.9, 1)] * 5
        result = calibration_metrics(pairs)
        assert result["ece"] >= 0.0
        assert result["ece"] <= 1.0

    def test_mce_at_least_ece(self) -> None:
        pairs = [(0.2, 0), (0.3, 1), (0.7, 1), (0.8, 0)]
        result = calibration_metrics(pairs)
        assert result["mce"] >= result["ece"] - 1e-10

    def test_sharpness_constant_forecast(self) -> None:
        """A constant forecaster has zero sharpness."""
        pairs = [(0.5, 1)] * 10 + [(0.5, 0)] * 10
        result = calibration_metrics(pairs)
        assert abs(result["sharpness"]) < 1e-10

    def test_sharpness_extreme_forecasts(self) -> None:
        """Extreme forecasts (0 and 1) have high sharpness."""
        pairs = [(0.0, 0)] * 5 + [(1.0, 1)] * 5
        result = calibration_metrics(pairs)
        assert result["sharpness"] > 0.2

    def test_sharpness_known_value(self) -> None:
        pairs = [(0.0, 0), (1.0, 1)]
        result = calibration_metrics(pairs)
        assert abs(result["sharpness"] - 0.25) < 1e-10

    def test_empty_pairs(self) -> None:
        result = calibration_metrics([])
        assert result["ece"] == 0.0
        assert result["mce"] == 0.0
        assert result["sharpness"] == 0.0

    def test_single_pair(self) -> None:
        result = calibration_metrics([(0.7, 1)])
        assert result["sharpness"] == 0.0
        assert result["ece"] >= 0.0

    @given(
        pairs=st.lists(
            st.tuples(
                st.floats(min_value=0.0, max_value=1.0),
                st.sampled_from([0, 1]),
            ),
            min_size=1,
            max_size=50,
        ),
    )
    @settings(max_examples=50)
    def test_ece_in_range(self, pairs: list[tuple[float, int]]) -> None:
        result = calibration_metrics(pairs)
        assert 0.0 <= result["ece"] <= 1.0 + 1e-10
        assert 0.0 <= result["mce"] <= 1.0 + 1e-10
        assert result["sharpness"] >= -1e-10


class TestAnalyzeDecomposition:
    def test_returns_murphy_and_calibration(self) -> None:
        resolved = [_rq(f"q{i}", i % 2) for i in range(10)]
        forecasts = {f"q{i}": 0.5 for i in range(10)}
        result = analyze_decomposition(forecasts, resolved)
        assert "murphy" in result
        assert "calibration" in result
        assert "reliability" in result["murphy"]
        assert "ece" in result["calibration"]

    def test_empty_returns_empty_dicts(self) -> None:
        result = analyze_decomposition({}, [])
        assert result["murphy"] == {}
        assert result["calibration"] == {}

    def test_missing_forecasts_default_to_half(self) -> None:
        resolved = [_rq("q1", 1), _rq("q2", 0)]
        result = analyze_decomposition({}, resolved)
        assert result["murphy"]["resolution"] == pytest.approx(0.0, abs=1e-10)
