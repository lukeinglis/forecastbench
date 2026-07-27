"""Tests for hierarchical Platt scaling calibration."""

from __future__ import annotations

from pathlib import Path

from calibrate import (
    platt_calibrate as calibrate,
    calibrate_forecasts,
    fit_calibration,
    fit_platt,
    load_calibration,
    save_calibration,
)


class TestCalibrate:
    def test_identity_calibration(self) -> None:
        """a=1, b=0 should return the original probability."""
        for p in [0.1, 0.3, 0.5, 0.7, 0.9]:
            result = calibrate(p, a=1.0, b=0.0)
            assert abs(result - p) < 1e-6

    def test_positive_bias_shifts_up(self) -> None:
        result = calibrate(0.5, a=1.0, b=1.0)
        assert result > 0.5

    def test_negative_bias_shifts_down(self) -> None:
        result = calibrate(0.5, a=1.0, b=-1.0)
        assert result < 0.5

    def test_extreme_probabilities_valid(self) -> None:
        r0 = calibrate(0.001, a=2.0, b=0.5)
        r1 = calibrate(0.999, a=2.0, b=0.5)
        assert 0.0 <= r0 <= 1.0
        assert 0.0 <= r1 <= 1.0

    def test_defaults_are_identity(self) -> None:
        assert abs(calibrate(0.7) - 0.7) < 1e-6


class TestFitPlatt:
    def test_biased_data_corrects_direction(self) -> None:
        """Overconfident forecasts should be calibrated downward."""
        import random
        rng = random.Random(123)
        forecasts = [0.6 + rng.random() * 0.3 for _ in range(60)]
        outcomes = [1 if i < 18 else 0 for i in range(60)]
        a, b = fit_platt(forecasts, outcomes)
        calibrated_high = calibrate(0.8, a, b)
        assert calibrated_high < 0.8

    def test_well_calibrated_near_identity(self) -> None:
        import random
        rng = random.Random(42)
        probs = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
        forecasts = [probs[i % len(probs)] for i in range(70)]
        outcomes = [1 if rng.random() < p else 0 for p in forecasts]
        a, b = fit_platt(forecasts, outcomes)
        calibrated = calibrate(0.5, a, b)
        assert abs(calibrated - 0.5) < 0.15


class TestFitCalibration:
    def test_hierarchical_fallback_for_small_sources(self) -> None:
        """Sources with < min_samples should use regularized global prior."""
        forecasts = {}
        outcomes = {}
        sources = {}
        for i in range(30):
            qid = f"big_{i}"
            forecasts[qid] = 0.7 if i < 15 else 0.3
            outcomes[qid] = 1 if i < 15 else 0
            sources[qid] = "big_source"

        for i in range(5):
            qid = f"small_{i}"
            forecasts[qid] = 0.6
            outcomes[qid] = 1 if i < 3 else 0
            sources[qid] = "small_source"

        params = fit_calibration(forecasts, outcomes, sources)
        assert "big_source" in params
        assert "small_source" in params
        assert "_global" in params

    def test_global_params_present(self) -> None:
        forecasts = {f"q{i}": 0.5 for i in range(20)}
        outcomes = {f"q{i}": i % 2 for i in range(20)}
        sources = {f"q{i}": "src" for i in range(20)}
        params = fit_calibration(forecasts, outcomes, sources)
        assert "_global" in params

    def test_empty_inputs_returns_empty(self) -> None:
        params = fit_calibration({}, {}, {})
        assert params == {}

    def test_min_samples_threshold(self) -> None:
        n = 10
        forecasts = {f"q{i}": 0.6 for i in range(n)}
        outcomes = {f"q{i}": 1 if i < n // 2 else 0 for i in range(n)}
        sources = {f"q{i}": "exact_threshold" for i in range(n)}
        params = fit_calibration(forecasts, outcomes, sources)
        assert "exact_threshold" in params


class TestSaveLoadRoundtrip:
    def test_roundtrip(self, tmp_path: Path) -> None:
        params = {
            "source_a": {"a": 1.2, "b": -0.3},
            "source_b": {"a": 0.9, "b": 0.1},
            "_global": {"a": 1.0, "b": 0.0},
        }
        path = tmp_path / "cal.json"
        save_calibration(params, path)
        loaded = load_calibration(path)
        assert loaded == params

    def test_creates_parent_dirs(self, tmp_path: Path) -> None:
        path = tmp_path / "nested" / "deep" / "cal.json"
        params = {"src": {"a": 1.0, "b": 0.0}}
        save_calibration(params, path)
        assert path.exists()
        loaded = load_calibration(path)
        assert loaded == params

    def test_preserves_float_precision(self, tmp_path: Path) -> None:
        params = {"src": {"a": 1.123456789, "b": -0.987654321}}
        path = tmp_path / "cal.json"
        save_calibration(params, path)
        loaded = load_calibration(path)
        assert abs(loaded["src"]["a"] - 1.123456789) < 1e-9
        assert abs(loaded["src"]["b"] - (-0.987654321)) < 1e-9


class TestCalibrateForecasts:
    def test_per_source_params(self) -> None:
        params = {
            "src_a": {"a": 1.0, "b": 1.0},
            "src_b": {"a": 1.0, "b": -1.0},
        }
        forecasts = {"q1": 0.5, "q2": 0.5}
        sources = {"q1": "src_a", "q2": "src_b"}
        result = calibrate_forecasts(forecasts, sources, params)
        assert result["q1"] > 0.5
        assert result["q2"] < 0.5

    def test_unknown_source_uses_global(self) -> None:
        params = {
            "_global": {"a": 1.0, "b": 0.0},
            "src_a": {"a": 1.0, "b": 1.0},
        }
        forecasts = {"q1": 0.7}
        sources = {"q1": "unknown"}
        result = calibrate_forecasts(forecasts, sources, params)
        assert abs(result["q1"] - 0.7) < 1e-6

    def test_preserves_all_question_ids(self) -> None:
        params = {"_global": {"a": 1.0, "b": 0.0}}
        forecasts = {"q1": 0.3, "q2": 0.5, "q3": 0.8}
        sources = {"q1": "src", "q2": "src", "q3": "src"}
        result = calibrate_forecasts(forecasts, sources, params)
        assert set(result.keys()) == {"q1", "q2", "q3"}
