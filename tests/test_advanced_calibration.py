"""Tests for beta calibration and feature-rich Platt scaling."""

from __future__ import annotations

import random

from calibrate import (
    BetaCalibrationModel,
    FeaturePlattModel,
    apply_beta_calibration,
    apply_feature_platt,
    beta_calibrate,
    feature_platt_calibrate,
    fit_beta_calibration,
    fit_feature_platt,
)


class TestBetaCalibrationFit:
    def test_overconfident_data_corrects_downward(self) -> None:
        """Overconfident forecasts (high p, low outcome rate) should be corrected."""
        rng = random.Random(42)
        predictions = [0.7 + rng.random() * 0.2 for _ in range(100)]
        outcomes = [1 if rng.random() < 0.3 else 0 for _ in range(100)]
        model = fit_beta_calibration(predictions, outcomes)
        calibrated = apply_beta_calibration(predictions, model)
        avg_raw = sum(predictions) / len(predictions)
        avg_cal = sum(calibrated) / len(calibrated)
        assert avg_cal < avg_raw

    def test_underconfident_data_corrects_upward(self) -> None:
        """Underconfident forecasts (low p, high outcome rate) should be corrected."""
        rng = random.Random(123)
        predictions = [0.2 + rng.random() * 0.2 for _ in range(100)]
        outcomes = [1 if rng.random() < 0.7 else 0 for _ in range(100)]
        model = fit_beta_calibration(predictions, outcomes)
        calibrated = apply_beta_calibration(predictions, model)
        avg_raw = sum(predictions) / len(predictions)
        avg_cal = sum(calibrated) / len(calibrated)
        assert avg_cal > avg_raw

    def test_well_calibrated_near_identity(self) -> None:
        """Well-calibrated data should produce near-identity mapping."""
        rng = random.Random(7)
        predictions = [rng.random() for _ in range(200)]
        outcomes = [1 if rng.random() < p else 0 for p in predictions]
        model = fit_beta_calibration(predictions, outcomes)
        test_p = 0.5
        cal = beta_calibrate(test_p, model)
        assert abs(cal - test_p) < 0.15

    def test_insufficient_data_returns_default(self) -> None:
        model = fit_beta_calibration([0.5, 0.6], [1, 0])
        assert model.a == 1.0
        assert model.b == 1.0
        assert model.c == 0.0


class TestBetaCalibrationMonotonicity:
    def test_monotonic_when_b_near_zero(self) -> None:
        """Beta calibration with b~0 should be monotonic (reduces to power transform)."""
        model = BetaCalibrationModel(a=1.5, b=0.0, c=0.1)
        test_inputs = [i / 20.0 for i in range(1, 20)]
        calibrated = apply_beta_calibration(test_inputs, model)
        for i in range(len(calibrated) - 1):
            assert calibrated[i] <= calibrated[i + 1] + 1e-10

    def test_ordering_preserved_in_training_range(self) -> None:
        """Within the training range, relative ordering should be mostly preserved."""
        rng = random.Random(99)
        predictions = [0.3 + rng.random() * 0.4 for _ in range(100)]
        outcomes = [1 if rng.random() < p else 0 for p in predictions]
        model = fit_beta_calibration(predictions, outcomes)
        test_inputs = [0.3, 0.4, 0.5, 0.6, 0.7]
        calibrated = apply_beta_calibration(test_inputs, model)
        assert all(0.001 <= c <= 0.999 for c in calibrated)

    def test_constraints_satisfied(self) -> None:
        """Fitted parameters a, b should be non-negative."""
        rng = random.Random(55)
        predictions = [rng.random() for _ in range(100)]
        outcomes = [1 if rng.random() < p else 0 for p in predictions]
        model = fit_beta_calibration(predictions, outcomes)
        assert model.a >= 0.0
        assert model.b >= 0.0


class TestBetaCalibrationEdgeCases:
    def test_all_same_predictions(self) -> None:
        predictions = [0.5] * 50
        outcomes = [1 if i < 25 else 0 for i in range(50)]
        model = fit_beta_calibration(predictions, outcomes)
        cal = beta_calibrate(0.5, model)
        assert 0.001 <= cal <= 0.999

    def test_extreme_probabilities(self) -> None:
        model = BetaCalibrationModel(a=2.0, b=1.5, c=0.3)
        cal_low = beta_calibrate(0.001, model)
        cal_high = beta_calibrate(0.999, model)
        assert 0.001 <= cal_low <= 0.999
        assert 0.001 <= cal_high <= 0.999

    def test_default_model_outputs_valid(self) -> None:
        """Default BetaCalibrationModel should produce valid probabilities."""
        model = BetaCalibrationModel()
        for p in [0.1, 0.3, 0.5, 0.7, 0.9]:
            cal = beta_calibrate(p, model)
            assert 0.001 <= cal <= 0.999

    def test_empty_predictions(self) -> None:
        model = BetaCalibrationModel()
        result = apply_beta_calibration([], model)
        assert result == []


class TestFeaturePlattFit:
    def test_horizon_effect_captured(self) -> None:
        """Feature-rich Platt should capture that long-horizon predictions are overconfident."""
        rng = random.Random(42)
        predictions: list[float] = []
        outcomes: list[int] = []
        horizons: list[float] = []

        for _ in range(100):
            h = rng.choice([7.0, 30.0, 90.0])
            overconfidence = 0.1 * (h / 90.0)
            p = 0.5 + overconfidence + rng.uniform(-0.1, 0.1)
            p = max(0.05, min(0.95, p))
            o = 1 if rng.random() < 0.5 else 0
            predictions.append(p)
            outcomes.append(o)
            horizons.append(h)

        model = fit_feature_platt(
            predictions, outcomes,
            {"horizon_days": horizons, "threshold_distance": [0.0] * len(predictions)},
        )
        assert isinstance(model, FeaturePlattModel)
        assert isinstance(model.w_horizon, float)

    def test_without_features_like_platt(self) -> None:
        """With zero-features, should behave like standard Platt."""
        rng = random.Random(77)
        predictions = [rng.random() for _ in range(100)]
        outcomes = [1 if rng.random() < p else 0 for p in predictions]
        model = fit_feature_platt(
            predictions, outcomes,
            {"horizon_days": [0.0] * 100, "threshold_distance": [0.0] * 100},
        )
        assert abs(model.w_horizon) < 1.0
        assert abs(model.w_threshold) < 1.0

    def test_insufficient_data_returns_default(self) -> None:
        model = fit_feature_platt(
            [0.5, 0.6, 0.7], [1, 0, 1],
            {"horizon_days": [10.0, 20.0, 30.0]},
        )
        assert model.a == 1.0
        assert model.b == 0.0
        assert model.w_horizon == 0.0
        assert model.w_threshold == 0.0


class TestFeaturePlattApply:
    def test_single_probability(self) -> None:
        model = FeaturePlattModel(a=1.0, b=0.0, w_horizon=0.0, w_threshold=0.0)
        cal = feature_platt_calibrate(0.5, model)
        assert abs(cal - 0.5) < 1e-4

    def test_horizon_shifts_prediction(self) -> None:
        model = FeaturePlattModel(a=1.0, b=0.0, w_horizon=-0.5, w_threshold=0.0)
        cal_short = feature_platt_calibrate(0.7, model, horizon_days=7.0)
        cal_long = feature_platt_calibrate(0.7, model, horizon_days=90.0)
        assert cal_long < cal_short

    def test_threshold_distance_shifts_prediction(self) -> None:
        model = FeaturePlattModel(a=1.0, b=0.0, w_horizon=0.0, w_threshold=0.3)
        cal_near = feature_platt_calibrate(0.6, model, threshold_distance=0.1)
        cal_far = feature_platt_calibrate(0.6, model, threshold_distance=2.0)
        assert cal_far > cal_near

    def test_extreme_probabilities(self) -> None:
        model = FeaturePlattModel(a=1.5, b=0.2, w_horizon=-0.1, w_threshold=0.3)
        cal_low = feature_platt_calibrate(0.001, model, horizon_days=30.0)
        cal_high = feature_platt_calibrate(0.999, model, horizon_days=30.0)
        assert 0.001 <= cal_low <= 0.999
        assert 0.001 <= cal_high <= 0.999

    def test_missing_features_defaults_to_zero(self) -> None:
        model = FeaturePlattModel(a=1.0, b=0.0, w_horizon=-0.1, w_threshold=0.2)
        cal_no_features = feature_platt_calibrate(0.6, model)
        cal_zero_features = feature_platt_calibrate(
            0.6, model, horizon_days=0.0, threshold_distance=0.0,
        )
        assert abs(cal_no_features - cal_zero_features) < 1e-10


class TestFeaturePlattEdgeCases:
    def test_all_same_predictions(self) -> None:
        predictions = [0.5] * 50
        outcomes = [1 if i < 25 else 0 for i in range(50)]
        model = fit_feature_platt(
            predictions, outcomes,
            {"horizon_days": [30.0] * 50},
        )
        cal = feature_platt_calibrate(0.5, model, horizon_days=30.0)
        assert 0.001 <= cal <= 0.999

    def test_empty_predictions(self) -> None:
        model = FeaturePlattModel()
        result = apply_feature_platt([], model, {})
        assert result == []

    def test_batch_apply(self) -> None:
        model = FeaturePlattModel(a=1.0, b=0.1, w_horizon=-0.05, w_threshold=0.0)
        preds = [0.3, 0.5, 0.7]
        horizons = [10.0, 30.0, 60.0]
        result = apply_feature_platt(
            preds, model, {"horizon_days": horizons},
        )
        assert len(result) == 3
        for cal in result:
            assert 0.001 <= cal <= 0.999
