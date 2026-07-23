"""Tests for calibrate.py — isotonic regression calibration post-processing."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from calibrate import (
    isotonic_regression,
    calibrate,
    load_calibration_models,
    learn,
    _interpolate,
)


class TestIsotonicRegression:
    def test_monotone_output(self) -> None:
        preds = [0.1, 0.3, 0.2, 0.5, 0.4, 0.7, 0.6, 0.9, 0.8]
        outcomes = [0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 1.0]
        breakpoints = isotonic_regression(preds, outcomes)
        ys = [y for _, y in breakpoints]
        for i in range(len(ys) - 1):
            assert ys[i] <= ys[i + 1], f"Monotonicity violated at index {i}: {ys[i]} > {ys[i+1]}"

    def test_under_confident_pattern(self) -> None:
        """Low predictions that all resolve to 1 should map upward."""
        preds = [0.3, 0.5, 0.7]
        outcomes = [1.0, 1.0, 1.0]
        breakpoints = isotonic_regression(preds, outcomes)
        assert len(breakpoints) >= 1
        for _, y in breakpoints:
            assert y == pytest.approx(1.0)

    def test_perfect_calibration(self) -> None:
        preds = [0.0, 0.5, 1.0]
        outcomes = [0.0, 0.5, 1.0]
        breakpoints = isotonic_regression(preds, outcomes)
        ys = [y for _, y in breakpoints]
        for i in range(len(ys) - 1):
            assert ys[i] <= ys[i + 1]

    def test_empty_input(self) -> None:
        assert isotonic_regression([], []) == []

    def test_single_point(self) -> None:
        breakpoints = isotonic_regression([0.5], [1.0])
        assert len(breakpoints) == 1
        assert breakpoints[0][1] == pytest.approx(1.0)

    def test_all_same_predictions(self) -> None:
        preds = [0.5, 0.5, 0.5]
        outcomes = [0.0, 1.0, 1.0]
        breakpoints = isotonic_regression(preds, outcomes)
        assert len(breakpoints) >= 1
        for _, y in breakpoints:
            assert 0.0 <= y <= 1.0

    def test_violation_merging(self) -> None:
        """When outcomes violate monotonicity, PAVA should merge blocks."""
        preds = [0.2, 0.4, 0.6, 0.8]
        outcomes = [1.0, 0.0, 1.0, 0.0]
        breakpoints = isotonic_regression(preds, outcomes)
        ys = [y for _, y in breakpoints]
        for i in range(len(ys) - 1):
            assert ys[i] <= ys[i + 1]


class TestInterpolate:
    def test_between_breakpoints(self) -> None:
        bps = [(0.3, 0.5), (0.7, 0.9)]
        result = _interpolate(0.5, bps)
        assert result == pytest.approx(0.7)

    def test_at_breakpoint(self) -> None:
        bps = [(0.3, 0.5), (0.7, 0.9)]
        assert _interpolate(0.3, bps) == pytest.approx(0.5)
        assert _interpolate(0.7, bps) == pytest.approx(0.9)

    def test_below_range(self) -> None:
        bps = [(0.3, 0.5), (0.7, 0.9)]
        assert _interpolate(0.1, bps) == pytest.approx(0.5)

    def test_above_range(self) -> None:
        bps = [(0.3, 0.5), (0.7, 0.9)]
        assert _interpolate(0.95, bps) == pytest.approx(0.9)

    def test_single_breakpoint(self) -> None:
        bps = [(0.5, 0.7)]
        assert _interpolate(0.3, bps) == pytest.approx(0.7)
        assert _interpolate(0.9, bps) == pytest.approx(0.7)

    def test_empty_breakpoints(self) -> None:
        assert _interpolate(0.5, []) == pytest.approx(0.5)


class TestCalibrate:
    def test_identity_fallback_no_model(self) -> None:
        result = calibrate(0.42, "nonexistent_source", models={})
        assert result == pytest.approx(0.42)

    def test_identity_fallback_none_models(self, tmp_path: Path) -> None:
        result = calibrate(0.42, "nonexistent_source", models={})
        assert result == pytest.approx(0.42)

    def test_interpolation_with_model(self) -> None:
        models = {"fred": [(0.3, 0.5), (0.7, 0.9)]}
        result = calibrate(0.5, "fred", models=models)
        assert result == pytest.approx(0.7)

    def test_clamp_low(self) -> None:
        models = {"fred": [(0.0, 0.0), (0.1, 0.0)]}
        result = calibrate(0.0, "fred", models=models)
        assert result == pytest.approx(0.001)

    def test_clamp_high(self) -> None:
        models = {"fred": [(0.9, 1.0), (1.0, 1.0)]}
        result = calibrate(1.0, "fred", models=models)
        assert result == pytest.approx(0.999)

    def test_clamp_range(self) -> None:
        models = {"src": [(0.5, 0.5)]}
        result = calibrate(0.5, "src", models=models)
        assert 0.001 <= result <= 0.999


class TestLoadCalibrationModels:
    def test_loads_from_disk(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        cal_dir = tmp_path / "calibration"
        cal_dir.mkdir()
        model_data = {"breakpoints": [[0.3, 0.5], [0.7, 0.9]], "n_points": 100, "mean_shift": 0.1}
        (cal_dir / "fred.json").write_text(json.dumps(model_data))

        monkeypatch.setattr("calibrate.CALIBRATION_DIR", cal_dir)
        models = load_calibration_models()
        assert "fred" in models
        assert models["fred"] == [(0.3, 0.5), (0.7, 0.9)]

    def test_empty_dir(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        cal_dir = tmp_path / "calibration"
        cal_dir.mkdir()
        monkeypatch.setattr("calibrate.CALIBRATION_DIR", cal_dir)
        models = load_calibration_models()
        assert models == {}

    def test_missing_dir(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        cal_dir = tmp_path / "nonexistent"
        monkeypatch.setattr("calibrate.CALIBRATION_DIR", cal_dir)
        models = load_calibration_models()
        assert models == {}


class TestLearn:
    def test_creates_calibration_files(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        cal_dir = tmp_path / "calibration"
        monkeypatch.setattr("calibrate.CALIBRATION_DIR", cal_dir)
        monkeypatch.setattr("calibrate.MIN_DATA_POINTS", 3)

        questions = [
            {"id": f"q{i}", "source": "fred", "question": f"Q{i}"}
            for i in range(5)
        ]
        from fetch_data import QuestionSet, ResolvedQuestion

        qs = QuestionSet(
            forecast_due_date="2024-01-01",
            question_set="test",
            questions=questions,
        )
        resolved = [
            ResolvedQuestion(
                id=f"q{i}", source="fred", question=f"Q{i}",
                outcome=o, resolution_date="2024-02-01",
                forecast_due_date="2024-01-01",
            )
            for i, o in enumerate([1, 0, 1, 0, 1])
        ]

        def mock_load_data() -> tuple:
            return [qs], resolved

        monkeypatch.setattr("fetch_data.load_data", mock_load_data)
        monkeypatch.setattr("fetch_data.join_resolved_questions",
                            lambda qsets, res: resolved)

        forecasts = {f"q{i}": p for i, p in enumerate([0.8, 0.2, 0.7, 0.3, 0.6])}
        outcomes = {f"q{i}": o for i, o in enumerate([1, 0, 1, 0, 1])}
        result_data = {
            "forecasts": forecasts,
            "outcomes": outcomes,
            "metadata": {"question_sets_used": ["2024-01-01"]},
        }
        result_path = tmp_path / "result.json"
        result_path.write_text(json.dumps(result_data))

        learn(str(result_path))

        assert (cal_dir / "fred.json").exists()
        model = json.loads((cal_dir / "fred.json").read_text())
        assert "breakpoints" in model
        assert model["n_points"] == 5

    def test_skips_source_with_few_points(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        cal_dir = tmp_path / "calibration"
        monkeypatch.setattr("calibrate.CALIBRATION_DIR", cal_dir)

        from fetch_data import QuestionSet, ResolvedQuestion

        questions = [{"id": "q0", "source": "fred", "question": "Q0"}]
        qs = QuestionSet(
            forecast_due_date="2024-01-01", question_set="test", questions=questions,
        )
        resolved = [
            ResolvedQuestion(
                id="q0", source="fred", question="Q0",
                outcome=1, resolution_date="2024-02-01",
                forecast_due_date="2024-01-01",
            )
        ]

        monkeypatch.setattr("fetch_data.load_data", lambda: ([qs], resolved))
        monkeypatch.setattr("fetch_data.join_resolved_questions",
                            lambda qsets, res: resolved)

        result_data = {
            "forecasts": {"q0": 0.5},
            "outcomes": {"q0": 1},
            "metadata": {"question_sets_used": ["2024-01-01"]},
        }
        result_path = tmp_path / "result.json"
        result_path.write_text(json.dumps(result_data))

        learn(str(result_path))

        assert not cal_dir.exists() or not list(cal_dir.glob("*.json"))


class TestCompositeIdStripping:
    def test_composite_id_maps_to_base_source(self) -> None:
        """Composite IDs like {base_id}_{date} should use the base question's source."""
        import re
        composite_re = re.compile(r"^(.+)_(\d{4}-\d{2}-\d{2})$")

        m = composite_re.match("abc123_2024-06-15")
        assert m is not None
        assert m.group(1) == "abc123"
        assert m.group(2) == "2024-06-15"

    def test_non_composite_id_no_match(self) -> None:
        import re
        composite_re = re.compile(r"^(.+)_(\d{4}-\d{2}-\d{2})$")
        m = composite_re.match("simple_id")
        assert m is None

    def test_calibration_with_composite_ids(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Calibration should work when forecast IDs are composite."""
        from eval import _apply_calibration
        from fetch_data import Question

        questions = [
            Question(id="base1", source="fred", question="Q1"),
        ]
        models_data = {"fred": [(0.3, 0.5), (0.7, 0.9)]}

        forecasts = {"base1_2024-06-15": 0.5}

        monkeypatch.setattr("eval.load_calibration_models", lambda: models_data)
        result = _apply_calibration(forecasts, questions)
        assert result["base1_2024-06-15"] != 0.5
        assert result["base1_2024-06-15"] == pytest.approx(0.7)


class TestApplyCalibrationIntegration:
    def test_no_models_returns_original(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from eval import _apply_calibration
        from fetch_data import Question

        questions = [Question(id="q1", source="fred", question="Q1")]
        forecasts = {"q1": 0.5}

        monkeypatch.setattr("eval.load_calibration_models", lambda: {})
        result = _apply_calibration(forecasts, questions)
        assert result == forecasts

    def test_calibration_modifies_forecasts(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from eval import _apply_calibration
        from fetch_data import Question

        questions = [Question(id="q1", source="dbnomics", question="Q1")]
        forecasts = {"q1": 0.5}
        models = {"dbnomics": [(0.3, 0.6), (0.7, 0.95)]}

        monkeypatch.setattr("eval.load_calibration_models", lambda: models)
        result = _apply_calibration(forecasts, questions)
        assert result["q1"] != 0.5
        assert result["q1"] == pytest.approx(0.775)
