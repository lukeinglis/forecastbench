"""Tests for error analysis and reporting."""

from __future__ import annotations

import json
import os
import tempfile

from fetch_data import ResolvedQuestion
from analyze import (
    classify_error,
    compute_calibration,
    analyze_run,
    save_results,
    QuestionResult,
)


def _make_resolved(**kwargs) -> ResolvedQuestion:
    defaults = {
        "id": "q1",
        "source": "acled",
        "question": "Will X happen?",
        "outcome": 1,
        "forecast_due_date": "2024-01-01",
    }
    defaults.update(kwargs)
    return ResolvedQuestion(**defaults)


class TestClassifyError:
    def test_well_calibrated_high_confidence_correct(self) -> None:
        assert classify_error(0.8, 1) == "well_calibrated"

    def test_well_calibrated_low_confidence_correct(self) -> None:
        assert classify_error(0.2, 0) == "well_calibrated"

    def test_overconfident_predicted_yes_actual_no(self) -> None:
        assert classify_error(0.9, 0) == "overconfident"

    def test_overconfident_predicted_no_actual_yes(self) -> None:
        assert classify_error(0.1, 1) == "overconfident"

    def test_underconfident(self) -> None:
        assert classify_error(0.6, 1) == "underconfident"

    def test_boundary_well_calibrated(self) -> None:
        assert classify_error(0.7, 1) == "well_calibrated"

    def test_exact_match(self) -> None:
        assert classify_error(1.0, 1) == "well_calibrated"

    def test_exact_zero(self) -> None:
        assert classify_error(0.0, 0) == "well_calibrated"


class TestComputeCalibration:
    def test_returns_correct_number_of_bins(self) -> None:
        results = [
            QuestionResult(
                question_id="q1",
                question_text="Q1",
                source="acled",
                forecast=0.3,
                outcome=0,
                brier_score=0.09,
                error_magnitude=0.3,
                error_type="well_calibrated",
                direction="over",
            ),
            QuestionResult(
                question_id="q2",
                question_text="Q2",
                source="acled",
                forecast=0.7,
                outcome=1,
                brier_score=0.09,
                error_magnitude=0.3,
                error_type="well_calibrated",
                direction="under",
            ),
        ]
        bins = compute_calibration(results, n_bins=5)
        assert len(bins) == 5

    def test_empty_bins_have_zero_questions(self) -> None:
        results = [
            QuestionResult(
                question_id="q1",
                question_text="Q1",
                source="acled",
                forecast=0.15,
                outcome=1,
                brier_score=0.7225,
                error_magnitude=0.85,
                error_type="overconfident",
                direction="under",
            ),
        ]
        bins = compute_calibration(results, n_bins=10)
        non_empty = [b for b in bins if b.n_questions > 0]
        assert len(non_empty) == 1
        assert non_empty[0].n_questions == 1


class TestAnalyzeRun:
    def test_basic_run(self) -> None:
        resolved = [
            _make_resolved(id="q1", outcome=1),
            _make_resolved(id="q2", outcome=0),
            _make_resolved(id="q3", outcome=1),
        ]
        forecasts = {"q1": 0.9, "q2": 0.1, "q3": 0.5}
        result = analyze_run(forecasts, resolved, "test-model")

        assert result.model == "test-model"
        assert result.aggregate.n_questions == 3
        assert result.aggregate.mean_brier > 0
        assert len(result.questions) == 3
        assert len(result.calibration) == 10
        assert len(result.worst_predictions) <= 10

    def test_missing_forecasts_default_to_half(self) -> None:
        resolved = [_make_resolved(id="q1", outcome=1)]
        forecasts: dict[str, float] = {}
        result = analyze_run(forecasts, resolved, "test-model")
        assert result.questions[0].forecast == 0.5

    def test_worst_predictions_sorted_by_brier(self) -> None:
        resolved = [
            _make_resolved(id="q1", outcome=1),
            _make_resolved(id="q2", outcome=0),
        ]
        forecasts = {"q1": 0.1, "q2": 0.9}
        result = analyze_run(forecasts, resolved, "test-model")
        brier_scores = [w.brier_score for w in result.worst_predictions]
        assert brier_scores == sorted(brier_scores, reverse=True)


class TestSaveResults:
    def test_creates_file(self) -> None:
        resolved = [_make_resolved(id="q1", outcome=1)]
        forecasts = {"q1": 0.8}
        result = analyze_run(forecasts, resolved, "test-model")

        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = save_results(result, output_dir=tmpdir)
            assert os.path.exists(filepath)
            data = json.loads(open(filepath).read())
            assert data["model"] == "test-model"
            assert "aggregate" in data
            assert "questions" in data
