"""Tests for submit.py."""

from __future__ import annotations

from pathlib import Path

import pytest

from fetch_data import Question, ResolvedQuestion
from submit import (
    SubmissionMetadata,
    assemble_submission,
    save_submission,
    validate_coverage,
)


def _make_resolved(
    qid: str,
    source: str,
    outcome: int,
    due: str = "2024-01-01",
    resolution_date: str | None = None,
) -> ResolvedQuestion:
    return ResolvedQuestion(
        id=qid,
        source=source,
        question=f"Q {qid}",
        outcome=outcome,
        forecast_due_date=due,
        resolution_date=resolution_date,
    )


def _make_question(
    qid: str,
    source: str,
    due: str = "2024-01-01",
) -> Question:
    return Question(
        id=qid,
        source=source,
        question=f"Q {qid}",
        forecast_due_date=due,
    )


def _make_metadata(question_set: str = "2024-01-01", org: str = "test-org") -> SubmissionMetadata:
    return SubmissionMetadata(
        organization=org,
        model="test-model",
        model_organization="test-model-org",
        question_set=question_set,
    )


class TestAssembleSubmission:
    def test_assemble_submission(self) -> None:
        """Build submission and verify JSON structure has all required fields."""
        questions = [
            _make_resolved("q1", "acled", 1, resolution_date="2024-02-01"),
            _make_resolved("q2", "metaculus", 0),
        ]
        forecasts = {"q1": 0.8, "q2": 0.3}
        metadata = _make_metadata()

        result = assemble_submission(forecasts, questions, metadata)

        assert result["organization"] == "test-org"
        assert result["model"] == "test-model"
        assert result["model_organization"] == "test-model-org"
        assert result["question_set"] == "2024-01-01"
        assert len(result["forecasts"]) == 2

        entry_q1 = next(e for e in result["forecasts"] if e["id"] == "q1")
        assert entry_q1["forecast"] == 0.8
        assert entry_q1["source"] == "acled"
        assert entry_q1["resolution_date"] == "2024-02-01"

        entry_q2 = next(e for e in result["forecasts"] if e["id"] == "q2")
        assert entry_q2["forecast"] == 0.3
        assert "resolution_date" not in entry_q2

    def test_assemble_missing_forecasts_default_05(self) -> None:
        """Questions without forecasts get 0.5."""
        questions = [
            _make_resolved("q1", "acled", 1),
            _make_resolved("q2", "acled", 0),
            _make_resolved("q3", "metaculus", 1),
        ]
        forecasts = {"q1": 0.9}  # q2 and q3 missing
        metadata = _make_metadata()

        result = assemble_submission(forecasts, questions, metadata)

        entry_q2 = next(e for e in result["forecasts"] if e["id"] == "q2")
        assert entry_q2["forecast"] == 0.5

        entry_q3 = next(e for e in result["forecasts"] if e["id"] == "q3")
        assert entry_q3["forecast"] == 0.5


class TestValidateCoverage:
    def test_validate_coverage_full(self) -> None:
        """100% coverage passes."""
        questions = [
            _make_question("d1", "acled"),
            _make_question("d2", "acled"),
            _make_question("m1", "metaculus"),
            _make_question("m2", "polymarket"),
        ]
        submission = {"forecasts": [
            {"id": "d1", "forecast": 0.8}, {"id": "d2", "forecast": 0.2},
            {"id": "m1", "forecast": 0.7}, {"id": "m2", "forecast": 0.3},
        ]}

        result = validate_coverage(submission, questions)

        assert result.passes is True
        assert result.market_coverage == 1.0
        assert result.dataset_coverage == 1.0
        assert result.market_total == 2
        assert result.market_covered == 2
        assert result.dataset_total == 2
        assert result.dataset_covered == 2

    def test_validate_coverage_below_threshold(self) -> None:
        """<95% coverage fails."""
        market_qs = [_make_question(f"m{i}", "metaculus") for i in range(20)]
        dataset_qs = [_make_question(f"d{i}", "acled") for i in range(10)]
        all_qs = market_qs + dataset_qs

        submission = {"forecasts": [
            *({"id": f"m{i}", "forecast": 0.5} for i in range(18)),
            *({"id": f"d{i}", "forecast": 0.5} for i in range(10)),
        ]}

        result = validate_coverage(submission, all_qs)
        assert result.passes is False
        assert result.market_coverage == 0.9

    def test_validate_coverage_at_threshold(self) -> None:
        """Exactly 95% passes."""
        market_qs = [_make_question(f"m{i}", "metaculus") for i in range(20)]
        dataset_qs = [_make_question(f"d{i}", "acled") for i in range(20)]
        all_qs = market_qs + dataset_qs

        submission = {"forecasts": [
            *({"id": f"m{i}", "forecast": 0.5} for i in range(19)),
            *({"id": f"d{i}", "forecast": 0.5} for i in range(19)),
        ]}

        result = validate_coverage(submission, all_qs)
        assert result.passes is True
        assert result.market_coverage == 0.95
        assert result.dataset_coverage == 0.95

    def test_validate_coverage_empty_category(self) -> None:
        """No market questions, dataset-only still works."""
        questions = [
            _make_question("d1", "acled"),
            _make_question("d2", "acled"),
        ]
        submission = {"forecasts": [
            {"id": "d1", "forecast": 0.8}, {"id": "d2", "forecast": 0.2},
        ]}

        result = validate_coverage(submission, questions)
        assert result.passes is True
        assert result.market_coverage == 0.0
        assert result.market_total == 0
        assert result.dataset_total == 2

    def test_validate_coverage_empty_questions(self) -> None:
        """Empty question list produces 0.0 coverage and fails."""
        submission = {"forecasts": [{"id": "q1", "forecast": 0.5}]}
        result = validate_coverage(submission, [])
        assert result.market_coverage == 0.0
        assert result.dataset_coverage == 0.0
        assert result.passes is False

    def test_validate_coverage_accepts_resolved_questions(self) -> None:
        """Backward compatibility: ResolvedQuestion objects still work."""
        questions = [
            _make_resolved("d1", "acled", 1),
            _make_resolved("m1", "metaculus", 0),
        ]
        submission = {"forecasts": [
            {"id": "d1", "forecast": 0.8}, {"id": "m1", "forecast": 0.7},
        ]}
        result = validate_coverage(submission, questions)
        assert result.passes is True


class TestSaveSubmission:
    def test_save_submission_file_naming(self, tmp_path: Path) -> None:
        """Verify filename follows ForecastBench convention."""
        submission = {
            "organization": "test-org",
            "question_set": "2024-01-01",
            "forecasts": [{"id": "q1", "forecast": 0.5}],
        }

        path = save_submission(submission, output_dir=tmp_path)

        assert path.name == "2024-01-01.test-org.1.json"
        assert path.exists()
        assert path.parent == tmp_path

    def test_save_submission_sequence_numbering(self, tmp_path: Path) -> None:
        """Multiple saves increment N."""
        submission = {
            "organization": "test-org",
            "question_set": "2024-01-01",
            "forecasts": [{"id": "q1", "forecast": 0.5}],
        }

        p1 = save_submission(submission, output_dir=tmp_path)
        p2 = save_submission(submission, output_dir=tmp_path)
        p3 = save_submission(submission, output_dir=tmp_path)

        assert p1.name == "2024-01-01.test-org.1.json"
        assert p2.name == "2024-01-01.test-org.2.json"
        assert p3.name == "2024-01-01.test-org.3.json"

    def test_save_submission_max_three(self, tmp_path: Path) -> None:
        """4th submission raises ValueError."""
        submission = {
            "organization": "test-org",
            "question_set": "2024-01-01",
            "forecasts": [{"id": "q1", "forecast": 0.5}],
        }

        save_submission(submission, output_dir=tmp_path)
        save_submission(submission, output_dir=tmp_path)
        save_submission(submission, output_dir=tmp_path)

        with pytest.raises(ValueError, match="max 3"):
            save_submission(submission, output_dir=tmp_path)
