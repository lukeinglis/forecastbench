"""Tests for error analysis module."""

from __future__ import annotations

import json
from pathlib import Path

from fetch_data import ResolvedQuestion
from analyze import AnalysisResult, run_analysis, save_analysis


def _make_resolved(
    qid: str, source: str, outcome: int, question: str = "Test?"
) -> ResolvedQuestion:
    return ResolvedQuestion(
        id=qid,
        source=source,
        question=question,
        outcome=outcome,
        forecast_due_date="2024-01-01",
    )


class TestRunAnalysis:
    def test_per_source_breakdown(self) -> None:
        resolved = [
            _make_resolved("q1", "acled", 1),
            _make_resolved("q2", "acled", 0),
            _make_resolved("q3", "metaculus", 1),
            _make_resolved("q4", "metaculus", 0),
        ]
        forecasts = {"q1": 0.9, "q2": 0.1, "q3": 0.5, "q4": 0.5}

        result = run_analysis(forecasts, resolved)

        assert "acled" in result.per_source_scores
        assert "metaculus" in result.per_source_scores
        assert isinstance(result.per_source_scores["acled"], float)
        assert isinstance(result.per_source_scores["metaculus"], float)

    def test_worst_questions_sorted_descending(self) -> None:
        resolved = [
            _make_resolved("q1", "acled", 1),
            _make_resolved("q2", "acled", 0),
            _make_resolved("q3", "acled", 1),
        ]
        forecasts = {"q1": 0.1, "q2": 0.9, "q3": 0.9}

        result = run_analysis(forecasts, resolved)

        scores = [q["brier_score"] for q in result.worst_questions]
        assert scores == sorted(scores, reverse=True)

    def test_worst_questions_capped_at_20(self) -> None:
        resolved = [
            _make_resolved(f"q{i}", "acled", i % 2)
            for i in range(30)
        ]
        forecasts = {f"q{i}": 0.5 for i in range(30)}

        result = run_analysis(forecasts, resolved)

        assert len(result.worst_questions) <= 20

    def test_worst_questions_fields(self) -> None:
        resolved = [_make_resolved("q1", "acled", 1)]
        forecasts = {"q1": 0.3}

        result = run_analysis(forecasts, resolved)

        q = result.worst_questions[0]
        assert "id" in q
        assert "question" in q
        assert "brier_score" in q
        assert "forecast" in q
        assert "outcome" in q

    def test_calibration_buckets_structure(self) -> None:
        resolved = [
            _make_resolved(f"q{i}", "acled", i % 2)
            for i in range(20)
        ]
        forecasts = {f"q{i}": i / 20.0 for i in range(20)}

        result = run_analysis(forecasts, resolved)

        assert len(result.calibration_buckets) == 10
        for bucket in result.calibration_buckets:
            assert "bucket_range" in bucket
            assert "predicted_mean" in bucket
            assert "actual_frequency" in bucket
            assert "count" in bucket

    def test_calibration_bucket_ranges(self) -> None:
        resolved = [_make_resolved("q1", "acled", 1)]
        forecasts = {"q1": 0.5}

        result = run_analysis(forecasts, resolved)

        ranges = [b["bucket_range"] for b in result.calibration_buckets]
        assert ranges[0] == "0.0-0.1"
        assert ranges[9] == "0.9-1.0"

    def test_missing_forecast_defaults_to_half(self) -> None:
        resolved = [_make_resolved("q1", "acled", 1)]
        forecasts: dict[str, float] = {}

        result = run_analysis(forecasts, resolved)

        assert result.worst_questions[0]["forecast"] == 0.5


class TestSaveAnalysis:
    def test_creates_json_file(self, tmp_path: Path) -> None:
        result = AnalysisResult(
            per_source_scores={"acled": 0.25},
            worst_questions=[{"id": "q1", "question": "Test?", "brier_score": 0.81,
                              "forecast": 0.1, "outcome": 1}],
            calibration_buckets=[{"bucket_range": "0.0-0.1", "predicted_mean": 0.05,
                                  "actual_frequency": 0.5, "count": 2}],
        )

        save_analysis(result, tmp_path)

        json_path = tmp_path / "analysis.json"
        assert json_path.exists()
        data = json.loads(json_path.read_text())
        assert "per_source_scores" in data
        assert "worst_questions" in data
        assert "calibration_buckets" in data

    def test_creates_markdown_file(self, tmp_path: Path) -> None:
        result = AnalysisResult(
            per_source_scores={"acled": 0.25},
            worst_questions=[{"id": "q1", "question": "Test?", "brier_score": 0.81,
                              "forecast": 0.1, "outcome": 1}],
            calibration_buckets=[{"bucket_range": "0.0-0.1", "predicted_mean": 0.05,
                                  "actual_frequency": 0.5, "count": 2}],
        )

        save_analysis(result, tmp_path)

        md_path = tmp_path / "analysis.md"
        assert md_path.exists()
        content = md_path.read_text()
        assert "# ForecastBench Error Analysis" in content
        assert "Per-Source" in content
        assert "Calibration" in content
