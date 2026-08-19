"""Tests for CutoffEnvironment integration into the eval pipeline."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

from fetch_data import Question, QuestionSet, ResolvedQuestion
from eval import run_eval, _apply_cutoff


def _dummy_forecaster(question: Question, resolution_date: str | None = None, **kwargs: Any) -> float:
    return 0.5


class TestApplyCutoff:
    def test_question_with_freeze_datetime_gets_temporal_framing(self) -> None:
        q = Question(
            id="q1", source="acled", question="Will X happen?",
            background="Original background",
            freeze_datetime="2024-06-15",
        )
        result = _apply_cutoff(q)
        assert "forecast based on information available as of this date" in result.background.lower()
        assert "2024-06-15" in result.background
        assert "Original background" in result.background

    def test_question_without_freeze_datetime_unchanged(self) -> None:
        q = Question(
            id="q2", source="metaculus", question="Will Y happen?",
            background="Some background",
        )
        result = _apply_cutoff(q)
        assert result.background == "Some background"

    def test_question_with_none_freeze_datetime_unchanged(self) -> None:
        q = Question(
            id="q3", source="metaculus", question="Will Z happen?",
            background="Background text",
            freeze_datetime=None,
        )
        result = _apply_cutoff(q)
        assert result.background == "Background text"

    def test_question_with_empty_background_gets_context(self) -> None:
        q = Question(
            id="q4", source="acled", question="Will A happen?",
            background="",
            freeze_datetime="2024-03-01",
        )
        result = _apply_cutoff(q)
        assert "2024-03-01" in result.background
        assert "forecast based on information available" in result.background.lower()

    def test_preserves_other_fields(self) -> None:
        q = Question(
            id="q5", source="fred", question="GDP growth?",
            background="Econ data",
            resolution_criteria="Resolves YES if...",
            freeze_datetime="2024-09-01",
            resolution_dates=["2024-12-31"],
        )
        result = _apply_cutoff(q)
        assert result.id == "q5"
        assert result.source == "fred"
        assert result.question == "GDP growth?"
        assert result.resolution_criteria == "Resolves YES if..."
        assert result.resolution_dates == ["2024-12-31"]

    def test_original_question_not_mutated(self) -> None:
        q = Question(
            id="q6", source="acled", question="Test?",
            background="Original",
            freeze_datetime="2024-01-01",
        )
        _apply_cutoff(q)
        assert q.background == "Original"

    def test_falls_back_to_forecast_due_date(self) -> None:
        q = Question(
            id="q7", source="acled", question="Test?",
            background="BG",
            freeze_datetime=None,
            forecast_due_date="2024-07-15",
        )
        result = _apply_cutoff(q)
        assert "2024-07-15" in result.background
        assert "forecast based on information available" in result.background.lower()


class TestCutoffInRunEval:
    def test_cutoff_applied_in_pipeline(self, tmp_path: Path, monkeypatch: Any) -> None:
        """run_eval should apply temporal framing to questions with freeze_datetime."""
        import eval as eval_mod

        resolved = [
            ResolvedQuestion(
                id="q0", source="acled", question="Q0",
                outcome=1, forecast_due_date="2024-01-01",
                freeze_datetime="2024-06-15",
                resolution_date="2024-06-29",
            ),
            ResolvedQuestion(
                id="q1", source="metaculus", question="Q1",
                outcome=0, forecast_due_date="2024-01-01",
                freeze_datetime="2024-06-15",
            ),
        ]
        question_sets = [
            QuestionSet(
                forecast_due_date="2024-01-01",
                question_set="set_0",
                questions=[
                    Question(id="q0", source="acled", question="Q0",
                             freeze_datetime="2024-06-15",
                             resolution_dates=["2024-06-29"]),
                    Question(id="q1", source="metaculus", question="Q1",
                             freeze_datetime="2024-06-15"),
                ],
            ),
            QuestionSet(forecast_due_date="2024-02-01", question_set="set_1", questions=[]),
            QuestionSet(forecast_due_date="2024-03-01", question_set="set_2", questions=[]),
        ]

        results_dir = tmp_path / "results"
        results_dir.mkdir()

        seen_questions: dict[str, str] = {}

        def _capturing_forecaster(question: Question, **kwargs: Any) -> float:
            seen_questions[question.id] = question.background or ""
            return 0.5

        monkeypatch.setattr(eval_mod, "RESULTS_DIR", results_dir)
        monkeypatch.setattr(eval_mod, "load_data", lambda: (question_sets, resolved))
        monkeypatch.setattr(eval_mod, "CACHE_DIR", tmp_path / "cache")

        asyncio.run(run_eval(_capturing_forecaster, n_held_out=2, raw=True))

        assert "q0" in seen_questions
        assert "2024-06-15" in seen_questions["q0"]
        assert "forecast based on information available" in seen_questions["q0"].lower()

    def test_cutoff_skipped_without_any_date(self, tmp_path: Path, monkeypatch: Any) -> None:
        """Questions without freeze_datetime or forecast_due_date are unchanged."""
        import eval as eval_mod

        resolved = [
            ResolvedQuestion(
                id="q0", source="metaculus", question="Q0",
                outcome=1, forecast_due_date="2024-01-01",
            ),
        ]
        question_sets = [
            QuestionSet(
                forecast_due_date="2024-01-01",
                question_set="set_0",
                questions=[
                    Question(id="q0", source="metaculus", question="Q0"),
                ],
            ),
            QuestionSet(forecast_due_date="2024-02-01", question_set="set_1", questions=[]),
            QuestionSet(forecast_due_date="2024-03-01", question_set="set_2", questions=[]),
        ]

        results_dir = tmp_path / "results"
        results_dir.mkdir()

        monkeypatch.setattr(eval_mod, "RESULTS_DIR", results_dir)
        monkeypatch.setattr(eval_mod, "load_data", lambda: (question_sets, resolved))
        monkeypatch.setattr(eval_mod, "CACHE_DIR", tmp_path / "cache")

        q_no_date = Question(id="bare", source="metaculus", question="Bare?", background="Just text")
        result = _apply_cutoff(q_no_date)
        assert result.background == "Just text"

    def test_cutoff_uses_forecast_due_date_fallback(self, tmp_path: Path, monkeypatch: Any) -> None:
        """When freeze_datetime is absent but forecast_due_date is set, cutoff uses it."""
        import eval as eval_mod

        resolved = [
            ResolvedQuestion(
                id="q0", source="metaculus", question="Q0",
                outcome=1, forecast_due_date="2024-01-01",
            ),
        ]
        question_sets = [
            QuestionSet(
                forecast_due_date="2024-01-01",
                question_set="set_0",
                questions=[
                    Question(id="q0", source="metaculus", question="Q0"),
                ],
            ),
            QuestionSet(forecast_due_date="2024-02-01", question_set="set_1", questions=[]),
            QuestionSet(forecast_due_date="2024-03-01", question_set="set_2", questions=[]),
        ]

        results_dir = tmp_path / "results"
        results_dir.mkdir()

        seen_questions: dict[str, str] = {}

        def _capturing_forecaster(question: Question, **kwargs: Any) -> float:
            seen_questions[question.id] = question.background or ""
            return 0.5

        monkeypatch.setattr(eval_mod, "RESULTS_DIR", results_dir)
        monkeypatch.setattr(eval_mod, "load_data", lambda: (question_sets, resolved))
        monkeypatch.setattr(eval_mod, "CACHE_DIR", tmp_path / "cache")

        asyncio.run(run_eval(_capturing_forecaster, n_held_out=2, raw=True))

        assert "q0" in seen_questions
        assert "2024-01-01" in seen_questions["q0"]
        assert "forecast based on information available" in seen_questions["q0"].lower()
