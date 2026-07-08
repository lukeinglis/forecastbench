"""Tests for chronological cutoff enforcement."""

from __future__ import annotations

from fetch_data import Question
from cutoff import CutoffEnvironment

SIMULATED_IGNORANCE_WORDS = ["pretend", "ignore", "forget", "act as if", "simulate"]


def _make_question(freeze: str | None = None, background: str = "") -> Question:
    return Question(
        id="q1",
        source="metaculus",
        question="Will X happen by 2025?",
        background=background,
        resolution_criteria="Resolves YES if X happens.",
        freeze_datetime=freeze,
    )


class TestTemporalFraming:
    def test_frame_contains_freeze_datetime(self) -> None:
        env = CutoffEnvironment("2024-06-15")
        q = _make_question()
        context = env.frame_temporal_context(q)
        assert "2024-06-15" in context

    def test_frame_contains_forecast_instruction(self) -> None:
        env = CutoffEnvironment("2024-06-15")
        q = _make_question()
        context = env.frame_temporal_context(q)
        assert "forecast based on information available" in context.lower()

    def test_no_simulated_ignorance_in_frame(self) -> None:
        env = CutoffEnvironment("2024-06-15")
        q = _make_question()
        context = env.frame_temporal_context(q).lower()
        for word in SIMULATED_IGNORANCE_WORDS:
            assert word not in context, f"Simulated ignorance word '{word}' found in temporal framing"

    def test_no_simulated_ignorance_in_prepared_question(self) -> None:
        env = CutoffEnvironment("2024-06-15")
        q = _make_question(background="Some background info")
        prepared = env.prepare_question(q)
        bg = prepared.background.lower()
        for word in SIMULATED_IGNORANCE_WORDS:
            assert word not in bg, f"Simulated ignorance word '{word}' found in prepared question"


class TestPrepareQuestion:
    def test_prepared_question_has_temporal_context(self) -> None:
        env = CutoffEnvironment("2024-06-15")
        q = _make_question(background="Original background")
        prepared = env.prepare_question(q)
        assert "2024-06-15" in prepared.background
        assert "Original background" in prepared.background

    def test_prepared_question_preserves_other_fields(self) -> None:
        env = CutoffEnvironment("2024-06-15")
        q = _make_question(background="bg")
        prepared = env.prepare_question(q)
        assert prepared.id == q.id
        assert prepared.source == q.source
        assert prepared.question == q.question
        assert prepared.resolution_criteria == q.resolution_criteria

    def test_original_question_unchanged(self) -> None:
        env = CutoffEnvironment("2024-06-15")
        q = _make_question(background="Original")
        env.prepare_question(q)
        assert q.background == "Original"

    def test_empty_background_gets_context(self) -> None:
        env = CutoffEnvironment("2024-06-15")
        q = _make_question(background="")
        prepared = env.prepare_question(q)
        assert "2024-06-15" in prepared.background

    def test_different_freeze_dates(self) -> None:
        for date in ["2023-01-01", "2025-12-31", "2024-06-15T12:00:00Z"]:
            env = CutoffEnvironment(date)
            q = _make_question()
            context = env.frame_temporal_context(q)
            assert date in context
