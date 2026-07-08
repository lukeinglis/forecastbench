"""Chronological cutoff enforcement via temporal framing."""

from __future__ import annotations

from fetch_data import Question


class CutoffEnvironment:
    """Enforces chronological data cutoff using temporal framing.

    Uses temporal context framing rather than simulated ignorance,
    per arXiv:2601.13717 which shows simulated ignorance fails with
    a 52% performance gap.
    """

    def __init__(self, freeze_datetime: str) -> None:
        self.freeze_datetime = freeze_datetime

    def frame_temporal_context(self, question: Question) -> str:
        return (
            f"Current date: {self.freeze_datetime}. "
            "You should forecast based on information available as of this date."
        )

    def prepare_question(self, question: Question) -> Question:
        temporal_context = self.frame_temporal_context(question)
        new_background = f"{temporal_context}\n\n{question.background}" if question.background else temporal_context
        return question.model_copy(update={"background": new_background})
