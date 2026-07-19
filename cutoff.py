"""Chronological cutoff enforcement via temporal framing."""

from __future__ import annotations

from fetch_data import Question
from logging_config import get_logger

logger = get_logger("cutoff")


class CutoffEnvironment:
    """Enforces chronological data cutoff using temporal framing.

    Uses temporal context framing rather than simulated ignorance,
    per arXiv:2601.13717 which shows simulated ignorance fails with
    a 52% performance gap.
    """

    def __init__(self, freeze_datetime: str, display_date: str | None = None) -> None:
        self.freeze_datetime = freeze_datetime
        self.display_date = display_date or freeze_datetime
        logger.info("cutoff_environment_created", cutoff_date=freeze_datetime, display_date=self.display_date)

    def frame_temporal_context(self, question: Question) -> str:
        return (
            f"Today's Date: {self.display_date}. "
            "You should forecast based on information available as of this date."
        )

    def prepare_question(self, question: Question) -> Question:
        temporal_context = self.frame_temporal_context(question)
        new_background = f"{temporal_context}\n\n{question.background}" if question.background else temporal_context
        logger.debug("cutoff_applied", question_id=question.id, cutoff_date=self.freeze_datetime)
        return question.model_copy(update={"background": new_background})
