"""Dummy forecaster: returns 0.5 for every question."""

from __future__ import annotations

from typing import Any

from fetch_data import Question
from logging_config import get_logger

logger = get_logger("dummy_forecaster")


def forecast(
    question: Question,
    resolution_date: str | None = None,
    source: str | None = None,
    resolution_dates: Any = None,
    prompt_variant: str = "zero-shot",
) -> float:
    """Return 0.5 regardless of the question."""
    logger.debug("dummy_forecast", question_id=question.id, probability=0.5)
    return 0.5


if __name__ == "__main__":
    import asyncio
    from eval import run_eval
    asyncio.run(run_eval(forecast))
