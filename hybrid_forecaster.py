"""Hybrid forecaster — routes to belief state vs baseline per source."""

from __future__ import annotations

import os
from typing import Any

from baseline_agent import aforecast, aforecast_multi_horizon
from belief_forecaster import belief_forecast, belief_forecast_multi_horizon
from fetch_data import Question
from logging_config import get_logger

logger = get_logger("hybrid_forecaster")

BELIEF_SOURCES = frozenset(
    os.getenv("FORECAST_BELIEF_SOURCES", "acled,fred,yfinance,metaculus,dbnomics").split(",")
)


def _use_belief(source: str) -> bool:
    return source.lower() in BELIEF_SOURCES


async def hybrid_forecast(
    question: Question,
    resolution_date: str | None = None,
    source: str | None = None,
    resolution_dates: Any = None,
    prompt_variant: str = "zero-shot",
) -> float:
    effective_source = source or question.source
    if _use_belief(effective_source):
        logger.info("hybrid_route_belief", question_id=question.id, source=effective_source)
        return await belief_forecast(
            question,
            resolution_date=resolution_date,
            source=source,
            resolution_dates=resolution_dates,
            prompt_variant=prompt_variant,
        )
    else:
        logger.info("hybrid_route_baseline", question_id=question.id, source=effective_source)
        return await aforecast(
            question,
            resolution_date=resolution_date,
            source=source,
            resolution_dates=resolution_dates,
            prompt_variant=prompt_variant,
        )


async def hybrid_forecast_multi_horizon(
    question: Question,
    resolution_dates: list[str],
    source: str | None = None,
    prompt_variant: str = "dataset",
) -> list[float] | None:
    effective_source = source or question.source
    if _use_belief(effective_source):
        logger.info("hybrid_multi_route_belief", question_id=question.id, source=effective_source)
        return await belief_forecast_multi_horizon(
            question,
            resolution_dates,
            source=source,
            prompt_variant=prompt_variant,
        )
    else:
        logger.info("hybrid_multi_route_baseline", question_id=question.id, source=effective_source)
        return await aforecast_multi_horizon(
            question,
            resolution_dates,
            source=source,
            prompt_variant=prompt_variant,
        )
