"""Multi-model routing forecaster — selects the best LLM per question source."""

from __future__ import annotations

import json
import os
from typing import Any

import litellm

from lab_forecaster import aforecast, aforecast_multi_horizon
from fetch_data import Question
from logging_config import get_logger

litellm.drop_params = True

logger = get_logger("multi_model")

DEFAULT_ROUTING: dict[str, str] = {
    "acled": "vertex_ai/claude-sonnet-4@20250514",
    "wikipedia": "openai/o3-mini",
    "dbnomics": "openai/gpt-4o",
    "fred": "openai/gpt-5-mini",
    "yfinance": "openai/o3-mini",
    "manifold": "openai/gpt-5-mini",
    "polymarket": "openai/gpt-5-mini",
    "metaculus": "vertex_ai/claude-sonnet-4@20250514",
    "infer": "vertex_ai/claude-sonnet-4@20250514",
}
DEFAULT_MODEL = "vertex_ai/claude-sonnet-4@20250514"

_routing_override = os.getenv("FORECAST_MODEL_ROUTING")
MODEL_ROUTING: dict[str, str] = json.loads(_routing_override) if _routing_override else dict(DEFAULT_ROUTING)


def _model_for_source(source: str) -> str:
    return MODEL_ROUTING.get(source.lower(), DEFAULT_MODEL)


async def multi_model_forecast(
    question: Question,
    resolution_date: str | None = None,
    source: str | None = None,
    resolution_dates: Any = None,
    prompt_variant: str = "zero-shot",
) -> float:
    effective_source = source or question.source
    model = _model_for_source(effective_source)
    logger.info("multi_model_route", source=effective_source, model=model, question_id=question.id)
    return await aforecast(
        question,
        resolution_date=resolution_date,
        source=source,
        resolution_dates=resolution_dates,
        prompt_variant=prompt_variant,
        model_override=model,
    )


async def multi_model_forecast_multi_horizon(
    question: Question,
    resolution_dates: list[str],
    source: str | None = None,
    prompt_variant: str = "dataset",
) -> list[float] | None:
    effective_source = source or question.source
    model = _model_for_source(effective_source)
    logger.info("multi_model_route_multi", source=effective_source, model=model, question_id=question.id)
    return await aforecast_multi_horizon(
        question,
        resolution_dates,
        source=source,
        prompt_variant=prompt_variant,
        model_override=model,
    )
