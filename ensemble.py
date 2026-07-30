"""Multi-model ensemble forecaster.

Fans out forecasts to multiple LLMs in parallel and aggregates via log-odds
extremization. Event sources (acled, wikipedia) use only the primary model.

Aggregation methods controlled by FORECAST_ENSEMBLE_AGGREGATION env var:
- 'mean': arithmetic mean (default)
- 'geometric_mean_odds': geometric mean of odds ratios
- 'trimmed_mean': drop min/max then arithmetic mean (N>=4, else falls back to mean)
- 'extremized': log-odds extremization with FORECAST_EXTREMIZE_GAMMA (legacy default)
"""

from __future__ import annotations

import asyncio
import math
import os
from typing import Any

import litellm

from baseline_agent import (
    MODEL as PRIMARY_MODEL,
    _build_prompt,
    _ensure_vertex_credentials,
    _extract_probabilities,
    _extract_with_llm,
    _parse_probability,
)
from fetch_data import Question
from logging_config import get_logger

logger = get_logger("ensemble")

EXTREMIZE_GAMMA = float(os.getenv("FORECAST_EXTREMIZE_GAMMA", "1.5"))

ENSEMBLE_AGGREGATION = os.getenv("FORECAST_ENSEMBLE_AGGREGATION", "mean")

ENSEMBLE_MODELS = os.getenv(
    "FORECAST_ENSEMBLE_MODELS",
    "vertex_ai/claude-sonnet-4@20250514,openai/gpt-4o",
).split(",")

EVENT_SOURCES = frozenset(["acled", "wikipedia"])


def _arithmetic_mean(predictions: list[float]) -> float:
    return sum(predictions) / len(predictions)


def _geometric_mean_odds(predictions: list[float]) -> float:
    EPS = 0.02
    clamped = [max(EPS, min(1 - EPS, p)) for p in predictions]
    odds = [p / (1 - p) for p in clamped]
    clipped_odds = [max(0.01, min(100.0, o)) for o in odds]
    log_mean = sum(math.log(o) for o in clipped_odds) / len(clipped_odds)
    gmean_odds = math.exp(log_mean)
    result = gmean_odds / (1.0 + gmean_odds)
    return max(0.02, min(0.98, result))


def _trimmed_mean(predictions: list[float]) -> float:
    if len(predictions) < 4:
        return _arithmetic_mean(predictions)
    sorted_preds = sorted(predictions)
    trimmed = sorted_preds[1:-1]
    return sum(trimmed) / len(trimmed)


def _extremized_aggregate(predictions: list[float], gamma: float) -> float:
    EPS = 1e-6
    clamped = [max(EPS, min(1 - EPS, p)) for p in predictions]
    mean_logit = sum(math.log(p / (1 - p)) for p in clamped) / len(clamped)
    return 1.0 / (1.0 + math.exp(-gamma * mean_logit))


def aggregate_predictions(
    predictions: list[float],
    method: str = ENSEMBLE_AGGREGATION,
    gamma: float = EXTREMIZE_GAMMA,
) -> float:
    if len(predictions) == 1:
        return predictions[0]
    if method == "geometric_mean_odds":
        return _geometric_mean_odds(predictions)
    if method == "trimmed_mean":
        return _trimmed_mean(predictions)
    if method == "extremized":
        return _extremized_aggregate(predictions, gamma)
    return _arithmetic_mean(predictions)


def _aggregate_forecasts(predictions: list[float], gamma: float = EXTREMIZE_GAMMA) -> float:
    if len(predictions) == 1:
        return predictions[0]
    if ENSEMBLE_AGGREGATION == "geometric_mean_odds":
        return _geometric_mean_odds(predictions)
    if ENSEMBLE_AGGREGATION == "trimmed_mean":
        return _trimmed_mean(predictions)
    if ENSEMBLE_AGGREGATION == "extremized":
        return _extremized_aggregate(predictions, gamma)
    if ENSEMBLE_AGGREGATION == "mean":
        return _arithmetic_mean(predictions)
    EPS = 1e-6
    clamped = [max(EPS, min(1 - EPS, p)) for p in predictions]
    mean_logit = sum(math.log(p / (1 - p)) for p in clamped) / len(clamped)
    return 1.0 / (1.0 + math.exp(-gamma * mean_logit))


async def _single_model_forecast(
    model: str,
    prompt: str,
    question_id: str,
) -> float:
    if model.startswith("vertex_ai/"):
        _ensure_vertex_credentials()
    response = await litellm.acompletion(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
        timeout=60,
    )
    text = response.choices[0].message.content or ""
    prob = _parse_probability(text)
    logger.debug("model_forecast", model=model, question_id=question_id, probability=prob)
    return prob


async def ensemble_forecast(
    question: Question,
    resolution_date: str | None = None,
    source: str | None = None,
    resolution_dates: Any = None,
    prompt_variant: str = "zero-shot",
) -> float:
    effective_source = source or question.source

    prompt = _build_prompt(
        question,
        resolution_date=resolution_date,
        source=source,
        resolution_dates=resolution_dates,
        prompt_variant=prompt_variant,
    )

    if effective_source.lower() in EVENT_SOURCES:
        logger.info("ensemble_event_source", question_id=question.id, source=effective_source, model=PRIMARY_MODEL)
        _ensure_vertex_credentials()
        response = await litellm.acompletion(
            model=PRIMARY_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            timeout=60,
        )
        text = response.choices[0].message.content or ""
        return _parse_probability(text)

    models = ENSEMBLE_MODELS
    logger.info("ensemble_start", question_id=question.id, n_models=len(models))

    tasks = [_single_model_forecast(m, prompt, question.id) for m in models]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    predictions: list[float] = []
    for model, result in zip(models, results):
        if isinstance(result, BaseException):
            logger.warning("ensemble_model_failed", model=model, question_id=question.id, error=str(result))
        else:
            predictions.append(result)

    if not predictions:
        raise ValueError(f"All {len(models)} ensemble models failed for question {question.id}")

    aggregate = _aggregate_forecasts(predictions)
    logger.info(
        "ensemble_complete",
        question_id=question.id,
        individual=predictions,
        aggregate=aggregate,
        n_succeeded=len(predictions),
        n_total=len(models),
    )
    return aggregate


async def _single_model_multi_horizon(
    model: str,
    prompt: str,
    question_id: str,
    n_horizons: int,
) -> list[float] | None:
    try:
        if model.startswith("vertex_ai/"):
            _ensure_vertex_credentials()
        response = await litellm.acompletion(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            timeout=120,
        )
        text = response.choices[0].message.content or ""

        probs = _extract_probabilities(text, n_horizons)
        if probs is not None:
            return probs

        probs = await _extract_with_llm(text, n_horizons)
        return probs
    except Exception:
        logger.warning("ensemble_multi_horizon_error", model=model, question_id=question_id, exc_info=True)
        return None


async def ensemble_forecast_multi_horizon(
    question: Question,
    resolution_dates: list[str],
    source: str | None = None,
    prompt_variant: str = "dataset",
) -> list[float] | None:
    effective_source = source or question.source
    n_horizons = len(resolution_dates)

    prompt = _build_prompt(
        question,
        source=source,
        resolution_dates=resolution_dates,
        prompt_variant=prompt_variant,
    )

    if effective_source.lower() in EVENT_SOURCES:
        logger.info("ensemble_multi_horizon_event_source", question_id=question.id, source=effective_source)
        return await _single_model_multi_horizon(PRIMARY_MODEL, prompt, question.id, n_horizons)

    models = ENSEMBLE_MODELS
    logger.info("ensemble_multi_horizon_start", question_id=question.id, n_models=len(models), n_horizons=n_horizons)

    tasks = [_single_model_multi_horizon(m, prompt, question.id, n_horizons) for m in models]
    results = await asyncio.gather(*tasks)

    successful: list[list[float]] = []
    for model, result in zip(models, results):
        if result is not None:
            successful.append(result)
        else:
            logger.warning("ensemble_multi_horizon_model_failed", model=model, question_id=question.id)

    if not successful:
        logger.warning("ensemble_multi_horizon_all_failed", question_id=question.id)
        return None

    aggregated = [
        _aggregate_forecasts([model_probs[i] for model_probs in successful])
        for i in range(n_horizons)
    ]
    logger.info(
        "ensemble_multi_horizon_complete",
        question_id=question.id,
        n_horizons=n_horizons,
        n_succeeded=len(successful),
        n_total=len(models),
    )
    return aggregated
