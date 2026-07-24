"""Sub-question decomposition forecaster for complex questions."""

from __future__ import annotations

import re
from typing import Any

import litellm

from baseline_agent import (
    MAX_TOKENS,
    MODEL,
    TEMPERATURE,
    _ensure_vertex_credentials,
    _extract_probabilities,
    _extract_with_llm,
    _parse_probability,
    _save_response_log,
)
from fetch_data import MARKET_SOURCES, Question
from logging_config import get_logger

logger = get_logger("decompose")

DECOMPOSITION_PROMPT = """\
You are decomposing a forecasting question into key factors.

Question: {question}
Background: {background}
Resolution Criteria: {resolution_criteria}

List 3-5 key factors that would influence the probability of this question resolving YES.
For each factor, assess the current state and whether it increases or decreases the probability.

Format each factor as:
FACTOR: [description]
DIRECTION: [increases|decreases|neutral]
WEIGHT: [high|medium|low]

After listing all factors, provide your final probability estimate.
Output your final answer as a number between 0 and 1 with asterisks: *p*
"""

DECOMPOSITION_MARKET_PROMPT = """\
You are an expert superforecaster decomposing a forecasting question into key factors.

Question: {question}
Background: {background}
Resolution Criteria: {resolution_criteria}

Today's Date: {today_date}
Question resolution date: {resolution_date}

List 3-5 key factors that would influence the probability of this question resolving YES.
For each factor, assess the current state and whether it increases or decreases the probability.

Format each factor as:
FACTOR: [description]
DIRECTION: [increases|decreases|neutral]
WEIGHT: [high|medium|low]

After listing all factors, provide your final probability estimate.
You MUST give a probability estimate between 0 and 1 UNDER ALL CIRCUMSTANCES.
Output your final answer as a number between 0 and 1 with asterisks: *p*
"""

DECOMPOSITION_DATASET_PROMPT = """\
You are an expert superforecaster decomposing a forecasting question into key factors.

You're going to predict the probability of the following potential outcome at each of the resolution dates.

Question: {question}
Background: {background}
Resolution Criteria: {resolution_criteria}

Current value on {freeze_datetime}: {freeze_datetime_value}
Value Explanation: {freeze_datetime_value_explanation}

Today's Date: {today_date}
Question resolution dates: {list_of_resolution_dates}

For this question, first list 3-5 key factors that would influence the probability.
For each factor, assess the current state and whether it increases or decreases the probability.

Format each factor as:
FACTOR: [description]
DIRECTION: [increases|decreases|neutral]
WEIGHT: [high|medium|low]

After listing all factors, provide your final probability estimate for each resolution date.
Output your answer (a number between 0 and 1) with an asterisk at the beginning and end of the decimal. \
(For example, if there are n resolution dates, you would output different *p* for each resolution date) \
Do not output anything else after the probabilities.
"""

_CONDITION_WORDS = re.compile(r"\b(?:and|or|if|unless|provided that|conditional on)\b", re.IGNORECASE)


def is_complex_question(question: Question) -> bool:
    """Heuristic: questions with long resolution criteria, multiple conditions, or combination_of."""
    if question.combination_of:
        return True
    criteria = question.resolution_criteria or ""
    if len(criteria) > 200:
        return True
    if len(_CONDITION_WORDS.findall(criteria)) >= 2:
        return True
    return False


def _build_decomposition_prompt(
    question: Question,
    resolution_date: str | None = None,
    source: str | None = None,
    resolution_dates: Any = None,
) -> str:
    effective_source = source or question.source
    is_market = effective_source.lower() in MARKET_SOURCES

    background = question.background or ""
    mrc = getattr(question, "market_info_resolution_criteria", None)
    if mrc and mrc != "N/A":
        background = (background + "\n" + mrc) if background else mrc

    today_date = (
        getattr(question, "forecast_due_date", None)
        or question.freeze_datetime
        or ""
    )

    if not is_market:
        effective_rd = resolution_dates or getattr(question, "resolution_dates", None)
        dates_list: list[str] = []
        if effective_rd and isinstance(effective_rd, list):
            dates_list = [str(d) for d in effective_rd if d and str(d).upper() != "N/A"]

        fv = getattr(question, "freeze_datetime_value", None)
        fd = question.freeze_datetime or ""

        return DECOMPOSITION_DATASET_PROMPT.format(
            question=question.question,
            background=background,
            resolution_criteria=question.resolution_criteria or "",
            freeze_datetime=fd,
            freeze_datetime_value=fv if fv is not None else "",
            freeze_datetime_value_explanation=getattr(question, "freeze_datetime_value_explanation", None) or "",
            today_date=today_date,
            list_of_resolution_dates=dates_list,
        )

    effective_resolution_date = resolution_date
    if not effective_resolution_date and is_market:
        effective_resolution_date = getattr(question, "market_info_close_datetime", None)

    return DECOMPOSITION_MARKET_PROMPT.format(
        question=question.question,
        background=background,
        resolution_criteria=question.resolution_criteria or "",
        today_date=today_date,
        resolution_date=effective_resolution_date or "",
    )


async def decomposed_forecast(
    question: Question,
    resolution_date: str | None = None,
    source: str | None = None,
    resolution_dates: Any = None,
    prompt_variant: str = "zero-shot",
) -> float:
    """Single-horizon decomposition forecast. Single LLM call with structured decomposition prompt."""
    logger.info("decomposed_forecast_start", question_id=question.id, model=MODEL)
    _ensure_vertex_credentials()

    prompt = _build_decomposition_prompt(
        question,
        resolution_date=resolution_date,
        source=source,
        resolution_dates=resolution_dates,
    )

    try:
        response = await litellm.acompletion(
            model=MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
            timeout=60,
        )
    except Exception:
        logger.error("decomposed_forecast_api_error", question_id=question.id, model=MODEL, exc_info=True)
        raise

    text = response.choices[0].message.content or ""
    prob = _parse_probability(text)
    logger.info("decomposed_forecast_complete", question_id=question.id, probability=prob)
    return prob


async def decomposed_forecast_multi_horizon(
    question: Question,
    resolution_dates: list[str],
    source: str | None = None,
    prompt_variant: str = "dataset",
) -> list[float] | None:
    """Multi-horizon decomposition forecast. Single LLM call, extracts per-date probabilities."""
    n_horizons = len(resolution_dates)
    logger.info(
        "decomposed_multi_horizon_start",
        question_id=question.id,
        n_horizons=n_horizons,
        model=MODEL,
    )
    _ensure_vertex_credentials()

    prompt = _build_decomposition_prompt(
        question,
        source=source,
        resolution_dates=resolution_dates,
    )

    try:
        response = await litellm.acompletion(
            model=MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
            timeout=120,
        )
    except Exception:
        logger.error(
            "decomposed_multi_horizon_api_error",
            question_id=question.id,
            model=MODEL,
            exc_info=True,
        )
        return None

    text = response.choices[0].message.content or ""

    probs = _extract_probabilities(text, n_horizons)
    if probs is not None:
        logger.info(
            "decomposed_multi_horizon_complete",
            question_id=question.id,
            n_horizons=n_horizons,
            method="regex",
        )
        _save_response_log(question.id, text, "decompose_regex_success", n_horizons)
        return probs

    logger.info("decomposed_multi_horizon_regex_failed", question_id=question.id, trying="llm_extraction")
    probs = await _extract_with_llm(text, n_horizons)
    if probs is not None:
        logger.info(
            "decomposed_multi_horizon_complete",
            question_id=question.id,
            n_horizons=n_horizons,
            method="llm_extraction",
        )
        _save_response_log(question.id, text, "decompose_llm_success", n_horizons)
        return probs

    logger.warning(
        "decomposed_multi_horizon_fallback",
        question_id=question.id,
        n_horizons=n_horizons,
    )
    _save_response_log(question.id, text, "decompose_fallback", n_horizons)
    return None
