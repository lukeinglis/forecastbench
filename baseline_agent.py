"""Baseline LLM forecaster using litellm for probability estimation."""

from __future__ import annotations

import ast
import os
import re
from typing import Any

import litellm

from fetch_data import MARKET_SOURCES, Question
from logging_config import get_logger

logger = get_logger("baseline_agent")

# Pinned to specific snapshot for benchmark reproducibility. Override via FORECAST_MODEL env var.
MODEL = os.getenv("FORECAST_MODEL", "claude-sonnet-4-20250514")

EXTRACTION_MODEL = os.getenv("FORECAST_EXTRACTION_MODEL", MODEL)

FORECAST_EXTRACTION_PROMPT = """You are extracting probabilities from text.

Your task is to extract probabilistic forecasts from the MODEL RESPONSE only.

Expected number of probabilities: {n_horizons}

Rules:
- Do not make a forecast.
- Do not infer missing values.
- Do not revise, smooth, calibrate, average, or replace any probabilities.
- Extract only probabilities explicitly stated in the MODEL RESPONSE.
- Preserve the order in which the probabilities appear in the MODEL RESPONSE.
- If the MODEL RESPONSE contains final-answer probabilities paired with resolution dates, return them in the same order as those dates appear in the MODEL RESPONSE.
- If the MODEL RESPONSE contains multiple candidate sets of probabilities, use only the final answer set.
- If you cannot identify exactly {n_horizons} final-answer probabilities, return [].

Return only a Python list of decimal probabilities, e.g. [0.1, 0.2, 0.3].
Do not output explanation or any other text.

MODEL RESPONSE:
```text
{model_response}
```"""

# -- Prompt templates matching upstream ForecastBench (Halawi et al. 2024) --
# From: src/llm_forecaster/prompts.py

ZERO_SHOT_MARKET_PROMPT = """\
You are an expert superforecaster, familiar with the work of Tetlock and others. \
Make a prediction of the probability that the question will be resolved as true. \
You MUST give a probability estimate between 0 and 1 UNDER ALL CIRCUMSTANCES. \
If for some reason you can't answer, pick the base rate, but return a number between 0 and 1.

Question:
{question}

Question Background:
{background}

Resolution Criteria:
{resolution_criteria}

Today's Date: {today_date}

Question resolution date: {resolution_date}

Output your answer (a number between 0 and 1) with an asterisk at the beginning and end of the decimal.
Do not output anything else.
Answer: {{ Insert answer here }}"""

ZERO_SHOT_MARKET_WITH_FREEZE_VALUE_PROMPT = """\
You are an expert superforecaster, familiar with the work of Tetlock and others. \
Make a prediction of the probability that the question will be resolved as true. \
You MUST give a probability estimate between 0 and 1 UNDER ALL CIRCUMSTANCES. \
If for some reason you can't answer, pick the base rate, but return a number between 0 and 1.

Question:
{question}

Question Background:
{background}

Resolution Criteria:
{resolution_criteria}

Market value on {freeze_datetime}:
{freeze_datetime_value}

Today's Date: {today_date}

Question resolution date: {resolution_date}

Output your answer (a number between 0 and 1) with an asterisk at the beginning and end of the decimal.
Do not output anything else.
Answer: {{ Insert answer here }}"""

ZERO_SHOT_DATASET_PROMPT = """\
You are an expert superforecaster, familiar with the work of Tetlock and others. \
Make a prediction of the probability that the question will be resolved as true. \
You MUST give a probability estimate between 0 and 1 UNDER ALL CIRCUMSTANCES. \
If for some reason you can't answer, pick the base rate, but return a number between 0 and 1.

You're going to predict the probability of the following potential outcome "at each of the resolution dates".

Question:
{question}

Question Background:
{background}

Resolution Criteria:
{resolution_criteria}

Current value on {freeze_datetime}:
{freeze_datetime_value}

Value Explanation:
{freeze_datetime_value_explanation}

Today's Date: {today_date}

Question resolution dates: {list_of_resolution_dates}

Output your answer (a number between 0 and 1) with an asterisk at the beginning and end of the decimal. \
(For example, if there are n resolution dates, you would output different *p* for each resolution date) \
Do not output anything else.
Answer: {{ Insert answer here }}"""


def _build_prompt(
    question: Question,
    resolution_date: str | None = None,
    source: str | None = None,
    resolution_dates: Any = None,
    prompt_variant: str = "zero-shot",
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

    effective_resolution_date = resolution_date
    if not effective_resolution_date and is_market:
        effective_resolution_date = getattr(question, "market_info_close_datetime", None)

    fv = getattr(question, "freeze_datetime_value", None)
    fd = question.freeze_datetime or ""

    if prompt_variant == "zero-shot-fv" and is_market and fv is not None and fd:
        return ZERO_SHOT_MARKET_WITH_FREEZE_VALUE_PROMPT.format(
            question=question.question,
            background=background,
            resolution_criteria=question.resolution_criteria or "",
            freeze_datetime=fd,
            freeze_datetime_value=fv,
            today_date=today_date,
            resolution_date=effective_resolution_date or "",
        )

    if prompt_variant == "dataset" and not is_market:
        effective_rd = resolution_dates or getattr(question, "resolution_dates", None)
        dates_str = ""
        if effective_rd and isinstance(effective_rd, list):
            valid = [d for d in effective_rd if d and str(d).upper() != "N/A"]
            dates_str = ", ".join(str(d) for d in valid)

        return ZERO_SHOT_DATASET_PROMPT.format(
            question=question.question,
            background=background,
            resolution_criteria=question.resolution_criteria or "",
            freeze_datetime=fd,
            freeze_datetime_value=fv if fv is not None else "",
            freeze_datetime_value_explanation=getattr(question, "freeze_datetime_value_explanation", None) or "",
            today_date=today_date,
            list_of_resolution_dates=dates_str,
        )

    return ZERO_SHOT_MARKET_PROMPT.format(
        question=question.question,
        background=background,
        resolution_criteria=question.resolution_criteria or "",
        today_date=today_date,
        resolution_date=effective_resolution_date or "",
    )


def _parse_probability(text: str) -> float:
    asterisk = re.search(r"\*\s*(0?\.\d+|1\.0{0,}|0(?:\.0{0,})?)\s*\*", text)
    if asterisk:
        prob = float(asterisk.group(1))
        clamped = max(0.01, min(0.99, prob))
        logger.debug("parsed_probability", raw_match=asterisk.group(1), parsed=clamped, format="asterisk")
        return clamped
    match = re.search(r"[Pp]robability[\s:=]+\s*(0?\.\d+|1\.0{0,}|0(?:\.0{0,})?)", text)
    if not match:
        match = re.search(r"(?:^|\s|:)\s*(0?\.\d+|1\.0{0,}|0(?:\.0{0,})?)\s*$", text, re.MULTILINE)
    if not match:
        match = re.search(r"(0?\.\d+|1\.0{0,})", text)
    if match:
        prob = float(match.group(1))
        clamped = max(0.01, min(0.99, prob))
        logger.debug("parsed_probability", raw_match=match.group(1), parsed=clamped, format="standard")
        return clamped
    logger.debug("parsed_probability", raw_match=None, parsed=0.5, fallback=True)
    return 0.5


def forecast(
    question: Question,
    resolution_date: str | None = None,
    source: str | None = None,
    resolution_dates: Any = None,
    prompt_variant: str = "zero-shot",
) -> float:
    logger.info("forecast_start", question_id=question.id, model=MODEL)
    prompt = _build_prompt(
        question,
        resolution_date=resolution_date,
        source=source,
        resolution_dates=resolution_dates,
        prompt_variant=prompt_variant,
    )
    try:
        response = litellm.completion(
            model=MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            timeout=60,
        )
    except Exception:
        logger.error("forecast_api_error", question_id=question.id, model=MODEL, exc_info=True)
        raise
    text = response.choices[0].message.content or ""
    prob = _parse_probability(text)
    logger.info("forecast_complete", question_id=question.id, probability=prob)
    return prob


async def aforecast(
    question: Question,
    resolution_date: str | None = None,
    source: str | None = None,
    resolution_dates: Any = None,
    prompt_variant: str = "zero-shot",
) -> float:
    logger.info("forecast_start", question_id=question.id, model=MODEL, async_mode=True)
    prompt = _build_prompt(
        question,
        resolution_date=resolution_date,
        source=source,
        resolution_dates=resolution_dates,
        prompt_variant=prompt_variant,
    )
    try:
        response = await litellm.acompletion(
            model=MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            timeout=60,
        )
    except Exception:
        logger.error("forecast_api_error", question_id=question.id, model=MODEL, exc_info=True)
        raise
    text = response.choices[0].message.content or ""
    prob = _parse_probability(text)
    logger.info("forecast_complete", question_id=question.id, probability=prob)
    return prob


def _extract_probabilities(text: str, n_expected: int) -> list[float] | None:
    """Try to extract exactly n_expected probabilities from text using regex.

    When the model repeats probabilities in reasoning and a final answer block,
    takes the LAST n_expected matches (the final answer set).
    """
    asterisk_matches = re.findall(r"\*\s*(0?\.\d+|1\.0{0,}|0(?:\.0{0,})?)\s*\*", text)
    if len(asterisk_matches) == n_expected:
        return [max(0.01, min(0.99, float(m))) for m in asterisk_matches]
    if len(asterisk_matches) > n_expected and len(asterisk_matches) % n_expected == 0:
        last_n = asterisk_matches[-n_expected:]
        return [max(0.01, min(0.99, float(m))) for m in last_n]

    all_decimals = re.findall(r"(?<!\d)(0?\.\d+|1\.0{0,}|0(?:\.0{0,})?)(?!\d)", text)
    valid = [float(d) for d in all_decimals if 0 <= float(d) <= 1]
    if len(valid) == n_expected:
        return [max(0.01, min(0.99, v)) for v in valid]
    if len(valid) > n_expected:
        last_n = valid[-n_expected:]
        return [max(0.01, min(0.99, v)) for v in last_n]

    return None


async def _extract_with_llm(text: str, n_expected: int) -> list[float] | None:
    """Use a cheap LLM to extract probabilities when regex fails."""
    prompt = FORECAST_EXTRACTION_PROMPT.format(
        n_horizons=n_expected,
        model_response=text,
    )
    try:
        response = await litellm.acompletion(
            model=EXTRACTION_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            timeout=30,
        )
        result_text = response.choices[0].message.content or ""
        parsed = ast.literal_eval(result_text.strip())
        if isinstance(parsed, list) and len(parsed) == n_expected:
            if all(isinstance(v, (int, float)) and 0 <= v <= 1 for v in parsed):
                return [max(0.01, min(0.99, float(v))) for v in parsed]
        logger.warning(
            "extraction_llm_invalid",
            n_expected=n_expected,
            n_got=len(parsed) if isinstance(parsed, list) else 0,
        )
        return None
    except Exception:
        logger.warning("extraction_llm_error", model=EXTRACTION_MODEL, exc_info=True)
        return None


async def aforecast_multi_horizon(
    question: Question,
    resolution_dates: list[str],
    source: str | None = None,
    prompt_variant: str = "dataset",
    forecast_due_date: str | None = None,
) -> list[float]:
    """Forecast multiple horizons in a single LLM call.

    Returns a list of probabilities, one per resolution date.
    """
    n_horizons = len(resolution_dates)
    logger.info(
        "multi_horizon_start",
        question_id=question.id,
        n_horizons=n_horizons,
        model=MODEL,
    )

    prompt = _build_prompt(
        question,
        source=source,
        resolution_dates=resolution_dates,
        prompt_variant=prompt_variant,
    )

    try:
        response = await litellm.acompletion(
            model=MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            timeout=120,
        )
    except Exception:
        logger.error(
            "multi_horizon_api_error",
            question_id=question.id,
            model=MODEL,
            exc_info=True,
        )
        return [0.5] * n_horizons

    text = response.choices[0].message.content or ""

    probs = _extract_probabilities(text, n_horizons)
    if probs is not None:
        logger.info(
            "multi_horizon_complete",
            question_id=question.id,
            n_horizons=n_horizons,
            method="regex",
        )
        return probs

    logger.info("multi_horizon_regex_failed", question_id=question.id, trying="llm_extraction")
    probs = await _extract_with_llm(text, n_horizons)
    if probs is not None:
        logger.info(
            "multi_horizon_complete",
            question_id=question.id,
            n_horizons=n_horizons,
            method="llm_extraction",
        )
        return probs

    logger.warning(
        "multi_horizon_fallback",
        question_id=question.id,
        n_horizons=n_horizons,
    )
    return [0.5] * n_horizons


if __name__ == "__main__":
    import asyncio
    from eval import run_eval
    asyncio.run(run_eval(aforecast))
