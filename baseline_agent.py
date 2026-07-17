"""Baseline LLM forecaster using litellm for probability estimation."""

from __future__ import annotations

import os
import re
from typing import Any

import litellm

from cutoff import CutoffEnvironment
from fetch_data import MARKET_SOURCES, Question
from logging_config import get_logger

logger = get_logger("baseline_agent")

# Pinned to specific snapshot for benchmark reproducibility. Override via FORECAST_MODEL env var.
MODEL = os.getenv("FORECAST_MODEL", "claude-sonnet-4-20250514")

# -- Prompt variants (attributed to Halawi et al. 2024, ForecastBench) --

ZERO_SHOT_PROMPT = """You are an expert superforecaster with a track record of well-calibrated probabilistic predictions. Your goal is to estimate the probability that the following question resolves to YES.

{temporal_context}

{source_context}{resolution_date_section}Question: {question}

{background_section}{criteria_section}

Think step-by-step:
1. Identify the base rate for this type of event
2. Consider relevant factors that adjust the probability up or down
3. Check for common biases (anchoring, availability, representativeness)
4. Provide your final calibrated probability

Your final answer MUST be a single probability between 0.01 and 0.99 on its own line, formatted as:

Probability: <number>"""

ZERO_SHOT_FV_MARKET_PROMPT = """You are an expert superforecaster with a track record of well-calibrated probabilistic predictions.

{temporal_context}

Question: {question}

{background_section}{criteria_section}{freeze_value_section}{resolution_date_section}

Think step-by-step:
1. Identify the base rate for this type of event
2. Consider relevant factors that adjust the probability up or down
3. Check for common biases (anchoring, availability, representativeness)
4. Provide your final calibrated probability

Output your answer (a number between 0 and 1) with an asterisk at the beginning and end of the decimal. Do not use any other formatting."""

DATASET_PROMPT = """You are an expert superforecaster with a track record of well-calibrated probabilistic predictions.

{temporal_context}

{source_context}Question: {question}

{background_section}{criteria_section}{freeze_value_section}{resolution_dates_section}

Think step-by-step:
1. Consider the historical data and trend indicated by the freeze value
2. Identify the base rate for this type of event
3. Consider relevant factors that adjust the probability up or down
4. For each resolution date, assess how the probability changes over time

Output your answer (a number between 0 and 1) with an asterisk at the beginning and end of the decimal. Do not use any other formatting."""

PROMPT_TEMPLATE = ZERO_SHOT_PROMPT


def _build_prompt(
    question: Question,
    resolution_date: str | None = None,
    source: str | None = None,
    resolution_dates: Any = None,
    prompt_variant: str = "zero-shot",
) -> str:
    if question.freeze_datetime:
        env = CutoffEnvironment(question.freeze_datetime)
        prepared = env.prepare_question(question)
        temporal_context = env.frame_temporal_context(question)
    else:
        prepared = question
        temporal_context = ""

    effective_source = source or question.source
    is_market = effective_source.lower() in MARKET_SOURCES

    source_context = (
        f"Source Context: {prepared.source_intro}\n\n"
        if getattr(prepared, "source_intro", None)
        else ""
    )

    effective_resolution_date = resolution_date
    if not effective_resolution_date and is_market:
        effective_resolution_date = getattr(prepared, "market_info_close_datetime", None)
    resolution_date_section = (
        f"Target resolution date: {effective_resolution_date}\n\n"
        if effective_resolution_date
        else ""
    )

    background_section = f"Background: {prepared.background}\n" if prepared.background else ""
    mrc = getattr(prepared, "market_info_resolution_criteria", None)
    if mrc and mrc != "N/A":
        background_section = background_section + "\n" + mrc
    criteria_section = (
        f"Resolution Criteria: {prepared.resolution_criteria}\n"
        if prepared.resolution_criteria
        else ""
    )

    freeze_value_section = ""
    if prompt_variant == "zero-shot-fv" or prompt_variant == "dataset":
        fv = getattr(prepared, "freeze_datetime_value", None)
        fd = getattr(prepared, "freeze_datetime", None)
        if fv is not None and fd is not None:
            freeze_value_section = f"Market value on {fd}:\n{fv}\n"
            fv_expl = getattr(prepared, "freeze_datetime_value_explanation", None)
            if fv_expl:
                freeze_value_section += f"Explanation: {fv_expl}\n"

    resolution_dates_section = ""
    effective_rd = resolution_dates or getattr(prepared, "resolution_dates", None)
    if prompt_variant == "dataset" and effective_rd and isinstance(effective_rd, list):
        valid_dates = [d for d in effective_rd if d and str(d).upper() != "N/A"]
        if valid_dates:
            resolution_dates_section = "Resolution dates: " + ", ".join(str(d) for d in valid_dates) + "\n"

    if prompt_variant == "zero-shot-fv" and is_market:
        template = ZERO_SHOT_FV_MARKET_PROMPT
    elif prompt_variant == "dataset" and not is_market:
        template = DATASET_PROMPT
    else:
        template = ZERO_SHOT_PROMPT

    return template.format(
        temporal_context=temporal_context,
        question=prepared.question,
        background_section=background_section,
        criteria_section=criteria_section,
        source_context=source_context,
        resolution_date_section=resolution_date_section,
        freeze_value_section=freeze_value_section,
        resolution_dates_section=resolution_dates_section,
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


if __name__ == "__main__":
    import asyncio
    from eval import run_eval
    asyncio.run(run_eval(aforecast))
