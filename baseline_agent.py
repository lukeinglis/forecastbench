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


def _clamp(v: float) -> float:
    return max(0.01, min(0.99, v))


def _extract_answer_block(text: str) -> str | None:
    """Extract text after an 'Answer:' marker, or fall back to the last paragraph."""
    match = re.search(r"(?i)answer\s*:\s*", text)
    if match:
        return text[match.end():]
    paragraphs = text.strip().split("\n\n")
    if len(paragraphs) > 1:
        return paragraphs[-1]
    return None


_ASTERISK_RE = re.compile(r"\*\s*(0?\.\d+|1\.0{0,}|0(?:\.0{0,})?)\s*\*")
_DECIMAL_RE = re.compile(r"(?<!\d)(0?\.\d+|1\.0{0,}|0(?:\.0{0,})?)(?!\d)")
_TOKEN_RE = re.compile(r"(?:\*)?(\d*\.?\d+)(?:\*)?")


def _parse_probs_from_text(text: str, n_expected: int) -> list[float] | None:
    """Extract probabilities from a focused text block (e.g. Answer section)."""
    asterisks = _ASTERISK_RE.findall(text)
    if len(asterisks) == n_expected:
        return [_clamp(float(m)) for m in asterisks]
    decimals = _DECIMAL_RE.findall(text)
    valid = [float(d) for d in decimals if 0 <= float(d) <= 1]
    if len(valid) == n_expected:
        return [_clamp(v) for v in valid]
    return None


def _tokenize_and_extract(text: str, n_expected: int) -> list[float] | None:
    """Upstream approach: split into tokens, fullmatch each for a probability."""
    probabilities: list[float] = []
    for token in text.strip().replace(",", " ").replace("{", " ").replace("}", " ").split():
        m = _TOKEN_RE.fullmatch(token.strip())
        if m is None:
            continue
        val = float(m.group(1))
        if 0 <= val <= 1:
            probabilities.append(val)
    if len(probabilities) == n_expected:
        return [_clamp(p) for p in probabilities]
    if len(probabilities) > n_expected:
        return [_clamp(p) for p in probabilities[-n_expected:]]
    return None


def _asterisk_extract(text: str, n_expected: int) -> list[float] | None:
    """Find asterisk-wrapped probabilities in the full text."""
    matches = _ASTERISK_RE.findall(text)
    if len(matches) == n_expected:
        return [_clamp(float(m)) for m in matches]
    if len(matches) > n_expected:
        return [_clamp(float(m)) for m in matches[-n_expected:]]
    return None


def _decimal_extract(text: str, n_expected: int) -> list[float] | None:
    """Find any decimal probabilities in the full text."""
    all_decimals = _DECIMAL_RE.findall(text)
    valid = [float(d) for d in all_decimals if 0 <= float(d) <= 1]
    if len(valid) == n_expected:
        return [_clamp(v) for v in valid]
    if len(valid) > n_expected:
        return [_clamp(v) for v in valid[-n_expected:]]
    return None


def _extract_probabilities(text: str, n_expected: int) -> list[float] | None:
    """Multi-strategy extraction of probabilities from model response text.

    Strategies tried in order:
    1. Answer-block extraction (text after 'Answer:' or last paragraph)
    2. Upstream tokenize-and-fullmatch on full text
    3. Asterisk regex on full text with take-last-N
    4. Decimal regex on full text with take-last-N
    """
    answer_block = _extract_answer_block(text)
    if answer_block:
        probs = _parse_probs_from_text(answer_block, n_expected)
        if probs:
            logger.debug("extract_probabilities", method="answer_block", n=len(probs))
            return probs

    probs = _tokenize_and_extract(text, n_expected)
    if probs:
        logger.debug("extract_probabilities", method="tokenize", n=len(probs))
        return probs

    probs = _asterisk_extract(text, n_expected)
    if probs:
        logger.debug("extract_probabilities", method="asterisk", n=len(probs))
        return probs

    probs = _decimal_extract(text, n_expected)
    if probs:
        logger.debug("extract_probabilities", method="decimal", n=len(probs))
        return probs

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
        logger.debug("extraction_llm_response", response_text=result_text[:200])
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
) -> list[float] | None:
    """Forecast multiple horizons in a single LLM call.

    Returns a list of probabilities on success, or None on fallback so the
    caller knows not to cache placeholder values.
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
        return None

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
    return None


if __name__ == "__main__":
    import asyncio
    from eval import run_eval
    asyncio.run(run_eval(aforecast))
