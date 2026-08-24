"""Lab forecaster using litellm for probability estimation."""

from __future__ import annotations

import ast
import os
import re
import threading
import time
from typing import Any

import litellm

from fetch_data import MARKET_SOURCES, Question
from logging_config import get_logger

logger = get_logger("lab_forecaster")

litellm.vertex_project = os.getenv("VERTEX_PROJECT", "itpc-gcp-product-all-claude")
litellm.vertex_location = os.getenv("VERTEX_LOCATION", "europe-west1")

MODEL = os.getenv("FORECAST_MODEL", "vertex_ai/claude-sonnet-4@20250514")
EXTRACTION_MODEL = os.getenv("FORECAST_EXTRACTION_MODEL", "openai/gpt-4o-mini")
TEMPERATURE = float(os.getenv("FORECAST_TEMPERATURE", "0"))
MAX_TOKENS = int(os.getenv("FORECAST_MAX_TOKENS", "16384"))
VERTEX_LOCATION = os.getenv("VERTEXAI_LOCATION", "europe-west1")
THINKING_BUDGET = int(os.getenv("FORECAST_THINKING_BUDGET", "16000"))

_REFRESH_MARGIN_SECS = 300
_vertex_creds_lock = threading.Lock()
_vertex_credentials: Any = None
_vertex_token_expiry: float = 0.0

_cost_tracker: dict[str, float] = {}


def get_tracked_costs() -> dict[str, float]:
    return dict(_cost_tracker)


def clear_tracked_costs() -> None:
    _cost_tracker.clear()


def _track_cost(question_id: str, response: Any) -> None:
    try:
        cost = response._hidden_params.get("response_cost")
        if cost is not None:
            _cost_tracker[question_id] = float(cost)
    except (AttributeError, TypeError, ValueError):
        pass


def _get_google_auth() -> tuple[Any, Any]:
    import google.auth
    import google.auth.transport.requests
    return google.auth, google.auth.transport.requests


def _ensure_vertex_credentials(model: str | None = None) -> None:
    effective = model or MODEL
    if not effective.startswith("vertex_ai/"):
        return

    global _vertex_credentials, _vertex_token_expiry

    if time.monotonic() < _vertex_token_expiry:
        return

    with _vertex_creds_lock:
        if time.monotonic() < _vertex_token_expiry:
            return
        try:
            auth_mod, transport_mod = _get_google_auth()
            if _vertex_credentials is None:
                _vertex_credentials, _ = auth_mod.default()
            _vertex_credentials.refresh(transport_mod.Request())
            if hasattr(_vertex_credentials, "expiry") and _vertex_credentials.expiry:
                remaining = (_vertex_credentials.expiry.timestamp() - time.time())
                _vertex_token_expiry = time.monotonic() + max(0, remaining - _REFRESH_MARGIN_SECS)
            else:
                _vertex_token_expiry = time.monotonic() + 1800
            logger.debug("vertex_credentials_refreshed")
        except Exception:
            logger.warning("vertex_credentials_refresh_failed", exc_info=True)


# -- Prompt templates matching upstream ForecastBench (Halawi et al. 2024) --

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


def _format_question_text(text: str, forecast_due_date: str, is_dataset: bool) -> str:
    if not is_dataset:
        return text
    try:
        return text.format(
            forecast_due_date=forecast_due_date,
            resolution_date="each of the resolution dates provided below",
        )
    except (KeyError, IndexError, ValueError):
        return text


def _is_reasoning_model(model: str) -> bool:
    model_lower = model.lower()
    if "claude" in model_lower and THINKING_BUDGET > 0:
        return True
    if any(x in model_lower for x in ["o1", "o3", "luna"]):
        return True
    return False


def _forecast_kwargs(
    messages: list[dict[str, str]],
    timeout: int = 180,
    model: str | None = None,
) -> dict[str, Any]:
    effective_model = model or MODEL

    kwargs: dict[str, Any] = {
        "model": effective_model,
        "messages": messages,
        "max_tokens": MAX_TOKENS,
        "timeout": timeout,
        "vertex_location": VERTEX_LOCATION,
    }

    if _is_reasoning_model(effective_model):
        if "claude" in effective_model.lower():
            kwargs["thinking"] = {"type": "enabled", "budget_tokens": THINKING_BUDGET}
    else:
        kwargs["temperature"] = TEMPERATURE

    return kwargs


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

    if is_market:
        if fv is not None and fd:
            return ZERO_SHOT_MARKET_WITH_FREEZE_VALUE_PROMPT.format(
                question=question.question,
                background=background,
                resolution_criteria=question.resolution_criteria or "",
                freeze_datetime=fd,
                freeze_datetime_value=fv,
                today_date=today_date,
                resolution_date=effective_resolution_date or "",
            )
        return ZERO_SHOT_MARKET_PROMPT.format(
            question=question.question,
            background=background,
            resolution_criteria=question.resolution_criteria or "",
            today_date=today_date,
            resolution_date=effective_resolution_date or "",
        )

    effective_rd = resolution_dates or getattr(question, "resolution_dates", None)
    dates_list: list[str] = []
    if effective_rd and isinstance(effective_rd, list):
        dates_list = [str(d) for d in effective_rd if d and str(d).upper() != "N/A"]

    formatted_q = _format_question_text(question.question, today_date, is_dataset=True)

    return ZERO_SHOT_DATASET_PROMPT.format(
        question=formatted_q,
        background=background,
        resolution_criteria=question.resolution_criteria or "",
        freeze_datetime=fd,
        freeze_datetime_value=fv if fv is not None else "",
        freeze_datetime_value_explanation=getattr(question, "freeze_datetime_value_explanation", None) or "",
        today_date=today_date,
        list_of_resolution_dates=dates_list,
    )



_FULLMATCH_RE = re.compile(r"\*?\s*(0?\.\d+|1\.0{0,}|0(?:\.0{0,})?)\s*\*?")


def _parse_probability(text: str) -> float:
    fm = _FULLMATCH_RE.fullmatch(text.strip())
    if fm:
        return float(fm.group(1))
    asterisk = re.search(r"\*\s*(0?\.\d+|1\.0{0,}|0(?:\.0{0,})?)\s*\*", text)
    if asterisk:
        return float(asterisk.group(1))
    match = re.search(r"[Pp]robability[\s:=]+\s*(0?\.\d+|1\.0{0,}|0(?:\.0{0,})?)", text)
    if not match:
        match = re.search(r"(?:^|\s|:)\s*(0?\.\d+|1\.0{0,}|0(?:\.0{0,})?)\s*$", text, re.MULTILINE)
    if not match:
        match = re.search(r"(0?\.\d+|1\.0{0,})", text)
    if match:
        return float(match.group(1))
    raise ValueError(f"Could not parse probability from response: {text[:100]}")


_ASTERISK_RE = re.compile(r"\*\s*(0?\.\d+|1\.0{0,}|0(?:\.0{0,})?)\s*\*")
_DECIMAL_RE = re.compile(r"(?<!\d)(0?\.\d+|1\.0{0,}|0(?:\.0{0,})?)(?!\d)")
_TOKEN_RE = re.compile(r"(?:\*)?(\d*\.?\d+)(?:\*)?")


def _extract_answer_block(text: str) -> str | None:
    match = re.search(r"(?i)answer\s*:\s*", text)
    if match:
        return text[match.end():]
    paragraphs = text.strip().split("\n\n")
    if len(paragraphs) > 1:
        return paragraphs[-1]
    return None


def _parse_probs_from_text(text: str, n_expected: int) -> list[float] | None:
    asterisks = _ASTERISK_RE.findall(text)
    if len(asterisks) == n_expected:
        return [float(m) for m in asterisks]
    decimals = _DECIMAL_RE.findall(text)
    valid = [float(d) for d in decimals if 0 <= float(d) <= 1]
    if len(valid) == n_expected:
        return [float(v) for v in valid]
    return None


def _tokenize_and_extract(text: str, n_expected: int) -> list[float] | None:
    probabilities: list[float] = []
    for token in text.strip().replace(",", " ").replace("{", " ").replace("}", " ").split():
        m = _TOKEN_RE.fullmatch(token.strip())
        if m is None:
            continue
        val = float(m.group(1))
        if 0 <= val <= 1:
            probabilities.append(val)
    if len(probabilities) == n_expected:
        return [float(p) for p in probabilities]
    if len(probabilities) > n_expected:
        return [float(p) for p in probabilities[-n_expected:]]
    return None


def _asterisk_extract(text: str, n_expected: int) -> list[float] | None:
    matches = _ASTERISK_RE.findall(text)
    if len(matches) == n_expected:
        return [float(m) for m in matches]
    if len(matches) > n_expected:
        return [float(m) for m in matches[-n_expected:]]
    return None


def _decimal_extract(text: str, n_expected: int) -> list[float] | None:
    all_decimals = _DECIMAL_RE.findall(text)
    valid = [float(d) for d in all_decimals if 0 <= float(d) <= 1]
    if len(valid) == n_expected:
        return [float(v) for v in valid]
    if len(valid) > n_expected:
        return [float(v) for v in valid[-n_expected:]]
    return None


def _extract_probabilities(text: str, n_expected: int) -> list[float] | None:
    answer_block = _extract_answer_block(text)
    if answer_block:
        probs = _parse_probs_from_text(answer_block, n_expected)
        if probs:
            return probs

    probs = _tokenize_and_extract(text, n_expected)
    if probs:
        return probs

    probs = _asterisk_extract(text, n_expected)
    if probs:
        return probs

    probs = _decimal_extract(text, n_expected)
    if probs:
        return probs

    return None


async def _extract_with_llm(text: str, n_expected: int) -> list[float] | None:
    prompt = FORECAST_EXTRACTION_PROMPT.format(n_horizons=n_expected, model_response=text)
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
                return [float(v) for v in parsed]
        return None
    except Exception:
        logger.warning("extraction_llm_error", model=EXTRACTION_MODEL, exc_info=True)
        return None


def forecast(
    question: Question,
    resolution_date: str | None = None,
    source: str | None = None,
    resolution_dates: Any = None,
    prompt_variant: str = "zero-shot",
) -> float:
    logger.info("forecast_start", question_id=question.id, model=MODEL)
    _ensure_vertex_credentials()
    prompt = _build_prompt(question, resolution_date=resolution_date, source=source, resolution_dates=resolution_dates)
    messages = [{"role": "user", "content": prompt}]
    kwargs = _forecast_kwargs(messages)
    response = litellm.completion(**kwargs)
    _track_cost(question.id, response)
    text = response.choices[0].message.content or ""
    prob = _parse_probability(text)
    logger.info("forecast_complete", question_id=question.id, forecast_value=prob)
    return prob


async def aforecast(
    question: Question,
    resolution_date: str | None = None,
    source: str | None = None,
    resolution_dates: Any = None,
    prompt_variant: str = "zero-shot",
    model_override: str | None = None,
) -> float:
    model = model_override or MODEL
    logger.info("forecast_start", question_id=question.id, model=model, async_mode=True)
    _ensure_vertex_credentials(model)
    prompt = _build_prompt(question, resolution_date=resolution_date, source=source, resolution_dates=resolution_dates)
    messages = [{"role": "user", "content": prompt}]
    kwargs = _forecast_kwargs(messages, model=model)
    response = await litellm.acompletion(**kwargs)
    _track_cost(question.id, response)
    text = response.choices[0].message.content or ""
    prob = _parse_probability(text)
    logger.info("forecast_complete", question_id=question.id, probability=prob)
    return prob


def forecast_multi(
    question: Question,
    resolution_dates: list[str],
) -> list[float]:
    logger.info("forecast_multi_start", question_id=question.id, model=MODEL)
    _ensure_vertex_credentials()
    prompt = _build_prompt(question, resolution_dates=resolution_dates)
    messages = [{"role": "user", "content": prompt}]
    kwargs = _forecast_kwargs(messages)
    response = litellm.completion(**kwargs)
    text = response.choices[0].message.content or ""
    probs = _extract_probabilities(text, len(resolution_dates))
    if probs is not None:
        return probs
    raise ValueError(f"Could not extract {len(resolution_dates)} probabilities from response")


async def aforecast_multi_horizon(
    question: Question,
    resolution_dates: list[str],
    source: str | None = None,
    prompt_variant: str = "dataset",
    forecast_due_date: str | None = None,
    model_override: str | None = None,
) -> list[float] | None:
    n_horizons = len(resolution_dates)
    model = model_override or MODEL
    logger.info("multi_horizon_start", question_id=question.id, n_horizons=n_horizons, model=model)
    _ensure_vertex_credentials(model)
    prompt = _build_prompt(question, source=source, resolution_dates=resolution_dates)
    messages = [{"role": "user", "content": prompt}]
    kwargs = _forecast_kwargs(messages, model=model)
    try:
        response = await litellm.acompletion(**kwargs)
    except Exception:
        logger.error("multi_horizon_api_error", question_id=question.id, exc_info=True)
        return None
    _track_cost(question.id, response)
    text = response.choices[0].message.content or ""

    probs = _extract_probabilities(text, n_horizons)
    if probs is not None:
        return probs

    probs = await _extract_with_llm(text, n_horizons)
    if probs is not None:
        return probs

    logger.warning("multi_horizon_fallback", question_id=question.id, n_horizons=n_horizons)
    return None


async def aforecast_multi(
    question: Question,
    resolution_dates: list[str],
) -> list[float]:
    logger.info("aforecast_multi_start", question_id=question.id, model=MODEL)
    _ensure_vertex_credentials()
    prompt = _build_prompt(question, resolution_dates=resolution_dates)
    messages = [{"role": "user", "content": prompt}]
    kwargs = _forecast_kwargs(messages)
    response = await litellm.acompletion(**kwargs)
    text = response.choices[0].message.content or ""
    probs = _extract_probabilities(text, len(resolution_dates))
    if probs is not None:
        return probs
    raise ValueError(f"Could not extract {len(resolution_dates)} probabilities from response")


def multi_forecast(
    question: Question,
    resolution_dates: list[str],
    source: str | None = None,
    prompt_variant: str = "zero-shot",
) -> list[float]:
    n = len(resolution_dates)
    logger.info("multi_forecast_start", question_id=question.id, n_horizons=n, model=MODEL)
    _ensure_vertex_credentials()
    prompt = _build_prompt(question, source=source, resolution_dates=resolution_dates)
    messages = [{"role": "user", "content": prompt}]
    kwargs = _forecast_kwargs(messages)
    response = litellm.completion(**kwargs)
    _track_cost(question.id, response)
    text = response.choices[0].message.content or ""

    probs = _extract_probabilities(text, n)
    if probs is not None:
        return probs

    logger.warning("multi_forecast_regex_failed", question_id=question.id, n_horizons=n)
    fallback: list[float] = []
    for date in resolution_dates:
        try:
            p = forecast(question, resolution_date=date, source=source, prompt_variant=prompt_variant)
        except Exception:
            logger.warning("multi_forecast_single_fallback_error", question_id=question.id, date=date)
            p = 0.5
        fallback.append(p)
    return fallback


async def amulti_forecast(
    question: Question,
    resolution_dates: list[str],
    source: str | None = None,
    prompt_variant: str = "zero-shot",
    model_override: str | None = None,
) -> list[float]:
    n = len(resolution_dates)
    probs = await aforecast_multi_horizon(
        question, resolution_dates, source=source,
        prompt_variant=prompt_variant, model_override=model_override,
    )
    if probs is not None:
        return probs

    logger.info("amulti_forecast_single_fallback", question_id=question.id, n_horizons=n)
    fallback: list[float] = []
    for date in resolution_dates:
        try:
            p = await aforecast(
                question, resolution_date=date, source=source,
                prompt_variant=prompt_variant, model_override=model_override,
            )
        except Exception:
            logger.warning("amulti_forecast_single_fallback_error", question_id=question.id, date=date)
            p = 0.5
        fallback.append(p)
    return fallback


if __name__ == "__main__":
    import asyncio
    from eval import run_eval
    asyncio.run(run_eval(aforecast))
