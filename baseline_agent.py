"""Baseline LLM forecaster using litellm for probability estimation."""

from __future__ import annotations

import ast
import json
import os
import re
import threading
import time
from pathlib import Path
from typing import Any

import litellm

from fetch_data import MARKET_SOURCES, Question
from logging_config import get_logger

TIMESERIES_SOURCES = {"fred", "dbnomics", "yfinance"}

logger = get_logger("baseline_agent")

# Pinned to specific snapshot for benchmark reproducibility. Override via FORECAST_MODEL env var.
MODEL = os.getenv("FORECAST_MODEL", "vertex_ai/claude-sonnet-4-6")
EXTRACTION_MODEL = os.getenv("FORECAST_EXTRACTION_MODEL", "openai/gpt-4o-mini")
VERTEX_LOCATION = os.getenv("VERTEXAI_LOCATION", "europe-west1")
THINKING_ENABLED = os.getenv("FORECAST_THINKING", "true").lower() == "true"
MAX_TOKENS = int(os.getenv("FORECAST_MAX_TOKENS", "16384"))
ENSEMBLE_N = int(os.getenv("FORECAST_ENSEMBLE_N", "1"))
ENSEMBLE_TEMP = float(os.getenv("FORECAST_ENSEMBLE_TEMP", "0.7"))

_REFRESH_MARGIN_SECS = 300
_vertex_creds_lock = threading.Lock()
_vertex_credentials: Any = None
_vertex_token_expiry: float = 0.0


def _get_google_auth() -> tuple[Any, Any]:
    import google.auth
    import google.auth.transport.requests
    return google.auth, google.auth.transport.requests


def _ensure_vertex_credentials() -> None:
    """Refresh Google ADC credentials if using Vertex AI and token is expired or near-expiry."""
    if not MODEL.startswith("vertex_ai/"):
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

class LiteLLMAdapter:
    """Adapter bridging its_hub's AbstractLanguageModel to litellm."""

    def __init__(self, model: str, max_tokens: int, vertex_location: str) -> None:
        self.model = model
        self.max_tokens = max_tokens
        self.vertex_location = vertex_location

    async def agenerate_single(
        self,
        messages: list[dict[str, Any]],
        stop: str | None = None,
        **kwargs: Any,
    ) -> dict[str, str]:
        _ensure_vertex_credentials()
        call_kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": messages if isinstance(messages, list) else list(messages),
            "max_tokens": self.max_tokens,
            "vertex_location": self.vertex_location,
        }
        if "temperature" in kwargs:
            call_kwargs["temperature"] = kwargs["temperature"]
        else:
            call_kwargs["temperature"] = ENSEMBLE_TEMP
        if stop:
            call_kwargs["stop"] = stop
        call_kwargs["timeout"] = 180
        response = await litellm.acompletion(**call_kwargs)
        return {"role": "assistant", "content": response.choices[0].message.content or ""}


def _forecast_kwargs(
    messages: list[dict[str, str]],
    timeout: int = 180,
    source: str | None = None,
) -> dict[str, Any]:
    kwargs: dict[str, Any] = {
        "model": MODEL,
        "messages": messages,
        "max_tokens": MAX_TOKENS,
        "timeout": timeout,
        "vertex_location": VERTEX_LOCATION,
    }
    if source and source.lower() in TIMESERIES_SOURCES:
        kwargs["temperature"] = 0.3
    elif THINKING_ENABLED:
        kwargs["thinking"] = {"type": "enabled", "budget_tokens": MAX_TOKENS // 2}
    else:
        kwargs["temperature"] = 0.3
    return kwargs


RESPONSE_LOG_DIR = Path(".cache/response_logs")


def _save_response_log(question_id: str, response_text: str, outcome: str, n_expected: int) -> None:
    """Save raw model response for diagnostic analysis. Does not affect scoring."""
    try:
        RESPONSE_LOG_DIR.mkdir(parents=True, exist_ok=True)
        log_file = RESPONSE_LOG_DIR / f"{question_id}.json"
        log_file.write_text(json.dumps({
            "question_id": question_id,
            "outcome": outcome,
            "n_expected": n_expected,
            "response_length": len(response_text),
            "response": response_text,
        }, indent=2))
    except Exception:
        pass


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

SINGLE_DATE_DATASET_PROMPT = """\
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

Current value on {freeze_datetime}:
{freeze_datetime_value}

Value Explanation:
{freeze_datetime_value_explanation}

Today's Date: {today_date}

Question resolution date: {target_resolution_date}

Output your answer (a number between 0 and 1) with an asterisk at the beginning and end of the decimal.
Do not output anything else.
Answer: {{ Insert answer here }}"""

SCRATCHPAD_DATASET_PROMPT = """\
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

Before giving your final probabilities, work through these reasoning steps:

Reasoning:
1. What is the current value and what does it represent?
2. What would need to happen for the outcome at each resolution date?
3. What historical patterns or trends are relevant?
4. What is your base rate estimate?
5. How should you adjust from the base rate given current context?

Final Probabilities:
Output your answer (a number between 0 and 1) with an asterisk at the beginning and end of the decimal. \
(For example, if there are n resolution dates, you would output different *p* for each resolution date) \
Do not output anything else after the probabilities."""

DATASET_PROMPT_TEMPLATE = """You are an expert superforecaster, familiar with the research on forecasting. Your goal is to predict the probability of the resolution of the question at each of the resolution dates.

Today's Date: {today_date}
{data_availability_context}
Question: {question}

{background_section}{criteria_section}{freeze_value_section}Question resolution dates: {list_of_resolution_dates}

Output your answer (a number between 0 and 1) with an asterisk at the beginning and end of the decimal. (For example, if there are n resolution dates, you would output different *p* for each resolution date) Do not output anything else."""


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
        if prompt_variant == "zero-shot-no-fv":
            return ZERO_SHOT_MARKET_PROMPT.format(
                question=question.question,
                background=background,
                resolution_criteria=question.resolution_criteria or "",
                today_date=today_date,
                resolution_date=effective_resolution_date or "",
            )
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

    if resolution_date is not None:
        formatted_q = _format_question_text(question.question, today_date, is_dataset=True)
        return SINGLE_DATE_DATASET_PROMPT.format(
            question=formatted_q,
            background=background,
            resolution_criteria=question.resolution_criteria or "",
            freeze_datetime=fd,
            freeze_datetime_value=fv if fv is not None else "",
            freeze_datetime_value_explanation=getattr(question, "freeze_datetime_value_explanation", None) or "",
            today_date=today_date,
            target_resolution_date=resolution_date,
        )

    effective_rd = resolution_dates or getattr(question, "resolution_dates", None)
    dates_list: list[str] = []
    if effective_rd and isinstance(effective_rd, list):
        dates_list = [str(d) for d in effective_rd if d and str(d).upper() != "N/A"]

    formatted_q = _format_question_text(question.question, today_date, is_dataset=True)

    if prompt_variant in ("zero-shot", "zero-shot-fv", "dataset"):
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

    if prompt_variant == "default" and effective_source.lower() in TIMESERIES_SOURCES:
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

    return SCRATCHPAD_DATASET_PROMPT.format(
        question=formatted_q,
        background=background,
        resolution_criteria=question.resolution_criteria or "",
        freeze_datetime=fd,
        freeze_datetime_value=fv if fv is not None else "",
        freeze_datetime_value_explanation=getattr(question, "freeze_datetime_value_explanation", None) or "",
        today_date=today_date,
        list_of_resolution_dates=dates_list,
    )


def _build_dataset_prompt(
    question: Question,
    resolution_dates: list[str],
) -> str:
    today_date = getattr(question, "forecast_due_date", None) or question.freeze_datetime or ""

    data_availability_context = (
        f"You should forecast based on information available as of {question.freeze_datetime}."
        if question.freeze_datetime
        else ""
    )

    background_section = f"Question Background: {question.background}\n" if question.background else ""
    criteria_section = (
        f"Resolution Criteria: {question.resolution_criteria}\n"
        if question.resolution_criteria
        else ""
    )

    freeze_value_section = ""
    if question.freeze_datetime_value is not None:
        freeze_value_section = f"Current value on {question.freeze_datetime}: {question.freeze_datetime_value}\n"
        if question.freeze_datetime_value_explanation:
            freeze_value_section += f"Value Explanation: {question.freeze_datetime_value_explanation}\n"

    list_of_resolution_dates = ", ".join(resolution_dates)

    formatted_q = _format_question_text(question.question, today_date, is_dataset=True)
    return DATASET_PROMPT_TEMPLATE.format(
        today_date=today_date,
        data_availability_context=data_availability_context,
        question=formatted_q,
        background_section=background_section,
        criteria_section=criteria_section,
        freeze_value_section=freeze_value_section,
        list_of_resolution_dates=list_of_resolution_dates,
    )


_FULLMATCH_RE = re.compile(r"\*?\s*(0?\.\d+|1\.0{0,}|0(?:\.0{0,})?)\s*\*?")


def _parse_probabilities(text: str, n_horizons: int) -> list[float]:
    matches = re.findall(r"\*(0?\.\d+|1\.0+)\*", text)
    if len(matches) == n_horizons:
        return [float(m) for m in matches]

    try:
        extraction_prompt = FORECAST_EXTRACTION_PROMPT.format(
            n_horizons=n_horizons, model_response=text,
        )
        response = litellm.completion(
            model=EXTRACTION_MODEL,
            messages=[{"role": "user", "content": extraction_prompt}],
            temperature=0,
            timeout=30,
        )
        content = response.choices[0].message.content or ""
        list_match = re.search(r"\[([^\]]*)\]", content)
        if list_match:
            parsed = json.loads(f"[{list_match.group(1)}]")
            if isinstance(parsed, list) and len(parsed) == n_horizons:
                return [float(v) for v in parsed]
    except Exception:
        pass

    raise ValueError(f"Could not extract {n_horizons} probabilities from response")


def _parse_probability(text: str) -> float:
    fm = _FULLMATCH_RE.fullmatch(text.strip())
    if fm:
        prob = float(fm.group(1))
        logger.debug("parsed_probability", raw_match=fm.group(1), parsed=prob, format="fullmatch")
        return prob
    asterisk = re.search(r"\*\s*(0?\.\d+|1\.0{0,}|0(?:\.0{0,})?)\s*\*", text)
    if asterisk:
        prob = float(asterisk.group(1))
        logger.debug("parsed_probability", raw_match=asterisk.group(1), parsed=prob, format="asterisk")
        return prob
    match = re.search(r"[Pp]robability[\s:=]+\s*(0?\.\d+|1\.0{0,}|0(?:\.0{0,})?)", text)
    if not match:
        match = re.search(r"(?:^|\s|:)\s*(0?\.\d+|1\.0{0,}|0(?:\.0{0,})?)\s*$", text, re.MULTILINE)
    if not match:
        match = re.search(r"(0?\.\d+|1\.0{0,})", text)
    if match:
        prob = float(match.group(1))
        logger.debug("parsed_probability", raw_match=match.group(1), parsed=prob, format="standard")
        return prob
    raise ValueError(f"Could not parse probability from response: {text[:100]}")


async def _ensemble_forecast(prompt: str, source: str | None = None) -> float | None:
    """Generate N responses via its_hub LMOrchestrator and average probabilities."""
    from its_hub.api.types import ChatMessages
    from its_hub.core.orchestrator import LMOrchestrator

    lm = LiteLLMAdapter(MODEL, MAX_TOKENS, VERTEX_LOCATION)
    orchestrator = LMOrchestrator(max_concurrency=ENSEMBLE_N)

    messages = [{"role": "user", "content": prompt}]
    chat_messages = ChatMessages(messages)
    batch = chat_messages.to_batch(ENSEMBLE_N)

    logger.info("ensemble_start", ensemble_n=ENSEMBLE_N, ensemble_temp=ENSEMBLE_TEMP)

    responses = await orchestrator.agenerate(lm, batch, temperature=ENSEMBLE_TEMP)

    probabilities: list[float] = []
    for i, resp in enumerate(responses):
        try:
            content = resp.get("content", "") if isinstance(resp, dict) else str(resp)
            prob = _parse_probability(content)
            probabilities.append(prob)
            logger.debug("ensemble_member_result", member=i, probability=prob)
        except (ValueError, Exception):
            logger.warning("ensemble_member_failed", member=i)

    if not probabilities:
        return None

    mean_prob = sum(probabilities) / len(probabilities)
    std = (
        (sum((p - mean_prob) ** 2 for p in probabilities) / len(probabilities)) ** 0.5
        if len(probabilities) > 1
        else 0.0
    )
    logger.info(
        "ensemble_aggregated", probabilities=probabilities, mean=mean_prob, std=std,
    )
    return mean_prob


async def _ensemble_forecast_multi_horizon(
    prompt: str, n_horizons: int, question_id: str, source: str | None = None,
) -> list[float] | None:
    """Generate N multi-horizon responses and average per-horizon probabilities."""
    from its_hub.api.types import ChatMessages
    from its_hub.core.orchestrator import LMOrchestrator

    lm = LiteLLMAdapter(MODEL, MAX_TOKENS, VERTEX_LOCATION)
    orchestrator = LMOrchestrator(max_concurrency=ENSEMBLE_N)

    messages = [{"role": "user", "content": prompt}]
    chat_messages = ChatMessages(messages)
    batch = chat_messages.to_batch(ENSEMBLE_N)

    logger.info(
        "ensemble_multi_horizon_start",
        question_id=question_id,
        ensemble_n=ENSEMBLE_N,
        n_horizons=n_horizons,
    )

    responses = await orchestrator.agenerate(lm, batch, temperature=ENSEMBLE_TEMP)

    all_probs: list[list[float]] = []
    for i, resp in enumerate(responses):
        content = resp.get("content", "") if isinstance(resp, dict) else str(resp)
        probs = _extract_probabilities(content, n_horizons)
        if probs is not None:
            all_probs.append(probs)
            logger.debug("ensemble_multi_member_result", member=i, probabilities=probs)
        else:
            logger.warning("ensemble_multi_member_failed", member=i)

    if not all_probs:
        return None

    averaged = [
        sum(member[h] for member in all_probs) / len(all_probs)
        for h in range(n_horizons)
    ]
    logger.info("ensemble_multi_aggregated", n_members=len(all_probs), averaged=averaged)
    return averaged


def forecast(
    question: Question,
    resolution_date: str | None = None,
    source: str | None = None,
    resolution_dates: Any = None,
    prompt_variant: str = "zero-shot",
) -> float:
    logger.info("forecast_start", question_id=question.id, model=MODEL, prompt_variant=prompt_variant)
    _ensure_vertex_credentials()
    effective_source = source or question.source
    prompt = _build_prompt(
        question,
        resolution_date=resolution_date,
        source=source,
        resolution_dates=resolution_dates,
        prompt_variant=prompt_variant,
    )

    if ENSEMBLE_N > 1:
        messages = [{"role": "user", "content": prompt}]
        ensemble_kwargs: dict[str, Any] = {
            "model": MODEL,
            "messages": messages,
            "max_tokens": MAX_TOKENS,
            "vertex_location": VERTEX_LOCATION,
            "temperature": ENSEMBLE_TEMP,
            "timeout": 180,
        }
        probabilities: list[float] = []
        for i in range(ENSEMBLE_N):
            try:
                resp = litellm.completion(**ensemble_kwargs)
                text = resp.choices[0].message.content or ""
                prob = _parse_probability(text)
                probabilities.append(prob)
            except Exception:
                logger.warning("sync_ensemble_member_failed", member=i)
        if probabilities:
            mean_prob = sum(probabilities) / len(probabilities)
            logger.info("sync_ensemble_aggregated", probabilities=probabilities, mean=mean_prob)
            return mean_prob

    try:
        messages = [{"role": "user", "content": prompt}]
        response = litellm.completion(**_forecast_kwargs(messages, source=effective_source))
    except Exception:
        logger.error("forecast_api_error", question_id=question.id, model=MODEL, exc_info=True)
        raise
    text = response.choices[0].message.content or ""
    prob = _parse_probability(text)
    logger.info("forecast_complete", question_id=question.id, forecast_value=prob, parse_success=True)
    return prob


def forecast_multi(
    question: Question,
    resolution_dates: list[str],
) -> list[float]:
    _ensure_vertex_credentials()
    prompt = _build_dataset_prompt(question, resolution_dates)
    messages = [{"role": "user", "content": prompt}]
    response = litellm.completion(**_forecast_kwargs(messages, source=question.source))
    text = response.choices[0].message.content or ""
    return _parse_probabilities(text, len(resolution_dates))


async def aforecast(
    question: Question,
    resolution_date: str | None = None,
    source: str | None = None,
    resolution_dates: Any = None,
    prompt_variant: str = "zero-shot",
) -> float:
    logger.info(
        "forecast_start",
        question_id=question.id,
        model=MODEL,
        prompt_variant=prompt_variant,
        async_mode=True,
    )
    _ensure_vertex_credentials()
    effective_source = source or question.source
    prompt = _build_prompt(
        question,
        resolution_date=resolution_date,
        source=source,
        resolution_dates=resolution_dates,
        prompt_variant=prompt_variant,
    )

    if ENSEMBLE_N > 1:
        result = await _ensemble_forecast(prompt, source=effective_source)
        if result is not None:
            logger.info(
                "forecast_complete",
                question_id=question.id,
                forecast_value=result,
                parse_success=True,
                ensemble=True,
            )
            return result
        logger.warning("ensemble_fallback_to_single", question_id=question.id)

    try:
        messages = [{"role": "user", "content": prompt}]
        response = await litellm.acompletion(**_forecast_kwargs(messages, source=effective_source))
    except Exception:
        logger.error("forecast_api_error", question_id=question.id, model=MODEL, exc_info=True)
        raise
    text = response.choices[0].message.content or ""
    prob = _parse_probability(text)
    logger.info("forecast_complete", question_id=question.id, forecast_value=prob, parse_success=True)
    return prob


def _to_float(v: float) -> float:
    return float(v)


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
        return [_to_float(float(m)) for m in asterisks]
    decimals = _DECIMAL_RE.findall(text)
    valid = [float(d) for d in decimals if 0 <= float(d) <= 1]
    if len(valid) == n_expected:
        return [_to_float(v) for v in valid]
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
        return [_to_float(p) for p in probabilities]
    if len(probabilities) > n_expected:
        return [_to_float(p) for p in probabilities[-n_expected:]]
    return None


def _asterisk_extract(text: str, n_expected: int) -> list[float] | None:
    """Find asterisk-wrapped probabilities in the full text."""
    matches = _ASTERISK_RE.findall(text)
    if len(matches) == n_expected:
        return [_to_float(float(m)) for m in matches]
    if len(matches) > n_expected:
        return [_to_float(float(m)) for m in matches[-n_expected:]]
    return None


def _decimal_extract(text: str, n_expected: int) -> list[float] | None:
    """Find any decimal probabilities in the full text."""
    all_decimals = _DECIMAL_RE.findall(text)
    valid = [float(d) for d in all_decimals if 0 <= float(d) <= 1]
    if len(valid) == n_expected:
        return [_to_float(v) for v in valid]
    if len(valid) > n_expected:
        return [_to_float(v) for v in valid[-n_expected:]]
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
                return [float(v) for v in parsed]
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

    effective_source = source or question.source
    _ensure_vertex_credentials()
    prompt = _build_prompt(
        question,
        source=source,
        resolution_dates=resolution_dates,
        prompt_variant=prompt_variant,
    )

    if ENSEMBLE_N > 1:
        ensemble_probs = await _ensemble_forecast_multi_horizon(
            prompt, n_horizons, question.id, source=effective_source,
        )
        if ensemble_probs is not None:
            logger.info(
                "multi_horizon_complete",
                question_id=question.id,
                n_horizons=n_horizons,
                method="ensemble",
            )
            _save_response_log(question.id, str(ensemble_probs), "ensemble_success", n_horizons)
            return ensemble_probs
        logger.warning("ensemble_multi_fallback_to_single", question_id=question.id)

    try:
        messages = [{"role": "user", "content": prompt}]
        response = await litellm.acompletion(**_forecast_kwargs(messages, source=effective_source))
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
        _save_response_log(question.id, text, "regex_success", n_horizons)
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
        _save_response_log(question.id, text, "llm_success", n_horizons)
        return probs

    logger.warning(
        "multi_horizon_fallback",
        question_id=question.id,
        n_horizons=n_horizons,
    )
    _save_response_log(question.id, text, "fallback", n_horizons)
    return None


async def aforecast_multi(
    question: Question,
    resolution_dates: list[str],
) -> list[float]:
    _ensure_vertex_credentials()
    prompt = _build_dataset_prompt(question, resolution_dates)
    messages = [{"role": "user", "content": prompt}]
    response = await litellm.acompletion(**_forecast_kwargs(messages, source=question.source))
    text = response.choices[0].message.content or ""
    return _parse_probabilities(text, len(resolution_dates))


if __name__ == "__main__":
    import asyncio
    from eval import run_eval
    asyncio.run(run_eval(aforecast))
