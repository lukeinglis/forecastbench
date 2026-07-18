"""Baseline LLM forecaster using litellm for probability estimation."""

from __future__ import annotations

import json
import os
import re

import litellm

from fetch_data import Question

MODEL = os.getenv("FORECAST_MODEL", "claude-sonnet-4-20250514")
FORECAST_EXTRACTION_MODEL = os.getenv("FORECAST_EXTRACTION_MODEL", "gpt-4o-mini")

PROMPT_TEMPLATE = """You are an expert superforecaster with a track record of well-calibrated probabilistic predictions. Your goal is to estimate the probability that the following question resolves to YES.

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

DATASET_PROMPT_TEMPLATE = """You are an expert superforecaster, familiar with the research on forecasting. Your goal is to predict the probability of the resolution of the question at each of the resolution dates.

Today's Date: {today_date}
{data_availability_context}
Question: {question}

{background_section}{criteria_section}{freeze_value_section}Question resolution dates: {list_of_resolution_dates}

Output your answer (a number between 0 and 1) with an asterisk at the beginning and end of the decimal. (For example, if there are n resolution dates, you would output different *p* for each resolution date) Do not output anything else."""

FORECAST_EXTRACTION_PROMPT = """You are extracting probabilities from text.

Expected number of probabilities: {n_horizons}

Rules:
- Do not make a forecast.
- Do not infer missing values.
- Extract only probabilities explicitly stated in the MODEL RESPONSE.
- Preserve the order in which the probabilities appear.
- If you cannot identify exactly {n_horizons} final-answer probabilities, return [].

Return only a Python list of decimal probabilities, e.g. [0.1, 0.2, 0.3].

MODEL RESPONSE:
{model_response}"""


def _build_prompt(
    question: Question,
    resolution_date: str | None = None,
) -> str:
    today_date = getattr(question, "forecast_due_date", None) or question.freeze_datetime

    if question.freeze_datetime:
        temporal_context = f"Today's Date: {today_date}. " if today_date else ""
        temporal_context += (
            f"You should forecast based on information available as of {question.freeze_datetime}."
        )
    else:
        temporal_context = f"Today's Date: {today_date}." if today_date else ""

    source_context = (
        f"Source Context: {question.source_intro}\n\n"
        if getattr(question, "source_intro", None)
        else ""
    )

    resolution_date_section = (
        f"Target resolution date: {resolution_date}\n\n"
        if resolution_date
        else ""
    )

    background_section = f"Background: {question.background}\n" if question.background else ""
    criteria_section = (
        f"Resolution Criteria: {question.resolution_criteria}\n"
        if question.resolution_criteria
        else ""
    )

    return PROMPT_TEMPLATE.format(
        temporal_context=temporal_context,
        question=question.question,
        background_section=background_section,
        criteria_section=criteria_section,
        source_context=source_context,
        resolution_date_section=resolution_date_section,
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

    return DATASET_PROMPT_TEMPLATE.format(
        today_date=today_date,
        data_availability_context=data_availability_context,
        question=question.question,
        background_section=background_section,
        criteria_section=criteria_section,
        freeze_value_section=freeze_value_section,
        list_of_resolution_dates=list_of_resolution_dates,
    )


def _parse_probabilities(text: str, n_horizons: int) -> list[float]:
    matches = re.findall(r"\*(0?\.\d+|1\.0+)\*", text)
    if len(matches) == n_horizons:
        return [max(0.01, min(0.99, float(m))) for m in matches]

    try:
        extraction_prompt = FORECAST_EXTRACTION_PROMPT.format(
            n_horizons=n_horizons, model_response=text,
        )
        response = litellm.completion(
            model=FORECAST_EXTRACTION_MODEL,
            messages=[{"role": "user", "content": extraction_prompt}],
            temperature=0,
            timeout=30,
        )
        content = response.choices[0].message.content or ""
        list_match = re.search(r"\[([^\]]*)\]", content)
        if list_match:
            parsed = json.loads(f"[{list_match.group(1)}]")
            if isinstance(parsed, list) and len(parsed) == n_horizons:
                return [max(0.01, min(0.99, float(v))) for v in parsed]
    except Exception:
        pass

    return [0.5] * n_horizons


def _parse_probability(text: str) -> float:
    match = re.search(r"(?:^|\s|:)\s*(0?\.\d+|1\.0{0,}|0(?:\.0{0,})?)\s*$", text, re.MULTILINE)
    if not match:
        match = re.search(r"(0?\.\d+|1\.0{0,})", text)
    if match:
        prob = float(match.group(1))
        return max(0.01, min(0.99, prob))
    return 0.5


def forecast(
    question: Question,
    resolution_date: str | None = None,
) -> float:
    prompt = _build_prompt(question, resolution_date=resolution_date)
    response = litellm.completion(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
        timeout=60,
    )
    text = response.choices[0].message.content or ""
    return _parse_probability(text)


def forecast_multi(
    question: Question,
    resolution_dates: list[str],
) -> list[float]:
    prompt = _build_dataset_prompt(question, resolution_dates)
    response = litellm.completion(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
        timeout=90,
    )
    text = response.choices[0].message.content or ""
    return _parse_probabilities(text, len(resolution_dates))


async def aforecast(
    question: Question,
    resolution_date: str | None = None,
) -> float:
    prompt = _build_prompt(question, resolution_date=resolution_date)
    response = await litellm.acompletion(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
        timeout=60,
    )
    text = response.choices[0].message.content or ""
    return _parse_probability(text)


async def aforecast_multi(
    question: Question,
    resolution_dates: list[str],
) -> list[float]:
    prompt = _build_dataset_prompt(question, resolution_dates)
    response = await litellm.acompletion(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
        timeout=90,
    )
    text = response.choices[0].message.content or ""
    return _parse_probabilities(text, len(resolution_dates))


if __name__ == "__main__":
    import asyncio
    from eval import run_eval
    asyncio.run(run_eval(aforecast))
