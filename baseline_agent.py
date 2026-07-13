"""Baseline LLM forecaster using litellm for probability estimation."""

from __future__ import annotations

import os
import re

import litellm

from cutoff import CutoffEnvironment
from fetch_data import Question

MODEL = os.getenv("FORECAST_MODEL", "claude-sonnet-4-20250514")

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


def _build_prompt(
    question: Question,
    resolution_date: str | None = None,
) -> str:
    if question.freeze_datetime:
        env = CutoffEnvironment(question.freeze_datetime)
        prepared = env.prepare_question(question)
        temporal_context = env.frame_temporal_context(question)
    else:
        prepared = question
        temporal_context = ""

    source_context = (
        f"Source Context: {prepared.source_intro}\n\n"
        if getattr(prepared, "source_intro", None)
        else ""
    )

    resolution_date_section = (
        f"Target resolution date: {resolution_date}\n\n"
        if resolution_date
        else ""
    )

    background_section = f"Background: {prepared.background}\n" if prepared.background else ""
    criteria_section = (
        f"Resolution Criteria: {prepared.resolution_criteria}\n"
        if prepared.resolution_criteria
        else ""
    )

    return PROMPT_TEMPLATE.format(
        temporal_context=temporal_context,
        question=prepared.question,
        background_section=background_section,
        criteria_section=criteria_section,
        source_context=source_context,
        resolution_date_section=resolution_date_section,
    )


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


if __name__ == "__main__":
    import asyncio
    from eval import run_eval
    asyncio.run(run_eval(aforecast))
