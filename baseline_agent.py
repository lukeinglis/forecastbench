"""Baseline LLM forecasting agent using litellm."""

from __future__ import annotations

import asyncio
import os
import re

import litellm

from cutoff import CutoffEnvironment, format_cutoff_instruction
from fetch_data import Question

litellm.suppress_debug_info = True

FORECAST_MODEL = os.environ.get("FORECAST_MODEL", "claude-sonnet-4-20250514")

SYSTEM_PROMPT = (
    "You are an expert probabilistic forecaster trained in calibration and base rate reasoning. "
    "You are given a question about a future event and must estimate the probability that it "
    "will resolve YES.\n\n"
    "Guidelines:\n"
    "- Consider base rates for similar events\n"
    "- Account for regression to the mean\n"
    "- Avoid overconfidence: use probabilities near 0.5 when genuinely uncertain\n"
    "- Reserve extreme probabilities (below 0.05 or above 0.95) for near-certain outcomes\n"
    "- Consider the question's resolution criteria carefully\n\n"
    "Respond with ONLY a single decimal number between 0.0 and 1.0 representing your "
    "probability estimate. Do not include any other text, explanation, or formatting."
)


def _build_user_prompt(
    question: Question, cutoff_instruction: str | None = None
) -> str:
    parts = [f"Question: {question.question}"]
    if question.background:
        parts.append(f"Background: {question.background}")
    if question.resolution_criteria:
        parts.append(f"Resolution criteria: {question.resolution_criteria}")
    if cutoff_instruction:
        parts.append(cutoff_instruction)
    parts.append("Respond with a single probability between 0.0 and 1.0:")
    return "\n\n".join(parts)


def _parse_probability(text: str) -> float:
    match = re.search(r"(\d+\.?\d*)", text)
    if match:
        value = float(match.group(1))
        return max(0.0, min(1.0, value))
    return 0.5


async def aforecast(
    question: Question, cutoff: CutoffEnvironment | None = None
) -> float:
    """Async forecast using litellm.acompletion."""
    cutoff_instruction: str | None = None
    if cutoff is not None:
        cutoff_instruction = format_cutoff_instruction(cutoff.freeze_datetime)

    response = await litellm.acompletion(
        model=FORECAST_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": _build_user_prompt(question, cutoff_instruction)},
        ],
        temperature=0.0,
    )
    content = response.choices[0].message.content or ""
    return _parse_probability(content)


def forecast(
    question: Question, cutoff: CutoffEnvironment | None = None
) -> float:
    """Sync wrapper around aforecast."""
    return asyncio.run(aforecast(question, cutoff))
