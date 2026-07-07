"""Baseline LLM forecasting agent using litellm."""

from __future__ import annotations

import asyncio
import os
import re

import litellm

from fetch_data import Question

litellm.suppress_debug_info = True

FORECAST_MODEL = os.environ.get("FORECAST_MODEL", "claude-sonnet-4-20250514")

SYSTEM_PROMPT = (
    "You are a professional forecaster. You are given a question about a future event "
    "and must estimate the probability that it will resolve YES. "
    "Respond with ONLY a single decimal number between 0.0 and 1.0 representing your "
    "probability estimate. Do not include any other text, explanation, or formatting."
)


def _build_user_prompt(question: Question) -> str:
    parts = [f"Question: {question.question}"]
    if question.background:
        parts.append(f"Background: {question.background}")
    if question.resolution_criteria:
        parts.append(f"Resolution criteria: {question.resolution_criteria}")
    parts.append("Respond with a single probability between 0.0 and 1.0:")
    return "\n\n".join(parts)


def _parse_probability(text: str) -> float:
    match = re.search(r"(\d+\.?\d*)", text)
    if match:
        value = float(match.group(1))
        return max(0.0, min(1.0, value))
    return 0.5


async def aforecast(question: Question) -> float:
    """Async forecast using litellm.acompletion."""
    response = await litellm.acompletion(
        model=FORECAST_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": _build_user_prompt(question)},
        ],
        temperature=0.0,
    )
    content = response.choices[0].message.content or ""
    return _parse_probability(content)


def forecast(question: Question) -> float:
    """Sync wrapper around aforecast."""
    return asyncio.run(aforecast(question))
