"""LLM baseline forecaster for ForecastBench evaluation."""

from __future__ import annotations

import logging
import os
import re

import litellm
from fetch_data import Question

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "vertex_ai/claude-sonnet-4@20250514"
DEFAULT_VERTEX_PROJECT = "itpc-gcp-product-all-claude"
DEFAULT_VERTEX_LOCATION = "us-east5"


def build_prompt(question: Question, today_date: str) -> str:
    parts = [
        "You are an expert superforecaster, familiar with the work of Tetlock and others.",
        "Make a prediction of the probability that the question will be resolved as true.",
        "You MUST give a probability estimate between 0 and 1 UNDER ALL CIRCUMSTANCES.",
        "If for some reason you can't answer, pick the base rate,",
        "but return a number between 0 and 1.",
        "",
        "Question:",
        question.question,
        "",
        "Question Background:",
        question.background,
        "",
        "Resolution Criteria:",
        question.resolution_criteria,
        "",
    ]

    if question.freeze_datetime_value is not None:
        parts.append(
            f"Current value on {question.freeze_datetime}: {question.freeze_datetime_value}"
        )
        parts.append("")

    parts.extend(
        [
            f"Today's Date: {today_date}",
            "",
            "Output your answer (a number between 0 and 1) with an asterisk"
            " at the beginning and end of the decimal.",
            "Do not output anything else.",
        ]
    )

    return "\n".join(parts)


def parse_probability(text: str | None) -> float:
    if not text:
        logger.warning("Empty LLM response, returning 0.5")
        return 0.5

    # asterisk pattern: *0.75*
    m = re.search(r"\*\s*(-?[0-9]*\.?[0-9]+)\s*\*", text)
    if m:
        return max(0.0, min(1.0, float(m.group(1))))

    # "Forecast:" pattern
    m = re.search(r"[Ff]orecast:\s*([0-9]*\.?[0-9]+)", text)
    if m:
        return max(0.0, min(1.0, float(m.group(1))))

    # bare decimal 0.XX
    m = re.search(r"\b([0-9]*\.[0-9]+)\b", text)
    if m:
        return max(0.0, min(1.0, float(m.group(1))))

    # percentage: 75%
    m = re.search(r"([0-9]+(?:\.[0-9]+)?)\s*%", text)
    if m:
        return max(0.0, min(1.0, float(m.group(1)) / 100.0))

    logger.warning("Could not parse probability from LLM response, returning 0.5")
    return 0.5


def forecast(question: Question, model: str | None = None, today_date: str | None = None) -> float:
    model = model or os.environ.get("FORECAST_MODEL", DEFAULT_MODEL)

    if today_date is None:
        today_date = question.freeze_datetime or "Unknown"

    prompt = build_prompt(question, today_date)

    vertex_kwargs = {}
    if model.startswith("vertex_ai/"):
        vertex_kwargs["vertex_project"] = os.environ.get("VERTEX_PROJECT", DEFAULT_VERTEX_PROJECT)
        vertex_kwargs["vertex_location"] = os.environ.get("VERTEX_LOCATION", DEFAULT_VERTEX_LOCATION)

    try:
        response = litellm.completion(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            max_tokens=2000,
            **vertex_kwargs,
        )
        content = response.choices[0].message.content
        return parse_probability(content)
    except Exception:
        logger.warning("LLM call failed, returning 0.5", exc_info=True)
        return 0.5


if __name__ == "__main__":
    from eval import run_baseline_eval

    run_baseline_eval()
