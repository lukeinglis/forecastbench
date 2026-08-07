"""Belief state tracking forecaster — BLF-style iterative reasoning with multi-trial aggregation."""

from __future__ import annotations

import asyncio
import json
import math
import os
import re
from dataclasses import dataclass, field
from typing import Any

import litellm

from lab_forecaster import (
    MODEL,
    TEMPERATURE,
    MAX_TOKENS,
    _ensure_vertex_credentials,
)
from fetch_data import Question
from logging_config import get_logger

logger = get_logger("belief_forecaster")

N_ITERATIONS = int(os.getenv("FORECAST_BELIEF_ITERATIONS", "3"))
N_TRIALS = int(os.getenv("FORECAST_BELIEF_TRIALS", "5"))


@dataclass
class BeliefState:
    probability: float = 0.5
    confidence: str = "low"
    evidence_for: list[str] = field(default_factory=list)
    evidence_against: list[str] = field(default_factory=list)
    reasoning: str = ""


BELIEF_PROMPT = """\
You are an expert superforecaster maintaining a structured belief state.

Question:
{question}

Background:
{background}

Resolution Criteria:
{resolution_criteria}

Today's Date: {today_date}
Resolution Date: {resolution_date}

Current Belief State (iteration {iteration} of {n_iterations}):
- Probability: {probability}
- Confidence: {confidence}
- Evidence For: {evidence_for}
- Evidence Against: {evidence_against}
- Prior Reasoning: {reasoning}

Instructions:
1. Review the question and your current belief state.
2. Consider new evidence for and against resolution.
3. Update your probability estimate, moving it toward your best judgment.
4. Increase confidence as you accumulate evidence.

You MUST output valid JSON matching this schema:
{{
  "probability": <float between 0 and 1>,
  "confidence": "<low|medium|high>",
  "evidence_for": ["<evidence point>", ...],
  "evidence_against": ["<evidence point>", ...],
  "reasoning": "<brief reasoning for this update>"
}}

Output ONLY the JSON object, no other text."""


def _parse_belief_state(text: str) -> BeliefState:
    """Parse a BeliefState from LLM response text."""
    json_match = re.search(r"\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}", text, re.DOTALL)
    if not json_match:
        raise ValueError(f"No JSON object found in response: {text[:200]}")

    data = json.loads(json_match.group())

    prob = float(data.get("probability", 0.5))
    prob = max(0.0, min(1.0, prob))

    confidence = str(data.get("confidence", "low")).lower()
    if confidence not in ("low", "medium", "high"):
        confidence = "low"

    evidence_for = data.get("evidence_for", [])
    if not isinstance(evidence_for, list):
        evidence_for = [str(evidence_for)]

    evidence_against = data.get("evidence_against", [])
    if not isinstance(evidence_against, list):
        evidence_against = [str(evidence_against)]

    return BeliefState(
        probability=prob,
        confidence=confidence,
        evidence_for=[str(e) for e in evidence_for],
        evidence_against=[str(e) for e in evidence_against],
        reasoning=str(data.get("reasoning", "")),
    )


def _build_belief_prompt(
    question: Question,
    state: BeliefState,
    iteration: int,
    n_iterations: int,
    resolution_date: str | None = None,
) -> str:
    today_date = (
        getattr(question, "forecast_due_date", None)
        or question.freeze_datetime
        or ""
    )
    effective_resolution_date = resolution_date or ""

    return BELIEF_PROMPT.format(
        question=question.question,
        background=question.background or "",
        resolution_criteria=question.resolution_criteria or "",
        today_date=today_date,
        resolution_date=effective_resolution_date,
        iteration=iteration,
        n_iterations=n_iterations,
        probability=state.probability,
        confidence=state.confidence,
        evidence_for=json.dumps(state.evidence_for),
        evidence_against=json.dumps(state.evidence_against),
        reasoning=state.reasoning or "(none yet)",
    )


def _aggregate_forecasts(probabilities: list[float]) -> float:
    """Aggregate probabilities via log-odds mean, clamping extremes."""
    eps = 1e-6
    clamped = [max(eps, min(1 - eps, p)) for p in probabilities]
    log_odds = [math.log(p / (1 - p)) for p in clamped]
    mean_log_odds = sum(log_odds) / len(log_odds)
    return 1.0 / (1.0 + math.exp(-mean_log_odds))


async def _single_trial(
    question: Question,
    resolution_date: str | None = None,
    source: str | None = None,
    resolution_dates: Any = None,
    prompt_variant: str = "zero-shot",
    n_iterations: int = N_ITERATIONS,
) -> float:
    """Run a single belief-state trial with iterative refinement."""
    state = BeliefState()

    for i in range(1, n_iterations + 1):
        prompt = _build_belief_prompt(
            question, state, iteration=i, n_iterations=n_iterations,
            resolution_date=resolution_date,
        )

        _ensure_vertex_credentials()
        try:
            response = await litellm.acompletion(
                model=MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=TEMPERATURE,
                max_tokens=MAX_TOKENS,
                timeout=60,
            )
        except Exception:
            logger.warning(
                "belief_iteration_api_error",
                question_id=question.id,
                iteration=i,
                exc_info=True,
            )
            break

        text = response.choices[0].message.content or ""
        try:
            state = _parse_belief_state(text)
        except (json.JSONDecodeError, ValueError):
            logger.warning(
                "belief_iteration_parse_error",
                question_id=question.id,
                iteration=i,
                response_preview=text[:200],
            )
            break

    return state.probability


async def belief_forecast(
    question: Question,
    resolution_date: str | None = None,
    source: str | None = None,
    resolution_dates: Any = None,
    prompt_variant: str = "zero-shot",
    n_trials: int = N_TRIALS,
    n_iterations: int = N_ITERATIONS,
) -> float:
    """Run multiple independent belief-state trials and aggregate via log-odds mean."""
    logger.info(
        "belief_forecast_start",
        question_id=question.id,
        n_trials=n_trials,
        n_iterations=n_iterations,
    )

    tasks = [
        _single_trial(
            question,
            resolution_date=resolution_date,
            source=source,
            resolution_dates=resolution_dates,
            prompt_variant=prompt_variant,
            n_iterations=n_iterations,
        )
        for _ in range(n_trials)
    ]

    results = await asyncio.gather(*tasks)
    probabilities = list(results)

    if len(probabilities) == 1:
        aggregated = probabilities[0]
    else:
        aggregated = _aggregate_forecasts(probabilities)

    logger.info(
        "belief_forecast_complete",
        question_id=question.id,
        individual=probabilities,
        aggregated=round(aggregated, 4),
    )
    return aggregated


async def belief_forecast_multi_horizon(
    question: Question,
    resolution_dates: list[str],
    source: str | None = None,
    prompt_variant: str = "zero-shot",
    n_trials: int = N_TRIALS,
    n_iterations: int = N_ITERATIONS,
) -> list[float] | None:
    """Forecast multiple horizons, each via independent belief-state trials."""
    logger.info(
        "belief_multi_horizon_start",
        question_id=question.id,
        n_horizons=len(resolution_dates),
    )

    results: list[float] = []
    for date_str in resolution_dates:
        prob = await belief_forecast(
            question,
            resolution_date=date_str,
            source=source,
            resolution_dates=resolution_dates,
            prompt_variant=prompt_variant,
            n_trials=n_trials,
            n_iterations=n_iterations,
        )
        results.append(prob)

    return results


if __name__ == "__main__":
    from eval import run_eval
    asyncio.run(run_eval(belief_forecast))
