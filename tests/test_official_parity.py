"""Parity tests against the official ForecastBench pipeline.

These tests verify that our prompt rendering, parsing, and scoring produce
identical results to the official forecastingresearch/forecastbench repo.

Reference code is snapshotted from:
  https://github.com/forecastingresearch/forecastbench
  Commit: main branch as of 2026-07-20

If the official pipeline changes, update the reference snapshots here.
"""

from __future__ import annotations

import re
from string import Formatter
from typing import Any

import pytest

from baseline_agent import (
    _build_prompt,
    _format_question_text,
    _parse_probability,
    _parse_probabilities,
)
from fetch_data import Question
from score import brier_score, brier_index


# ---------------------------------------------------------------------------
# Reference snapshots from official forecastingresearch/forecastbench
# Source: src/llm_forecaster/prompts.py
# ---------------------------------------------------------------------------

OFFICIAL_ZERO_SHOT_MARKET_PROMPT = """
You are an expert superforecaster, familiar with the work of Tetlock and others. Make a prediction of the probability that the question will be resolved as true. You MUST give a probability estimate between 0 and 1 UNDER ALL CIRCUMSTANCES. If for some reason you can't answer, pick the base rate, but return a number between 0 and 1.

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
Answer: {{ Insert answer here }}
"""

OFFICIAL_ZERO_SHOT_DATASET_PROMPT = """
You are an expert superforecaster, familiar with the work of Tetlock and others. Make a prediction of the probability that the question will be resolved as true. You MUST give a probability estimate between 0 and 1 UNDER ALL CIRCUMSTANCES. If for some reason you can't answer, pick the base rate, but return a number between 0 and 1.

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

Output your answer (a number between 0 and 1) with an asterisk at the beginning and end of the decimal. (For example, if there are n resolution dates, you would output different *p* for each resolution date) Do not output anything else.
Answer: {{ Insert answer here }}
"""

OFFICIAL_ZERO_SHOT_MARKET_WITH_FREEZE_VALUE_PROMPT = """
You are an expert superforecaster, familiar with the work of Tetlock and others. Make a prediction of the probability that the question will be resolved as true. You MUST give a probability estimate between 0 and 1 UNDER ALL CIRCUMSTANCES. If for some reason you can't answer, pick the base rate, but return a number between 0 and 1.

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
Answer: {{ Insert answer here }}
"""


# ---------------------------------------------------------------------------
# Reference: official render_template (src/llm_forecaster/prompts.py)
# ---------------------------------------------------------------------------

_FORMATTER = Formatter()


def _official_template_field_names(template: str) -> set[str]:
    field_names = set()
    for _literal_text, field_name, _format_spec, _conversion in _FORMATTER.parse(template):
        if field_name is None:
            continue
        if field_name == "":
            raise ValueError("Anonymous prompt fields are not supported.")
        field_root = field_name.split(".", maxsplit=1)[0].split("[", maxsplit=1)[0]
        field_names.add(field_root)
    return field_names


def _official_render_template(template: str, params: dict[str, Any]) -> str:
    required_fields = _official_template_field_names(template)
    provided_fields = set(params)
    missing = sorted(required_fields - provided_fields)
    if missing:
        raise ValueError(f"Missing prompt fields: {', '.join(missing)}")
    extra = sorted(provided_fields - required_fields)
    if extra:
        raise ValueError(f"Unexpected prompt fields: {', '.join(extra)}")
    return template.format(**params)


# ---------------------------------------------------------------------------
# Reference: official parsing (src/llm_forecaster/parsing.py)
# ---------------------------------------------------------------------------

_OFFICIAL_PROBABILITY_TOKEN = r"(?:\*)?(\d*\.?\d+)(?:\*)?"
_OFFICIAL_PROBABILITY_PATTERN = re.compile(_OFFICIAL_PROBABILITY_TOKEN)


def _official_extract_probability(text: str | None) -> float | None:
    if text is None or text.strip() == "":
        return None
    m = _OFFICIAL_PROBABILITY_PATTERN.fullmatch(text.strip())
    if m is None:
        return None
    number = float(m.group(1))
    if 0 <= number <= 1:
        return number
    return None


def _official_extract_probabilities(text: str | None) -> list[float]:
    if text is None or text.strip() == "":
        return []
    probabilities = []
    for token in text.strip().replace(",", " ").split():
        probability = _official_extract_probability(token)
        if probability is None:
            return []
        probabilities.append(probability)
    return probabilities


# ---------------------------------------------------------------------------
# Reference: official question formatting (src/llm_forecaster/runner.py)
# ---------------------------------------------------------------------------

OFFICIAL_DATASET_SOURCE_NAMES = frozenset(
    ["acled", "dbnomics", "fred", "wikipedia", "yfinance"]
)


def _official_formatted_question(
    question: dict[str, Any], forecast_due_date: str,
) -> str:
    if question["source"] not in OFFICIAL_DATASET_SOURCE_NAMES:
        return question["question"]
    return question["question"].format(
        forecast_due_date=forecast_due_date,
        resolution_date="each of the resolution dates provided below",
    )


def _official_background(question: dict[str, Any]) -> str:
    background = question["background"]
    if question.get("market_info_resolution_criteria", "N/A") != "N/A":
        background += "\n" + question["market_info_resolution_criteria"]
    return background


def _official_prompt_params(
    question: dict[str, Any],
    forecast_due_date: str,
    today_date: str,
) -> dict[str, Any]:
    params = {
        "question": _official_formatted_question(question, forecast_due_date),
        "background": _official_background(question),
        "resolution_criteria": question["resolution_criteria"],
        "today_date": today_date,
    }
    if question["source"] in OFFICIAL_DATASET_SOURCE_NAMES:
        params.update({
            "freeze_datetime": question["freeze_datetime"],
            "freeze_datetime_value": question["freeze_datetime_value"],
            "freeze_datetime_value_explanation": question["freeze_datetime_value_explanation"],
            "list_of_resolution_dates": question["resolution_dates"],
        })
    else:
        params["resolution_date"] = question["market_info_close_datetime"]
    return params


def _official_render_prompt(
    question: dict[str, Any],
    forecast_due_date: str,
    today_date: str,
) -> str:
    if question["source"] in OFFICIAL_DATASET_SOURCE_NAMES:
        template = OFFICIAL_ZERO_SHOT_DATASET_PROMPT
    else:
        template = OFFICIAL_ZERO_SHOT_MARKET_PROMPT
    params = _official_prompt_params(question, forecast_due_date, today_date)
    return _official_render_template(template, params)


# ---------------------------------------------------------------------------
# Test helpers
# ---------------------------------------------------------------------------

def _make_dataset_question(
    question_text: str = "Will GDP exceed threshold by {resolution_date}?",
    forecast_due_date: str = "2026-07-05",
) -> tuple[Question, dict[str, Any]]:
    q = Question(
        id="dq1",
        source="fred",
        question=question_text,
        background="Economic indicator question",
        resolution_criteria="Resolves YES if value exceeds threshold.",
        freeze_datetime="2026-06-25",
        freeze_datetime_value=4.534,
        freeze_datetime_value_explanation="Current aggregate value",
        resolution_dates=["2026-07-12", "2026-08-04", "2026-10-03"],
        forecast_due_date=forecast_due_date,
    )
    q_dict = {
        "id": "dq1",
        "source": "fred",
        "question": question_text,
        "background": "Economic indicator question",
        "resolution_criteria": "Resolves YES if value exceeds threshold.",
        "freeze_datetime": "2026-06-25",
        "freeze_datetime_value": 4.534,
        "freeze_datetime_value_explanation": "Current aggregate value",
        "resolution_dates": ["2026-07-12", "2026-08-04", "2026-10-03"],
        "market_info_resolution_criteria": "N/A",
        "market_info_close_datetime": None,
    }
    return q, q_dict


def _make_market_question(
    forecast_due_date: str = "2026-07-05",
) -> tuple[Question, dict[str, Any]]:
    q = Question(
        id="mq1",
        source="polymarket",
        question="Will event X happen?",
        background="Market background info",
        resolution_criteria="Resolves YES if event X occurs.",
        freeze_datetime="2026-06-25",
        market_info_close_datetime="2026-12-31",
        market_info_resolution_criteria="N/A",
        forecast_due_date=forecast_due_date,
    )
    q_dict = {
        "id": "mq1",
        "source": "polymarket",
        "question": "Will event X happen?",
        "background": "Market background info",
        "resolution_criteria": "Resolves YES if event X occurs.",
        "freeze_datetime": "2026-06-25",
        "market_info_close_datetime": "2026-12-31",
        "market_info_resolution_criteria": "N/A",
    }
    return q, q_dict


# ---------------------------------------------------------------------------
# Prompt parity tests
# ---------------------------------------------------------------------------

class TestDatasetPromptParity:
    def test_dataset_prompt_matches_official(self) -> None:
        q, q_dict = _make_dataset_question()
        forecast_due_date = "2026-07-05"

        our_prompt = _build_prompt(q, source=q.source, resolution_dates=q.resolution_dates)
        official_prompt = _official_render_prompt(q_dict, forecast_due_date, forecast_due_date)

        assert our_prompt.strip() == official_prompt.strip()

    def test_dataset_prompt_with_placeholders_matches(self) -> None:
        text = "Will value have increased by {resolution_date} compared to {forecast_due_date}?"
        q, q_dict = _make_dataset_question(question_text=text)
        forecast_due_date = "2026-07-05"

        our_prompt = _build_prompt(q, source=q.source, resolution_dates=q.resolution_dates)
        official_prompt = _official_render_prompt(q_dict, forecast_due_date, forecast_due_date)

        assert our_prompt.strip() == official_prompt.strip()
        assert "{resolution_date}" not in our_prompt
        assert "{forecast_due_date}" not in our_prompt

    def test_dataset_prompt_no_placeholders_matches(self) -> None:
        text = "Will temperature exceed 30C?"
        q, q_dict = _make_dataset_question(question_text=text)
        forecast_due_date = "2026-07-05"

        our_prompt = _build_prompt(q, source=q.source, resolution_dates=q.resolution_dates)
        official_prompt = _official_render_prompt(q_dict, forecast_due_date, forecast_due_date)

        assert our_prompt.strip() == official_prompt.strip()

    def test_resolution_dates_rendered_as_list(self) -> None:
        q, _ = _make_dataset_question()
        prompt = _build_prompt(q, source=q.source, resolution_dates=q.resolution_dates)
        assert "['2026-07-12', '2026-08-04', '2026-10-03']" in prompt

    def test_today_date_uses_forecast_due_date(self) -> None:
        q, _ = _make_dataset_question(forecast_due_date="2026-07-05")
        prompt = _build_prompt(q, source=q.source, resolution_dates=q.resolution_dates)
        assert "Today's Date: 2026-07-05" in prompt

    def test_all_dataset_sources_route_to_dataset_prompt(self) -> None:
        for source in OFFICIAL_DATASET_SOURCE_NAMES:
            q, q_dict = _make_dataset_question()
            q = q.model_copy(update={"source": source})
            q_dict["source"] = source
            forecast_due_date = "2026-07-05"

            our_prompt = _build_prompt(q, source=q.source, resolution_dates=q.resolution_dates)
            official_prompt = _official_render_prompt(q_dict, forecast_due_date, forecast_due_date)

            assert our_prompt.strip() == official_prompt.strip(), f"Mismatch for source: {source}"


class TestMarketPromptParity:
    def test_market_prompt_matches_official(self) -> None:
        q, q_dict = _make_market_question()
        forecast_due_date = "2026-07-05"

        our_prompt = _build_prompt(q, source=q.source)
        official_prompt = _official_render_prompt(q_dict, forecast_due_date, forecast_due_date)

        assert our_prompt.strip() == official_prompt.strip()

    def test_market_sources_route_to_market_prompt(self) -> None:
        for source in ["metaculus", "polymarket", "manifold", "infer"]:
            q, q_dict = _make_market_question()
            q = q.model_copy(update={"source": source})
            q_dict["source"] = source
            forecast_due_date = "2026-07-05"

            our_prompt = _build_prompt(q, source=q.source)
            official_prompt = _official_render_prompt(q_dict, forecast_due_date, forecast_due_date)

            assert our_prompt.strip() == official_prompt.strip(), f"Mismatch for source: {source}"


# ---------------------------------------------------------------------------
# Parsing parity tests
# ---------------------------------------------------------------------------

class TestParsingParity:
    """Our parsing should match official behavior on the same inputs."""

    @pytest.mark.parametrize("text,expected", [
        ("*0.73*", 0.73),
        ("0.65", 0.65),
        ("*0.5*", 0.5),
        ("*.85*", 0.85),
        ("*1.0*", 1.0),
        ("*0*", 0.0),
        ("0.001", 0.001),
    ])
    def test_fullmatch_cases_match_official(self, text: str, expected: float) -> None:
        official = _official_extract_probability(text)
        try:
            ours = _parse_probability(text)
        except ValueError:
            ours = None
        assert ours == pytest.approx(official), f"Mismatch on {text!r}: ours={ours} official={official}"

    @pytest.mark.parametrize("text,n,expected", [
        ("*0.30* *0.45* *0.60*", 3, [0.30, 0.45, 0.60]),
        ("*0.10* *0.90*", 2, [0.10, 0.90]),
        ("*0.5*", 1, [0.5]),
    ])
    def test_multi_extract_matches_official(self, text: str, n: int, expected: list[float]) -> None:
        official = _official_extract_probabilities(text)
        try:
            ours = _parse_probabilities(text, n)
        except ValueError:
            ours = []
        if len(official) == n:
            assert len(ours) == n
            for o, e in zip(ours, official):
                assert o == pytest.approx(e)

    def test_no_clamping_matches_official(self) -> None:
        official = _official_extract_probability("0.001")
        try:
            ours = _parse_probability("0.001")
        except ValueError:
            ours = None
        assert ours == pytest.approx(official)
        assert ours == pytest.approx(0.001)

    def test_parse_failure_raises_like_official(self) -> None:
        official = _official_extract_probability("I cannot determine")
        assert official is None
        with pytest.raises(ValueError):
            _parse_probability("I cannot determine")


# ---------------------------------------------------------------------------
# Scoring parity tests
# ---------------------------------------------------------------------------

class TestScoringParity:
    def test_brier_index_formula(self) -> None:
        assert brier_index(0.25) == pytest.approx(50.0)
        assert brier_index(0.0) == pytest.approx(100.0)
        assert brier_index(1.0) == pytest.approx(0.0)

    def test_always_half_gets_brier_025(self) -> None:
        pairs = [(0.5, 1), (0.5, 0), (0.5, 1), (0.5, 0)]
        mean_bs = sum(brier_score(f, o) for f, o in pairs) / len(pairs)
        assert mean_bs == pytest.approx(0.25)

    def test_brier_score_formula(self) -> None:
        assert brier_score(0.7, 1) == pytest.approx(0.09)
        assert brier_score(0.3, 0) == pytest.approx(0.09)
        assert brier_score(0.5, 0) == pytest.approx(0.25)
        assert brier_score(0.5, 1) == pytest.approx(0.25)
        assert brier_score(1.0, 1) == pytest.approx(0.0)
        assert brier_score(0.0, 0) == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Question text formatting parity
# ---------------------------------------------------------------------------

class TestQuestionFormattingParity:
    def test_dataset_format_matches_official(self) -> None:
        text = "Will value increase by {resolution_date} vs {forecast_due_date}?"
        forecast_due_date = "2026-07-05"

        official = text.format(
            forecast_due_date=forecast_due_date,
            resolution_date="each of the resolution dates provided below",
        )
        ours = _format_question_text(text, forecast_due_date, is_dataset=True)
        assert ours == official

    def test_market_not_formatted(self) -> None:
        text = "Will {resolution_date} event happen?"
        ours = _format_question_text(text, "2026-07-05", is_dataset=False)
        assert ours == text

    def test_no_placeholders_passthrough(self) -> None:
        text = "Will temperature exceed 30C?"
        ours = _format_question_text(text, "2026-07-05", is_dataset=True)
        assert ours == text
