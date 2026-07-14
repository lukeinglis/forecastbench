"""Tests for baseline LLM forecaster (mocked litellm)."""

from __future__ import annotations

from unittest.mock import MagicMock, patch, AsyncMock

import pytest

from fetch_data import Question
from baseline_agent import _build_prompt, _parse_probability, MODEL


def _make_question(freeze: str | None = "2024-06-15") -> Question:
    return Question(
        id="q1",
        source="metaculus",
        question="Will X happen?",
        background="Some background",
        resolution_criteria="Resolves YES if X.",
        freeze_datetime=freeze,
    )


def _mock_response(content: str) -> MagicMock:
    msg = MagicMock()
    msg.content = content
    choice = MagicMock()
    choice.message = msg
    resp = MagicMock()
    resp.choices = [choice]
    return resp


class TestBuildPrompt:
    def test_prompt_contains_question_text(self) -> None:
        q = _make_question()
        prompt = _build_prompt(q)
        assert "Will X happen?" in prompt

    def test_prompt_contains_background(self) -> None:
        q = _make_question()
        prompt = _build_prompt(q)
        assert "Some background" in prompt

    def test_prompt_contains_resolution_criteria(self) -> None:
        q = _make_question()
        prompt = _build_prompt(q)
        assert "Resolves YES if X." in prompt

    def test_prompt_contains_temporal_context(self) -> None:
        q = _make_question(freeze="2024-06-15")
        prompt = _build_prompt(q)
        assert "2024-06-15" in prompt

    def test_prompt_without_freeze_datetime(self) -> None:
        q = _make_question(freeze=None)
        prompt = _build_prompt(q)
        assert "Will X happen?" in prompt

    def test_prompt_contains_superforecaster_persona(self) -> None:
        q = _make_question()
        prompt = _build_prompt(q)
        assert "superforecaster" in prompt.lower()


class TestParseProbability:
    def test_extracts_decimal(self) -> None:
        assert _parse_probability("I estimate 0.73") == pytest.approx(0.73)

    def test_extracts_from_probability_line(self) -> None:
        text = "After analysis...\n\nProbability: 0.65"
        assert _parse_probability(text) == pytest.approx(0.65)

    def test_extracts_leading_zero_optional(self) -> None:
        assert _parse_probability("The answer is .85") == pytest.approx(0.85)

    def test_clamps_to_lower_bound(self) -> None:
        assert _parse_probability("Probability: 0.001") == pytest.approx(0.01)

    def test_clamps_to_upper_bound(self) -> None:
        assert _parse_probability("Probability: 0.999") == pytest.approx(0.99)

    def test_fallback_on_no_number(self) -> None:
        assert _parse_probability("I cannot determine") == pytest.approx(0.5)

    def test_fallback_on_empty(self) -> None:
        assert _parse_probability("") == pytest.approx(0.5)

    def test_extracts_from_verbose_response(self) -> None:
        text = """Let me think about this step by step.
        Base rate is around 30%. Adjusting for factors...
        My final estimate is 0.42."""
        result = _parse_probability(text)
        assert 0.01 <= result <= 0.99

    def test_extracts_zero(self) -> None:
        result = _parse_probability("Probability: 0")
        assert result == pytest.approx(0.01)

    def test_extracts_one(self) -> None:
        result = _parse_probability("Probability: 1.0")
        assert result == pytest.approx(0.99)


class TestParseProbabilityPriority:
    """Tests for priority regex matching explicit probability markers."""

    def test_explicit_probability_is(self) -> None:
        assert _parse_probability("My probability is 0.75") == pytest.approx(0.75)

    def test_explicit_probability_over_version_decimal(self) -> None:
        text = "Running v0.95 of the model, probability: 0.3"
        assert _parse_probability(text) == pytest.approx(0.3)

    def test_no_number_fallback(self) -> None:
        assert _parse_probability("no number here") == pytest.approx(0.5)

    def test_probability_equals(self) -> None:
        assert _parse_probability("Probability = 0.8") == pytest.approx(0.8)

    def test_probability_colon_no_space(self) -> None:
        assert _parse_probability("probability:0.42") == pytest.approx(0.42)


class TestForecastSync:
    @patch("baseline_agent.litellm")
    def test_calls_litellm_completion(self, mock_litellm: MagicMock) -> None:
        mock_litellm.completion.return_value = _mock_response("Probability: 0.73")
        from baseline_agent import forecast

        q = _make_question()
        result = forecast(q)

        mock_litellm.completion.assert_called_once()
        call_kwargs = mock_litellm.completion.call_args
        assert call_kwargs.kwargs["temperature"] == 0.3
        assert call_kwargs.kwargs["timeout"] == 60
        assert result == pytest.approx(0.73)


class TestForecastAsync:
    @patch("baseline_agent.litellm")
    async def test_calls_litellm_acompletion(self, mock_litellm: MagicMock) -> None:
        mock_litellm.acompletion = AsyncMock(return_value=_mock_response("Probability: 0.65"))
        from baseline_agent import aforecast

        q = _make_question()
        result = await aforecast(q)

        mock_litellm.acompletion.assert_called_once()
        assert result == pytest.approx(0.65)


class TestModelConfig:
    def test_default_model(self) -> None:
        assert "claude" in MODEL.lower() or "sonnet" in MODEL.lower()

    @patch.dict("os.environ", {"FORECAST_MODEL": "gpt-4o"})
    def test_model_configurable_via_env(self) -> None:
        import importlib
        import baseline_agent

        importlib.reload(baseline_agent)
        assert baseline_agent.MODEL == "gpt-4o"
        importlib.reload(baseline_agent)
