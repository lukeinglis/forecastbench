"""Tests for baseline LLM forecaster (mocked litellm)."""

from __future__ import annotations

from unittest.mock import MagicMock, patch, AsyncMock

import pytest

from fetch_data import Question
from baseline_agent import (
    _build_prompt,
    _build_dataset_prompt,
    _parse_probability,
    _parse_probabilities,
    MODEL,
)


def _make_question(
    freeze: str | None = "2024-06-15",
    forecast_due_date: str | None = None,
) -> Question:
    return Question(
        id="q1",
        source="metaculus",
        question="Will X happen?",
        background="Some background",
        resolution_criteria="Resolves YES if X.",
        freeze_datetime=freeze,
        forecast_due_date=forecast_due_date,
    )


def _make_dataset_question(
    freeze: str | None = "2024-06-05",
    forecast_due_date: str | None = "2024-06-15",
    freeze_value: float | None = 42.5,
    freeze_value_explanation: str | None = "Current GDP index",
) -> Question:
    return Question(
        id="dq1",
        source="acled",
        question="Will GDP exceed threshold?",
        background="Economic data question",
        resolution_criteria="Resolves YES if GDP > 50.",
        freeze_datetime=freeze,
        forecast_due_date=forecast_due_date,
        freeze_datetime_value=freeze_value,
        freeze_datetime_value_explanation=freeze_value_explanation,
        resolution_dates=["2024-07-01", "2024-08-01", "2024-09-01"],
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

    def test_today_date_uses_forecast_due_date_when_present(self) -> None:
        q = _make_question(freeze="2024-06-05", forecast_due_date="2024-06-15")
        prompt = _build_prompt(q)
        assert "Today's Date: 2024-06-15" in prompt
        assert "information available as of 2024-06-05" in prompt

    def test_today_date_falls_back_to_freeze_datetime(self) -> None:
        q = _make_question(freeze="2024-06-05", forecast_due_date=None)
        prompt = _build_prompt(q)
        assert "Today's Date: 2024-06-05" in prompt


class TestBuildDatasetPrompt:
    def test_includes_all_resolution_dates(self) -> None:
        q = _make_dataset_question()
        prompt = _build_dataset_prompt(q, ["2024-07-01", "2024-08-01", "2024-09-01"])
        assert "2024-07-01" in prompt
        assert "2024-08-01" in prompt
        assert "2024-09-01" in prompt

    def test_includes_asterisk_format_instruction(self) -> None:
        q = _make_dataset_question()
        prompt = _build_dataset_prompt(q, ["2024-07-01"])
        assert "asterisk" in prompt.lower()
        assert "*p*" in prompt

    def test_uses_forecast_due_date_for_today(self) -> None:
        q = _make_dataset_question(freeze="2024-06-05", forecast_due_date="2024-06-15")
        prompt = _build_dataset_prompt(q, ["2024-07-01"])
        assert "Today's Date: 2024-06-15" in prompt

    def test_includes_freeze_value(self) -> None:
        q = _make_dataset_question(freeze_value=42.5)
        prompt = _build_dataset_prompt(q, ["2024-07-01"])
        assert "42.5" in prompt

    def test_includes_freeze_value_explanation(self) -> None:
        q = _make_dataset_question(freeze_value_explanation="Current GDP index")
        prompt = _build_dataset_prompt(q, ["2024-07-01"])
        assert "Current GDP index" in prompt

    def test_omits_freeze_value_when_none(self) -> None:
        q = _make_dataset_question(freeze_value=None, freeze_value_explanation=None)
        prompt = _build_dataset_prompt(q, ["2024-07-01"])
        assert "Current value on" not in prompt

    def test_includes_data_availability_context(self) -> None:
        q = _make_dataset_question(freeze="2024-06-05")
        prompt = _build_dataset_prompt(q, ["2024-07-01"])
        assert "information available as of 2024-06-05" in prompt


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


class TestParseProbabilities:
    def test_extracts_asterisk_wrapped(self) -> None:
        text = "*0.65* *0.70* *0.80*"
        result = _parse_probabilities(text, 3)
        assert result == [pytest.approx(0.65), pytest.approx(0.70), pytest.approx(0.80)]

    @patch("baseline_agent.litellm")
    def test_plain_decimals_trigger_llm_fallback(self, mock_litellm: MagicMock) -> None:
        mock_litellm.completion.return_value = _mock_response("[0.65, 0.70, 0.80]")
        text = "0.65 0.70 0.80"
        result = _parse_probabilities(text, 3)
        mock_litellm.completion.assert_called_once()
        assert result == [pytest.approx(0.65), pytest.approx(0.70), pytest.approx(0.80)]

    def test_extracts_mixed_formats(self) -> None:
        text = "*0.30*\n*0.45*\n*0.60*\n*0.75*"
        result = _parse_probabilities(text, 4)
        assert len(result) == 4
        assert result[0] == pytest.approx(0.30)
        assert result[3] == pytest.approx(0.75)

    def test_clamps_all_values(self) -> None:
        text = "*0.001* *1.0* *0.50*"
        result = _parse_probabilities(text, 3)
        assert result[0] == pytest.approx(0.01)
        assert result[1] == pytest.approx(0.99)
        assert result[2] == pytest.approx(0.50)

    @patch("baseline_agent.litellm")
    def test_wrong_count_triggers_llm_fallback(self, mock_litellm: MagicMock) -> None:
        mock_litellm.completion.return_value = _mock_response("[0.30, 0.45, 0.60]")
        text = "*0.65* *0.70*"
        result = _parse_probabilities(text, 3)
        mock_litellm.completion.assert_called_once()
        assert result == [pytest.approx(0.30), pytest.approx(0.45), pytest.approx(0.60)]

    @patch("baseline_agent.litellm")
    def test_both_fail_returns_defaults(self, mock_litellm: MagicMock) -> None:
        mock_litellm.completion.return_value = _mock_response("[]")
        text = "I cannot determine the probabilities"
        result = _parse_probabilities(text, 3)
        assert result == [0.5, 0.5, 0.5]

    @patch("baseline_agent.litellm")
    def test_llm_extraction_exception_returns_defaults(self, mock_litellm: MagicMock) -> None:
        mock_litellm.completion.side_effect = Exception("API error")
        text = "some text without numbers"
        result = _parse_probabilities(text, 2)
        assert result == [0.5, 0.5]

    def test_eight_horizons(self) -> None:
        text = "*0.10* *0.20* *0.30* *0.40* *0.50* *0.60* *0.70* *0.80*"
        result = _parse_probabilities(text, 8)
        assert len(result) == 8
        assert result[0] == pytest.approx(0.10)
        assert result[7] == pytest.approx(0.80)

    @patch("baseline_agent.litellm")
    def test_ignores_stray_numbers_in_reasoning(self, mock_litellm: MagicMock) -> None:
        mock_litellm.completion.return_value = _mock_response("[0.40, 0.60]")
        text = "The base rate is 0.30 and after adjusting by 0.10 I get *0.40*"
        result = _parse_probabilities(text, 2)
        mock_litellm.completion.assert_called_once()
        assert result == [pytest.approx(0.40), pytest.approx(0.60)]


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

    @patch("baseline_agent.litellm")
    def test_forecast_returns_float(self, mock_litellm: MagicMock) -> None:
        mock_litellm.completion.return_value = _mock_response("Probability: 0.55")
        from baseline_agent import forecast

        q = _make_question()
        result = forecast(q)
        assert isinstance(result, float)


class TestForecastMulti:
    @patch("baseline_agent.litellm")
    def test_returns_list_matching_dates(self, mock_litellm: MagicMock) -> None:
        mock_litellm.completion.return_value = _mock_response("*0.30* *0.45* *0.60*")
        from baseline_agent import forecast_multi

        q = _make_dataset_question()
        dates = ["2024-07-01", "2024-08-01", "2024-09-01"]
        result = forecast_multi(q, resolution_dates=dates)

        assert len(result) == 3
        assert all(isinstance(p, float) for p in result)
        assert result == [pytest.approx(0.30), pytest.approx(0.45), pytest.approx(0.60)]

    @patch("baseline_agent.litellm")
    def test_uses_dataset_prompt_template(self, mock_litellm: MagicMock) -> None:
        mock_litellm.completion.return_value = _mock_response("*0.50*")
        from baseline_agent import forecast_multi

        q = _make_dataset_question()
        forecast_multi(q, resolution_dates=["2024-07-01"])

        call_args = mock_litellm.completion.call_args
        prompt = call_args.kwargs["messages"][0]["content"]
        assert "asterisk" in prompt.lower()
        assert "resolution dates" in prompt.lower()


class TestForecastAsync:
    @patch("baseline_agent.litellm")
    async def test_calls_litellm_acompletion(self, mock_litellm: MagicMock) -> None:
        mock_litellm.acompletion = AsyncMock(return_value=_mock_response("Probability: 0.65"))
        from baseline_agent import aforecast

        q = _make_question()
        result = await aforecast(q)

        mock_litellm.acompletion.assert_called_once()
        assert result == pytest.approx(0.65)


class TestAforecastMulti:
    @patch("baseline_agent.litellm")
    async def test_returns_list_matching_dates(self, mock_litellm: MagicMock) -> None:
        mock_litellm.acompletion = AsyncMock(
            return_value=_mock_response("*0.25* *0.50* *0.75*")
        )
        from baseline_agent import aforecast_multi

        q = _make_dataset_question()
        dates = ["2024-07-01", "2024-08-01", "2024-09-01"]
        result = await aforecast_multi(q, resolution_dates=dates)

        assert len(result) == 3
        assert result == [pytest.approx(0.25), pytest.approx(0.50), pytest.approx(0.75)]


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
