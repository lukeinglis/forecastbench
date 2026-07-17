"""Tests for baseline LLM forecaster (mocked litellm)."""

from __future__ import annotations

from unittest.mock import MagicMock, patch, AsyncMock

import pytest

from fetch_data import Question
from baseline_agent import _build_prompt, _parse_probability, MODEL


def _make_question(
    freeze: str | None = "2024-06-15",
    source: str = "metaculus",
    freeze_datetime_value: float | None = None,
    freeze_datetime_value_explanation: str | None = None,
    resolution_dates: list[str] | None = None,
    source_intro: str | None = None,
) -> Question:
    return Question(
        id="q1",
        source=source,
        question="Will X happen?",
        background="Some background",
        resolution_criteria="Resolves YES if X.",
        freeze_datetime=freeze,
        freeze_datetime_value=freeze_datetime_value,
        freeze_datetime_value_explanation=freeze_datetime_value_explanation,
        resolution_dates=resolution_dates,
        source_intro=source_intro,
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


class TestForecastDueDateInPrompt:
    def test_prompt_uses_forecast_due_date_as_todays_date(self) -> None:
        q = _make_question(freeze="2024-06-15")
        q = q.model_copy(update={"forecast_due_date": "2024-06-25"})
        prompt = _build_prompt(q)
        assert "2024-06-25" in prompt
        assert "2024-06-15" not in prompt

    def test_prompt_falls_back_to_freeze_datetime_without_forecast_due_date(self) -> None:
        q = _make_question(freeze="2024-06-15")
        prompt = _build_prompt(q)
        assert "2024-06-15" in prompt

    def test_prompt_uses_todays_date_label(self) -> None:
        q = _make_question(freeze="2024-06-15")
        q = q.model_copy(update={"forecast_due_date": "2024-06-25"})
        prompt = _build_prompt(q)
        assert "Today's Date:" in prompt


class TestPromptVariants:
    def test_zero_shot_default_uses_asterisk_format(self) -> None:
        q = _make_question()
        prompt = _build_prompt(q, prompt_variant="zero-shot")
        assert "asterisk" in prompt.lower()

    def test_zero_shot_fv_market_includes_freeze_value(self) -> None:
        q = _make_question(
            source="metaculus",
            freeze="2024-06-15",
            freeze_datetime_value=0.73,
        )
        prompt = _build_prompt(q, prompt_variant="zero-shot-fv")
        assert "Market value on 2024-06-15" in prompt
        assert "0.73" in prompt
        assert "asterisk" in prompt.lower()

    def test_zero_shot_fv_without_freeze_value_falls_back(self) -> None:
        q = _make_question(source="metaculus")
        prompt = _build_prompt(q, prompt_variant="zero-shot-fv")
        assert "Market value on" not in prompt

    def test_dataset_prompt_includes_resolution_dates(self) -> None:
        q = _make_question(
            source="fred",
            freeze="2024-06-15",
            freeze_datetime_value=3.5,
            freeze_datetime_value_explanation="Some explanation",
            resolution_dates=["2024-07-01", "2024-08-01", "2024-09-01"],
        )
        prompt = _build_prompt(q, prompt_variant="dataset")
        assert "resolution dates" in prompt.lower()
        assert "2024-07-01" in prompt
        assert "2024-08-01" in prompt
        assert "2024-09-01" in prompt
        assert "3.5" in prompt

    def test_dataset_prompt_includes_freeze_value(self) -> None:
        q = _make_question(
            source="acled",
            freeze="2024-06-15",
            freeze_datetime_value=42.0,
            freeze_datetime_value_explanation="count",
        )
        prompt = _build_prompt(q, prompt_variant="dataset")
        assert "42.0" in prompt

    def test_dataset_prompt_for_market_source_falls_back_to_zero_shot(self) -> None:
        q = _make_question(source="metaculus")
        prompt = _build_prompt(q, prompt_variant="dataset")
        assert "asterisk" in prompt.lower()

    def test_source_parameter_overrides_question_source(self) -> None:
        q = _make_question(
            source="metaculus",
            freeze="2024-06-15",
            freeze_datetime_value=0.5,
        )
        prompt = _build_prompt(q, prompt_variant="zero-shot-fv", source="fred")
        assert "Market value on" not in prompt

    def test_resolution_dates_parameter_overrides_question(self) -> None:
        q = _make_question(
            source="fred",
            freeze="2024-06-15",
            freeze_datetime_value=3.5,
            freeze_datetime_value_explanation="rate",
            resolution_dates=["2024-07-01"],
        )
        prompt = _build_prompt(
            q,
            prompt_variant="dataset",
            resolution_dates=["2025-01-01", "2025-06-01"],
        )
        assert "2025-01-01" in prompt
        assert "2025-06-01" in prompt


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


class TestAsteriskParsing:
    def test_asterisk_basic(self) -> None:
        assert _parse_probability("*0.75*") == pytest.approx(0.75)

    def test_asterisk_with_text(self) -> None:
        text = "Based on my analysis, my estimate is *0.42* for this question."
        assert _parse_probability(text) == pytest.approx(0.42)

    def test_asterisk_with_spaces(self) -> None:
        assert _parse_probability("* 0.65 *") == pytest.approx(0.65)

    def test_asterisk_clamped_low(self) -> None:
        assert _parse_probability("*0.001*") == pytest.approx(0.01)

    def test_asterisk_clamped_high(self) -> None:
        assert _parse_probability("*0.999*") == pytest.approx(0.99)

    def test_asterisk_priority_over_probability_line(self) -> None:
        text = "Probability: 0.30\n\n*0.75*"
        assert _parse_probability(text) == pytest.approx(0.75)

    def test_asterisk_zero(self) -> None:
        assert _parse_probability("*0*") == pytest.approx(0.01)

    def test_asterisk_one(self) -> None:
        assert _parse_probability("*1.0*") == pytest.approx(0.99)

    def test_asterisk_no_leading_zero(self) -> None:
        assert _parse_probability("*.85*") == pytest.approx(0.85)


class TestParseProbabilityPriority:
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

    @patch("baseline_agent.litellm")
    def test_passes_prompt_variant(self, mock_litellm: MagicMock) -> None:
        mock_litellm.completion.return_value = _mock_response("*0.55*")
        from baseline_agent import forecast

        q = _make_question(
            source="metaculus",
            freeze="2024-06-15",
            freeze_datetime_value=0.7,
        )
        result = forecast(q, prompt_variant="zero-shot-fv")

        assert result == pytest.approx(0.55)
        call_args = mock_litellm.completion.call_args
        prompt_content = call_args.kwargs["messages"][0]["content"]
        assert "asterisk" in prompt_content.lower()


class TestForecastAsync:
    @patch("baseline_agent.litellm")
    async def test_calls_litellm_acompletion(self, mock_litellm: MagicMock) -> None:
        mock_litellm.acompletion = AsyncMock(return_value=_mock_response("Probability: 0.65"))
        from baseline_agent import aforecast

        q = _make_question()
        result = await aforecast(q)

        mock_litellm.acompletion.assert_called_once()
        assert result == pytest.approx(0.65)

    @patch("baseline_agent.litellm")
    async def test_passes_prompt_variant_async(self, mock_litellm: MagicMock) -> None:
        mock_litellm.acompletion = AsyncMock(return_value=_mock_response("*0.42*"))
        from baseline_agent import aforecast

        q = _make_question(
            source="fred",
            freeze="2024-06-15",
            freeze_datetime_value=3.5,
            resolution_dates=["2024-07-01"],
        )
        result = await aforecast(q, prompt_variant="dataset", source="fred")

        assert result == pytest.approx(0.42)


class TestMarketInfoResolutionCriteria:
    def test_appended_when_present_and_not_na(self) -> None:
        q = _make_question()
        q = q.model_copy(update={"market_info_resolution_criteria": "Resolves based on official data."})
        prompt = _build_prompt(q)
        assert "Resolves based on official data." in prompt

    def test_not_appended_when_na(self) -> None:
        q = _make_question()
        q = q.model_copy(update={"market_info_resolution_criteria": "N/A"})
        prompt = _build_prompt(q)
        assert prompt.count("N/A") == 0 or "market_info_resolution_criteria" not in prompt

    def test_not_appended_when_none(self) -> None:
        q = _make_question()
        q = q.model_copy(update={"market_info_resolution_criteria": None})
        prompt = _build_prompt(q)
        assert "market_info_resolution_criteria" not in prompt


class TestMarketCloseAsResolutionDate:
    def test_market_close_used_when_no_resolution_date(self) -> None:
        q = _make_question(source="metaculus")
        q = q.model_copy(update={"market_info_close_datetime": "2024-12-31T23:59:00Z"})
        prompt = _build_prompt(q, resolution_date=None)
        assert "2024-12-31T23:59:00Z" in prompt
        assert "resolution date" in prompt.lower()

    def test_explicit_resolution_date_takes_precedence(self) -> None:
        q = _make_question(source="metaculus")
        q = q.model_copy(update={"market_info_close_datetime": "2024-12-31T23:59:00Z"})
        prompt = _build_prompt(q, resolution_date="2024-11-01")
        assert "2024-11-01" in prompt
        assert "2024-12-31T23:59:00Z" not in prompt

    def test_not_used_for_dataset_sources(self) -> None:
        q = _make_question(source="fred")
        q = q.model_copy(update={"market_info_close_datetime": "2024-12-31T23:59:00Z"})
        prompt = _build_prompt(q, resolution_date=None)
        assert "2024-12-31T23:59:00Z" not in prompt


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
