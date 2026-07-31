"""Tests for baseline LLM forecaster (mocked litellm)."""

from __future__ import annotations

import os
import time
from unittest.mock import MagicMock, patch, AsyncMock

import pytest

from fetch_data import Question
from baseline_agent import (
    _apply_horizon_dampening,
    _apply_timeseries_dampening,
    _build_prompt,
    _build_dataset_prompt,
    _parse_probability,
    _parse_probabilities,
    MODEL,
    TIMESERIES_SOURCES,
    TEMPERATURE,
    MAX_TOKENS,
)


def _make_question(
    freeze: str | None = "2024-06-15",
    forecast_due_date: str | None = None,
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
        forecast_due_date=forecast_due_date,
        freeze_datetime_value=freeze_datetime_value,
        freeze_datetime_value_explanation=freeze_datetime_value_explanation,
        resolution_dates=resolution_dates,
        source_intro=source_intro,
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
        assert "2024-06-15" in prompt

    def test_today_date_falls_back_to_freeze_datetime(self) -> None:
        q = _make_question(freeze="2024-06-05", forecast_due_date=None)
        prompt = _build_prompt(q)
        assert "2024-06-05" in prompt


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


class TestDatasetAutoRouting:
    """Dataset questions auto-route to ZERO_SHOT_DATASET_PROMPT regardless of prompt_variant."""

    def test_dataset_source_auto_routes_to_dataset_prompt(self) -> None:
        q = _make_question(
            source="fred",
            freeze="2024-06-15",
            freeze_datetime_value=3.5,
            freeze_datetime_value_explanation="Current rate",
            resolution_dates=["2024-07-01", "2024-08-01"],
        )
        prompt = _build_prompt(q)
        assert "resolution dates" in prompt.lower()
        assert "freeze_datetime_value_explanation" not in prompt
        assert "Value Explanation:" in prompt
        assert "Current rate" in prompt

    def test_dataset_source_ignores_zero_shot_variant(self) -> None:
        q = _make_question(
            source="fred",
            freeze="2024-06-15",
            freeze_datetime_value=3.5,
            freeze_datetime_value_explanation="Current rate",
            resolution_dates=["2024-07-01", "2024-08-01"],
        )
        prompt = _build_prompt(q, prompt_variant="zero-shot")
        assert "resolution dates" in prompt.lower()
        assert "Value Explanation:" in prompt

    def test_market_source_still_uses_market_prompt(self) -> None:
        q = _make_question(source="metaculus")
        prompt = _build_prompt(q)
        assert "Question resolution date:" in prompt
        assert "Question resolution dates:" not in prompt

    def test_market_source_with_fv_uses_freeze_value_prompt(self) -> None:
        q = _make_question(
            source="metaculus",
            freeze="2024-06-15",
            freeze_datetime_value=0.73,
        )
        prompt = _build_prompt(q, prompt_variant="zero-shot-fv")
        assert "Market value on 2024-06-15" in prompt
        assert "0.73" in prompt

    def test_format_question_text_called_for_dataset(self) -> None:
        q = _make_question(
            source="fred",
            freeze="2024-06-15",
            freeze_datetime_value=3.5,
            freeze_datetime_value_explanation="val",
            resolution_dates=["2024-07-01"],
            forecast_due_date="2024-06-15",
        )
        q = q.model_copy(update={
            "question": "Will GDP exceed {forecast_due_date} target by {resolution_date}?",
        })
        prompt = _build_prompt(q)
        assert "2024-06-15" in prompt
        assert "each of the resolution dates provided below" in prompt
        assert "{forecast_due_date}" not in prompt
        assert "{resolution_date}" not in prompt

    def test_all_dataset_sources_auto_route(self) -> None:
        for source in ["fred", "acled", "dbnomics", "wikipedia", "yfinance"]:
            q = _make_question(
                source=source,
                freeze="2024-06-15",
                freeze_datetime_value=1.0,
                freeze_datetime_value_explanation="val",
                resolution_dates=["2024-07-01"],
            )
            prompt = _build_prompt(q)
            assert "resolution dates" in prompt.lower(), f"Source {source} did not auto-route to dataset prompt"


class TestSingleDatePrompt:
    """Per-date mode uses SINGLE_DATE_DATASET_PROMPT with only the target date."""

    def test_per_date_prompt_shows_only_target_date(self) -> None:
        q = _make_question(
            source="fred",
            freeze="2024-06-15",
            freeze_datetime_value=3.5,
            freeze_datetime_value_explanation="Current rate",
            resolution_dates=["2024-07-01", "2024-08-01", "2024-09-01"],
        )
        prompt = _build_prompt(q, resolution_date="2024-07-01")
        assert "2024-07-01" in prompt
        assert "2024-08-01" not in prompt
        assert "2024-09-01" not in prompt
        assert "Question resolution date:" in prompt
        assert "Question resolution dates:" not in prompt

    def test_no_resolution_date_still_shows_all_dates(self) -> None:
        q = _make_question(
            source="fred",
            freeze="2024-06-15",
            freeze_datetime_value=3.5,
            freeze_datetime_value_explanation="Current rate",
            resolution_dates=["2024-07-01", "2024-08-01", "2024-09-01"],
        )
        prompt = _build_prompt(q, resolution_date=None)
        assert "2024-07-01" in prompt
        assert "2024-08-01" in prompt
        assert "2024-09-01" in prompt

    def test_per_date_singular_output_instruction(self) -> None:
        q = _make_question(
            source="fred",
            freeze="2024-06-15",
            freeze_datetime_value=3.5,
            freeze_datetime_value_explanation="Current rate",
            resolution_dates=["2024-07-01", "2024-08-01"],
        )
        prompt = _build_prompt(q, resolution_date="2024-07-01")
        assert "for each resolution date" not in prompt.lower()
        assert "asterisk" in prompt.lower()

    def test_per_date_applies_regardless_of_prompt_variant(self) -> None:
        q = _make_question(
            source="acled",
            freeze="2024-06-15",
            freeze_datetime_value=10.0,
            freeze_datetime_value_explanation="count",
            resolution_dates=["2024-07-01", "2024-08-01"],
        )
        for variant in ("default", "zero-shot", "zero-shot-fv", "dataset"):
            prompt = _build_prompt(q, resolution_date="2024-07-01", prompt_variant=variant)
            assert "2024-08-01" not in prompt, f"variant={variant} leaked other dates"
            assert "2024-07-01" in prompt, f"variant={variant} missing target date"

    def test_per_date_not_used_for_market_sources(self) -> None:
        q = _make_question(source="metaculus")
        prompt = _build_prompt(q, resolution_date="2024-07-01")
        assert "Question resolution date:" in prompt
        assert "Value Explanation:" not in prompt


class TestParseProbability:
    def test_extracts_decimal(self) -> None:
        assert _parse_probability("I estimate 0.73") == pytest.approx(0.73)

    def test_extracts_from_probability_line(self) -> None:
        text = "After analysis...\n\nProbability: 0.65"
        assert _parse_probability(text) == pytest.approx(0.65)

    def test_extracts_leading_zero_optional(self) -> None:
        assert _parse_probability("The answer is .85") == pytest.approx(0.85)

    def test_no_clamping_low(self) -> None:
        assert _parse_probability("Probability: 0.001") == pytest.approx(0.001)

    def test_no_clamping_high(self) -> None:
        assert _parse_probability("Probability: 0.999") == pytest.approx(0.999)

    def test_raises_on_no_number(self) -> None:
        with pytest.raises(ValueError):
            _parse_probability("I cannot determine")

    def test_raises_on_empty(self) -> None:
        with pytest.raises(ValueError):
            _parse_probability("")

    def test_extracts_from_verbose_response(self) -> None:
        text = """Let me think about this step by step.
        Base rate is around 30%. Adjusting for factors...
        My final estimate is 0.42."""
        result = _parse_probability(text)
        assert 0.0 <= result <= 1.0

    def test_extracts_zero(self) -> None:
        result = _parse_probability("Probability: 0")
        assert result == pytest.approx(0.0)

    def test_extracts_one(self) -> None:
        result = _parse_probability("Probability: 1.0")
        assert result == pytest.approx(1.0)

    def test_fullmatch_on_bare_number(self) -> None:
        assert _parse_probability("0.73") == pytest.approx(0.73)

    def test_fullmatch_on_asterisk_wrapped(self) -> None:
        assert _parse_probability("*0.73*") == pytest.approx(0.73)


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

    def test_no_clamping(self) -> None:
        text = "*0.001* *1.0* *0.50*"
        result = _parse_probabilities(text, 3)
        assert result[0] == pytest.approx(0.001)
        assert result[1] == pytest.approx(1.0)
        assert result[2] == pytest.approx(0.50)

    @patch("baseline_agent.litellm")
    def test_wrong_count_triggers_llm_fallback(self, mock_litellm: MagicMock) -> None:
        mock_litellm.completion.return_value = _mock_response("[0.30, 0.45, 0.60]")
        text = "*0.65* *0.70*"
        result = _parse_probabilities(text, 3)
        mock_litellm.completion.assert_called_once()
        assert result == [pytest.approx(0.30), pytest.approx(0.45), pytest.approx(0.60)]

    @patch("baseline_agent.litellm")
    def test_both_fail_raises_value_error(self, mock_litellm: MagicMock) -> None:
        mock_litellm.completion.return_value = _mock_response("[]")
        text = "I cannot determine the probabilities"
        with pytest.raises(ValueError):
            _parse_probabilities(text, 3)

    @patch("baseline_agent.litellm")
    def test_llm_extraction_exception_raises_value_error(self, mock_litellm: MagicMock) -> None:
        mock_litellm.completion.side_effect = Exception("API error")
        text = "some text without numbers"
        with pytest.raises(ValueError):
            _parse_probabilities(text, 2)

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


class TestAsteriskParsing:
    def test_asterisk_basic(self) -> None:
        assert _parse_probability("*0.75*") == pytest.approx(0.75)

    def test_asterisk_with_text(self) -> None:
        text = "Based on my analysis, my estimate is *0.42* for this question."
        assert _parse_probability(text) == pytest.approx(0.42)

    def test_asterisk_with_spaces(self) -> None:
        assert _parse_probability("* 0.65 *") == pytest.approx(0.65)

    def test_asterisk_no_clamping_low(self) -> None:
        assert _parse_probability("*0.001*") == pytest.approx(0.001)

    def test_asterisk_no_clamping_high(self) -> None:
        assert _parse_probability("*0.999*") == pytest.approx(0.999)

    def test_asterisk_priority_over_probability_line(self) -> None:
        text = "Probability: 0.30\n\n*0.75*"
        assert _parse_probability(text) == pytest.approx(0.75)

    def test_asterisk_zero(self) -> None:
        assert _parse_probability("*0*") == pytest.approx(0.0)

    def test_asterisk_one(self) -> None:
        assert _parse_probability("*1.0*") == pytest.approx(1.0)

    def test_asterisk_no_leading_zero(self) -> None:
        assert _parse_probability("*.85*") == pytest.approx(0.85)


class TestParseProbabilityPriority:
    def test_explicit_probability_is(self) -> None:
        assert _parse_probability("My probability is 0.75") == pytest.approx(0.75)

    def test_explicit_probability_over_version_decimal(self) -> None:
        text = "Running v0.95 of the model, probability: 0.3"
        assert _parse_probability(text) == pytest.approx(0.3)

    def test_no_number_raises(self) -> None:
        with pytest.raises(ValueError):
            _parse_probability("no number here")

    def test_probability_equals(self) -> None:
        assert _parse_probability("Probability = 0.8") == pytest.approx(0.8)

    def test_probability_colon_no_space(self) -> None:
        assert _parse_probability("probability:0.42") == pytest.approx(0.42)


class TestForecastSync:
    @patch("baseline_agent.litellm")
    def test_calls_litellm_completion(self, mock_litellm: MagicMock) -> None:
        mock_litellm.completion.return_value = _mock_response("Probability: 0.73")
        import baseline_agent
        from baseline_agent import forecast

        with patch.object(baseline_agent, "THINKING_ENABLED", False):
            q = _make_question()
            result = forecast(q)

            mock_litellm.completion.assert_called_once()
            call_kwargs = mock_litellm.completion.call_args
            assert call_kwargs.kwargs["temperature"] == 0.3
            assert call_kwargs.kwargs["timeout"] == 180
            assert result == pytest.approx(0.73)

    @patch("baseline_agent.litellm")
    def test_forecast_returns_float(self, mock_litellm: MagicMock) -> None:
        mock_litellm.completion.return_value = _mock_response("Probability: 0.55")
        from baseline_agent import forecast

        q = _make_question()
        result = forecast(q)
        assert isinstance(result, float)

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
    def test_default_model_uses_vertex_ai(self) -> None:
        assert MODEL.startswith("vertex_ai/")

    @patch.dict("os.environ", {"FORECAST_MODEL": "gpt-4o"})
    def test_model_configurable_via_env(self) -> None:
        import importlib
        import baseline_agent

        importlib.reload(baseline_agent)
        assert baseline_agent.MODEL == "gpt-4o"
        importlib.reload(baseline_agent)

    def test_default_temperature_is_zero(self) -> None:
        assert TEMPERATURE == 0

    def test_default_max_tokens_is_16384(self) -> None:
        assert MAX_TOKENS == 16384

    @patch.dict("os.environ", {"FORECAST_TEMPERATURE": "0.5"})
    def test_temperature_configurable_via_env(self) -> None:
        import importlib
        import baseline_agent

        importlib.reload(baseline_agent)
        assert baseline_agent.TEMPERATURE == 0.5
        importlib.reload(baseline_agent)

    @patch.dict("os.environ", {"FORECAST_MAX_TOKENS": "4096"})
    def test_max_tokens_configurable_via_env(self) -> None:
        import importlib
        import baseline_agent

        importlib.reload(baseline_agent)
        assert baseline_agent.MAX_TOKENS == 4096
        importlib.reload(baseline_agent)


class TestSelectModel:
    def test_returns_default_model_when_timeseries_model_empty(self) -> None:
        with patch.object(__import__("baseline_agent"), "TIMESERIES_MODEL", ""):
            from baseline_agent import _select_model
            assert _select_model("fred") == MODEL

    def test_returns_timeseries_model_for_fred(self) -> None:
        with patch.object(__import__("baseline_agent"), "TIMESERIES_MODEL", "vertex_ai/claude-opus-4-8@20250915"):
            from baseline_agent import _select_model
            assert _select_model("fred") == "vertex_ai/claude-opus-4-8@20250915"

    def test_returns_timeseries_model_for_dbnomics(self) -> None:
        with patch.object(__import__("baseline_agent"), "TIMESERIES_MODEL", "vertex_ai/claude-opus-4-8@20250915"):
            from baseline_agent import _select_model
            assert _select_model("dbnomics") == "vertex_ai/claude-opus-4-8@20250915"

    def test_returns_timeseries_model_for_yfinance(self) -> None:
        with patch.object(__import__("baseline_agent"), "TIMESERIES_MODEL", "vertex_ai/claude-opus-4-8@20250915"):
            from baseline_agent import _select_model
            assert _select_model("yfinance") == "vertex_ai/claude-opus-4-8@20250915"

    def test_returns_default_model_for_metaculus(self) -> None:
        with patch.object(__import__("baseline_agent"), "TIMESERIES_MODEL", "vertex_ai/claude-opus-4-8@20250915"):
            from baseline_agent import _select_model
            assert _select_model("metaculus") == MODEL

    def test_returns_default_model_for_none_source(self) -> None:
        with patch.object(__import__("baseline_agent"), "TIMESERIES_MODEL", "vertex_ai/claude-opus-4-8@20250915"):
            from baseline_agent import _select_model
            assert _select_model(None) == MODEL

    def test_case_insensitive_source(self) -> None:
        with patch.object(__import__("baseline_agent"), "TIMESERIES_MODEL", "vertex_ai/claude-opus-4-8@20250915"):
            from baseline_agent import _select_model
            assert _select_model("FRED") == "vertex_ai/claude-opus-4-8@20250915"

    def test_timeseries_sources_contains_expected(self) -> None:
        assert TIMESERIES_SOURCES == frozenset(["fred", "dbnomics", "yfinance"])

    @patch.dict("os.environ", {"FORECAST_TIMESERIES_MODEL": "openai/gpt-4o"})
    def test_env_var_is_read(self) -> None:
        import importlib
        import baseline_agent
        importlib.reload(baseline_agent)
        assert baseline_agent.TIMESERIES_MODEL == "openai/gpt-4o"
        importlib.reload(baseline_agent)

    @patch.dict("os.environ", {}, clear=False)
    def test_env_var_defaults_to_empty(self) -> None:
        import importlib
        import baseline_agent
        env = os.environ.copy()
        env.pop("FORECAST_TIMESERIES_MODEL", None)
        with patch.dict("os.environ", env, clear=True):
            importlib.reload(baseline_agent)
            assert baseline_agent.TIMESERIES_MODEL == ""
            importlib.reload(baseline_agent)


class TestVertexCredentialRefresh:
    def test_skips_refresh_when_token_valid(self) -> None:
        import baseline_agent
        old_expiry = baseline_agent._vertex_token_expiry
        old_creds = baseline_agent._vertex_credentials
        try:
            baseline_agent._vertex_token_expiry = time.monotonic() + 9999
            baseline_agent._vertex_credentials = MagicMock()
            with patch.object(baseline_agent, "MODEL", "vertex_ai/claude-sonnet-4@20250514"):
                baseline_agent._ensure_vertex_credentials()
            baseline_agent._vertex_credentials.refresh.assert_not_called()
        finally:
            baseline_agent._vertex_token_expiry = old_expiry
            baseline_agent._vertex_credentials = old_creds

    def test_skips_for_non_vertex_model(self) -> None:
        import baseline_agent
        old_expiry = baseline_agent._vertex_token_expiry
        try:
            baseline_agent._vertex_token_expiry = 0.0
            with patch.object(baseline_agent, "MODEL", "openai/gpt-4o"):
                baseline_agent._ensure_vertex_credentials()
        finally:
            baseline_agent._vertex_token_expiry = old_expiry

    def test_refreshes_expired_credentials(self) -> None:
        import baseline_agent
        old_expiry = baseline_agent._vertex_token_expiry
        old_creds = baseline_agent._vertex_credentials
        try:
            baseline_agent._vertex_credentials = None
            baseline_agent._vertex_token_expiry = 0.0

            mock_creds = MagicMock()
            mock_creds.expiry = None
            mock_auth = MagicMock()
            mock_auth.default.return_value = (mock_creds, "proj")
            mock_transport = MagicMock()

            with (
                patch.object(baseline_agent, "MODEL", "vertex_ai/claude-sonnet-4@20250514"),
                patch.object(baseline_agent, "_get_google_auth", return_value=(mock_auth, mock_transport)),
            ):
                baseline_agent._ensure_vertex_credentials()

            mock_creds.refresh.assert_called_once()
            assert baseline_agent._vertex_token_expiry > 0
        finally:
            baseline_agent._vertex_token_expiry = old_expiry
            baseline_agent._vertex_credentials = old_creds

    def test_refresh_failure_does_not_crash(self) -> None:
        import baseline_agent
        old_expiry = baseline_agent._vertex_token_expiry
        old_creds = baseline_agent._vertex_credentials
        try:
            baseline_agent._vertex_credentials = None
            baseline_agent._vertex_token_expiry = 0.0

            with (
                patch.object(baseline_agent, "MODEL", "vertex_ai/claude-sonnet-4@20250514"),
                patch.object(baseline_agent, "_get_google_auth", side_effect=Exception("no creds")),
            ):
                baseline_agent._ensure_vertex_credentials()

            assert baseline_agent._vertex_token_expiry == 0.0
        finally:
            baseline_agent._vertex_token_expiry = old_expiry
            baseline_agent._vertex_credentials = old_creds


class TestHorizonDampening:
    def test_near_dates_unchanged(self) -> None:
        probs = [0.8, 0.9]
        dates = ["2024-07-10", "2024-07-20"]
        result = _apply_horizon_dampening(probs, dates, "2024-07-01")
        assert result[0] == pytest.approx(0.8)
        assert result[1] == pytest.approx(0.9)

    def test_far_dates_regress_toward_half(self) -> None:
        probs = [0.8, 0.8]
        dates = ["2024-07-10", "2025-07-01"]
        result = _apply_horizon_dampening(probs, dates, "2024-07-01")
        assert result[0] == pytest.approx(0.8)
        assert result[1] == pytest.approx(0.5 + 0.3 * (0.8 - 0.5))

    def test_exactly_365_days_uses_min_factor(self) -> None:
        probs = [1.0]
        dates = ["2025-07-01"]
        result = _apply_horizon_dampening(probs, dates, "2024-07-01")
        assert result[0] == pytest.approx(0.5 + 0.3 * 0.5)

    def test_midrange_interpolates(self) -> None:
        probs = [0.8]
        dates = ["2024-12-29"]
        result = _apply_horizon_dampening(probs, dates, "2024-07-01")
        days = 181
        factor = 1.0 - 0.7 * (days - 30) / (365 - 30)
        assert result[0] == pytest.approx(0.5 + factor * 0.3)

    def test_invalid_forecast_due_date_returns_original(self) -> None:
        probs = [0.8]
        dates = ["2024-07-10"]
        result = _apply_horizon_dampening(probs, dates, "not-a-date")
        assert result == probs

    def test_invalid_resolution_date_keeps_original(self) -> None:
        probs = [0.8, 0.9]
        dates = ["bad-date", "2024-07-10"]
        result = _apply_horizon_dampening(probs, dates, "2024-07-01")
        assert result[0] == 0.8
        assert result[1] == pytest.approx(0.9)

    def test_prob_at_half_stays_at_half(self) -> None:
        probs = [0.5]
        dates = ["2025-07-01"]
        result = _apply_horizon_dampening(probs, dates, "2024-07-01")
        assert result[0] == pytest.approx(0.5)


class TestTimeseriesDampening:
    def test_half_confidence_shrinks_toward_half(self) -> None:
        with patch("baseline_agent.TIMESERIES_CONFIDENCE", 0.5):
            assert _apply_timeseries_dampening(0.8, "fred") == pytest.approx(0.65)
            assert _apply_timeseries_dampening(0.2, "fred") == pytest.approx(0.35)

    def test_zero_confidence_always_returns_half(self) -> None:
        with patch("baseline_agent.TIMESERIES_CONFIDENCE", 0.0):
            assert _apply_timeseries_dampening(0.8, "fred") == pytest.approx(0.5)
            assert _apply_timeseries_dampening(0.1, "dbnomics") == pytest.approx(0.5)

    def test_full_confidence_no_change(self) -> None:
        with patch("baseline_agent.TIMESERIES_CONFIDENCE", 1.0):
            assert _apply_timeseries_dampening(0.8, "yfinance") == pytest.approx(0.8)
            assert _apply_timeseries_dampening(0.2, "fred") == pytest.approx(0.2)

    def test_non_timeseries_source_unchanged(self) -> None:
        with patch("baseline_agent.TIMESERIES_CONFIDENCE", 0.0):
            assert _apply_timeseries_dampening(0.8, "metaculus") == pytest.approx(0.8)
            assert _apply_timeseries_dampening(0.2, "polymarket") == pytest.approx(0.2)

    def test_case_insensitive(self) -> None:
        with patch("baseline_agent.TIMESERIES_CONFIDENCE", 0.5):
            assert _apply_timeseries_dampening(0.8, "FRED") == pytest.approx(0.65)
            assert _apply_timeseries_dampening(0.8, "YFinance") == pytest.approx(0.65)

    def test_half_stays_at_half(self) -> None:
        with patch("baseline_agent.TIMESERIES_CONFIDENCE", 0.5):
            assert _apply_timeseries_dampening(0.5, "fred") == pytest.approx(0.5)


class TestBaseRateHint:
    _TEST_BASE_RATES = {"fred": 0.46, "dbnomics": 0.78, "yfinance": 0.43}

    def test_fred_prompt_contains_base_rate(self) -> None:
        q = _make_question(
            source="fred",
            freeze="2024-06-15",
            freeze_datetime_value=3.5,
            freeze_datetime_value_explanation="Current rate",
            resolution_dates=["2024-07-01"],
        )
        with patch("baseline_agent.BASE_RATE_HINT", True), \
             patch("baseline_agent.TIMESERIES_BASE_RATES", self._TEST_BASE_RATES):
            prompt = _build_prompt(q, source="fred")
        assert "46%" in prompt
        assert "Historical context" in prompt

    def test_dbnomics_shows_correct_percentage(self) -> None:
        q = _make_question(
            source="dbnomics",
            freeze="2024-06-15",
            freeze_datetime_value=100.0,
            freeze_datetime_value_explanation="Index value",
            resolution_dates=["2024-07-01"],
        )
        with patch("baseline_agent.BASE_RATE_HINT", True), \
             patch("baseline_agent.TIMESERIES_BASE_RATES", self._TEST_BASE_RATES):
            prompt = _build_prompt(q, source="dbnomics")
        assert "78%" in prompt

    def test_yfinance_shows_correct_percentage(self) -> None:
        q = _make_question(
            source="yfinance",
            freeze="2024-06-15",
            freeze_datetime_value=150.0,
            freeze_datetime_value_explanation="Stock price",
            resolution_dates=["2024-07-01"],
        )
        with patch("baseline_agent.BASE_RATE_HINT", True), \
             patch("baseline_agent.TIMESERIES_BASE_RATES", self._TEST_BASE_RATES):
            prompt = _build_prompt(q, source="yfinance")
        assert "43%" in prompt

    def test_market_source_has_no_base_rate(self) -> None:
        q = _make_question(source="metaculus")
        with patch("baseline_agent.BASE_RATE_HINT", True), \
             patch("baseline_agent.TIMESERIES_BASE_RATES", self._TEST_BASE_RATES):
            prompt = _build_prompt(q, source="metaculus")
        assert "Historical context" not in prompt

    def test_disabled_by_default(self) -> None:
        q = _make_question(
            source="fred",
            freeze="2024-06-15",
            freeze_datetime_value=3.5,
            freeze_datetime_value_explanation="Current rate",
            resolution_dates=["2024-07-01"],
        )
        prompt = _build_prompt(q, source="fred")
        assert "Historical context" not in prompt

    def test_disabled_by_env_var(self) -> None:
        q = _make_question(
            source="fred",
            freeze="2024-06-15",
            freeze_datetime_value=3.5,
            freeze_datetime_value_explanation="Current rate",
            resolution_dates=["2024-07-01"],
        )
        with patch("baseline_agent.BASE_RATE_HINT", False):
            prompt = _build_prompt(q, source="fred")
        assert "Historical context" not in prompt

    def test_non_timeseries_dataset_source_has_no_base_rate(self) -> None:
        q = _make_question(
            source="acled",
            freeze="2024-06-15",
            freeze_datetime_value=50.0,
            freeze_datetime_value_explanation="Count",
            resolution_dates=["2024-07-01"],
        )
        with patch("baseline_agent.BASE_RATE_HINT", True), \
             patch("baseline_agent.TIMESERIES_BASE_RATES", self._TEST_BASE_RATES):
            prompt = _build_prompt(q, source="acled")
        assert "Historical context" not in prompt

    def test_base_rate_inserted_before_output_instruction(self) -> None:
        q = _make_question(
            source="fred",
            freeze="2024-06-15",
            freeze_datetime_value=3.5,
            freeze_datetime_value_explanation="Current rate",
            resolution_dates=["2024-07-01"],
        )
        with patch("baseline_agent.BASE_RATE_HINT", True), \
             patch("baseline_agent.TIMESERIES_BASE_RATES", self._TEST_BASE_RATES):
            prompt = _build_prompt(q, source="fred")
        hist_idx = prompt.index("Historical context")
        output_idx = prompt.index("Output your answer")
        assert hist_idx < output_idx

    def test_load_base_rates_returns_empty_by_default(self) -> None:
        from baseline_agent import _load_base_rates
        with patch.dict("os.environ", {}, clear=False):
            env = os.environ.copy()
            env.pop("FORECAST_BASE_RATES", None)
            with patch.dict("os.environ", env, clear=True):
                assert _load_base_rates() == {}

    def test_load_base_rates_uses_env_override(self) -> None:
        from baseline_agent import _load_base_rates
        with patch.dict("os.environ", {"FORECAST_BASE_RATES": '{"fred": 0.5}'}):
            assert _load_base_rates() == {"fred": 0.5}


class TestTimeseriesThinking:
    """FORECAST_TIMESERIES_THINKING env var controls thinking for timeseries sources."""

    def test_timeseries_thinking_disabled_by_default(self) -> None:
        import baseline_agent
        from baseline_agent import _forecast_kwargs

        with patch.object(baseline_agent, "THINKING_ENABLED", True), \
             patch.object(baseline_agent, "TIMESERIES_THINKING", False):
            kwargs = _forecast_kwargs(
                [{"role": "user", "content": "test"}], source="fred",
            )
        assert "thinking" not in kwargs
        assert kwargs["temperature"] == 0.3

    def test_timeseries_thinking_enabled(self) -> None:
        import baseline_agent
        from baseline_agent import _forecast_kwargs

        with patch.object(baseline_agent, "THINKING_ENABLED", True), \
             patch.object(baseline_agent, "TIMESERIES_THINKING", True):
            kwargs = _forecast_kwargs(
                [{"role": "user", "content": "test"}], source="fred",
            )
        assert "thinking" in kwargs
        assert "temperature" not in kwargs

    def test_timeseries_thinking_requires_global(self) -> None:
        import baseline_agent
        from baseline_agent import _forecast_kwargs

        with patch.object(baseline_agent, "THINKING_ENABLED", False), \
             patch.object(baseline_agent, "TIMESERIES_THINKING", True):
            kwargs = _forecast_kwargs(
                [{"role": "user", "content": "test"}], source="fred",
            )
        assert "thinking" not in kwargs
        assert kwargs["temperature"] == 0.3

    def test_market_always_temperature(self) -> None:
        import baseline_agent
        from baseline_agent import _forecast_kwargs

        with patch.object(baseline_agent, "THINKING_ENABLED", True), \
             patch.object(baseline_agent, "TIMESERIES_THINKING", True):
            kwargs = _forecast_kwargs(
                [{"role": "user", "content": "test"}], source="metaculus",
            )
        assert "thinking" not in kwargs
        assert kwargs["temperature"] == 0.3


class TestForecastParityParams:
    """Verify forecast LLM calls use _forecast_kwargs (adaptive thinking / temperature=0.3 fallback)."""

    @patch("baseline_agent.litellm")
    def test_forecast_enables_thinking_for_event_sources(self, mock_litellm: MagicMock) -> None:
        mock_litellm.completion.return_value = _mock_response("*0.50*")
        import baseline_agent
        from baseline_agent import forecast

        with patch.object(baseline_agent, "THINKING_ENABLED", True):
            forecast(_make_question(source="acled"))
            kwargs = mock_litellm.completion.call_args.kwargs
            assert "thinking" in kwargs
            assert "temperature" not in kwargs

    @patch("baseline_agent.litellm")
    def test_forecast_disables_thinking_for_market_sources(self, mock_litellm: MagicMock) -> None:
        mock_litellm.completion.return_value = _mock_response("*0.50*")
        from baseline_agent import forecast

        forecast(_make_question(source="metaculus"))
        kwargs = mock_litellm.completion.call_args.kwargs
        assert "thinking" not in kwargs
        assert kwargs["temperature"] == 0.3

    @patch("baseline_agent.litellm")
    def test_forecast_uses_configured_max_tokens(self, mock_litellm: MagicMock) -> None:
        mock_litellm.completion.return_value = _mock_response("*0.50*")
        from baseline_agent import forecast, MAX_TOKENS

        forecast(_make_question())
        kwargs = mock_litellm.completion.call_args.kwargs
        assert kwargs["max_tokens"] == MAX_TOKENS

    @patch("baseline_agent.litellm")
    def test_forecast_multi_disables_thinking_for_timeseries(self, mock_litellm: MagicMock) -> None:
        mock_litellm.completion.return_value = _mock_response("*0.30* *0.50* *0.70*")
        from baseline_agent import forecast_multi

        q = Question(
            id="tsq1", source="fred", question="Will value exceed threshold?",
            freeze_datetime="2024-06-05", forecast_due_date="2024-06-15",
            freeze_datetime_value=42.5,
            resolution_dates=["2024-07-01", "2024-08-01", "2024-09-01"],
        )
        forecast_multi(q, ["2024-07-01", "2024-08-01", "2024-09-01"])
        kwargs = mock_litellm.completion.call_args.kwargs
        assert "thinking" not in kwargs
        assert kwargs["temperature"] == 0.3

    @patch("baseline_agent.litellm")
    def test_forecast_multi_uses_configured_max_tokens(self, mock_litellm: MagicMock) -> None:
        mock_litellm.completion.return_value = _mock_response("*0.30* *0.50* *0.70*")
        from baseline_agent import forecast_multi, MAX_TOKENS

        forecast_multi(_make_dataset_question(), ["2024-07-01", "2024-08-01", "2024-09-01"])
        kwargs = mock_litellm.completion.call_args.kwargs
        assert kwargs["max_tokens"] == MAX_TOKENS

    @patch("baseline_agent.litellm")
    async def test_aforecast_enables_thinking_for_event_sources(self, mock_litellm: MagicMock) -> None:
        mock_litellm.acompletion = AsyncMock(return_value=_mock_response("*0.50*"))
        import baseline_agent
        from baseline_agent import aforecast

        with patch.object(baseline_agent, "THINKING_ENABLED", True):
            await aforecast(_make_question(source="acled"))
            kwargs = mock_litellm.acompletion.call_args.kwargs
            assert "thinking" in kwargs
            assert "temperature" not in kwargs

    @patch("baseline_agent.litellm")
    async def test_aforecast_disables_thinking_for_market_sources(self, mock_litellm: MagicMock) -> None:
        mock_litellm.acompletion = AsyncMock(return_value=_mock_response("*0.50*"))
        from baseline_agent import aforecast

        await aforecast(_make_question(source="metaculus"))
        kwargs = mock_litellm.acompletion.call_args.kwargs
        assert "thinking" not in kwargs
        assert kwargs["temperature"] == 0.3

    @patch("baseline_agent.litellm")
    async def test_aforecast_uses_configured_max_tokens(self, mock_litellm: MagicMock) -> None:
        mock_litellm.acompletion = AsyncMock(return_value=_mock_response("*0.50*"))
        from baseline_agent import aforecast, MAX_TOKENS

        await aforecast(_make_question())
        kwargs = mock_litellm.acompletion.call_args.kwargs
        assert kwargs["max_tokens"] == MAX_TOKENS

    @patch("baseline_agent.litellm")
    async def test_aforecast_multi_horizon_enables_thinking_for_events(self, mock_litellm: MagicMock) -> None:
        mock_litellm.acompletion = AsyncMock(return_value=_mock_response("*0.30* *0.50* *0.70*"))
        import baseline_agent
        from baseline_agent import aforecast_multi_horizon

        with patch.object(baseline_agent, "THINKING_ENABLED", True):
            await aforecast_multi_horizon(
                _make_dataset_question(),
                ["2024-07-01", "2024-08-01", "2024-09-01"],
                source="acled",
            )
            kwargs = mock_litellm.acompletion.call_args.kwargs
            assert "thinking" in kwargs
            assert "temperature" not in kwargs

    @patch("baseline_agent.litellm")
    async def test_aforecast_multi_horizon_uses_configured_max_tokens(self, mock_litellm: MagicMock) -> None:
        mock_litellm.acompletion = AsyncMock(return_value=_mock_response("*0.30* *0.50* *0.70*"))
        from baseline_agent import aforecast_multi_horizon, MAX_TOKENS

        await aforecast_multi_horizon(
            _make_dataset_question(),
            ["2024-07-01", "2024-08-01", "2024-09-01"],
            source="acled",
        )
        kwargs = mock_litellm.acompletion.call_args.kwargs
        assert kwargs["max_tokens"] == MAX_TOKENS

    @patch("baseline_agent.litellm")
    async def test_aforecast_multi_disables_thinking_for_timeseries(self, mock_litellm: MagicMock) -> None:
        mock_litellm.acompletion = AsyncMock(return_value=_mock_response("*0.30* *0.50* *0.70*"))
        from baseline_agent import aforecast_multi

        q = Question(
            id="tsq1", source="fred", question="Will value exceed threshold?",
            freeze_datetime="2024-06-05", forecast_due_date="2024-06-15",
            freeze_datetime_value=42.5,
            resolution_dates=["2024-07-01", "2024-08-01", "2024-09-01"],
        )
        await aforecast_multi(q, ["2024-07-01", "2024-08-01", "2024-09-01"])
        kwargs = mock_litellm.acompletion.call_args.kwargs
        assert "thinking" not in kwargs
        assert kwargs["temperature"] == 0.3

    @patch("baseline_agent.litellm")
    async def test_aforecast_multi_uses_configured_max_tokens(self, mock_litellm: MagicMock) -> None:
        mock_litellm.acompletion = AsyncMock(return_value=_mock_response("*0.30* *0.50* *0.70*"))
        from baseline_agent import aforecast_multi, MAX_TOKENS

        await aforecast_multi(_make_dataset_question(), ["2024-07-01", "2024-08-01", "2024-09-01"])
        kwargs = mock_litellm.acompletion.call_args.kwargs
        assert kwargs["max_tokens"] == MAX_TOKENS

    @patch("baseline_agent.litellm")
    def test_extraction_calls_do_not_use_forecast_kwargs(self, mock_litellm: MagicMock) -> None:
        """Extraction/parsing calls should NOT use _forecast_kwargs."""
        mock_litellm.completion.return_value = _mock_response("[0.30, 0.50]")
        _parse_probabilities("no asterisks here", 2)
        kwargs = mock_litellm.completion.call_args.kwargs
        assert kwargs["temperature"] == 0
        assert "thinking" not in kwargs


class TestSourceSpecificPrompts:
    """Source-specific prompt routing for timeseries questions."""

    def _ts_question(self, source: str) -> Question:
        return _make_question(
            source=source,
            freeze="2024-06-15",
            freeze_datetime_value=3.5,
            freeze_datetime_value_explanation="Current value",
            resolution_dates=["2024-07-01", "2024-08-01"],
        )

    def test_fred_gets_macro_prompt(self) -> None:
        with patch("baseline_agent.SOURCE_SPECIFIC_PROMPTS", True):
            prompt = _build_prompt(self._ts_question("fred"))
        assert "macroeconomic forecaster" in prompt
        assert "monetary policy" in prompt
        assert "mean-revert" in prompt

    def test_yfinance_gets_financial_prompt(self) -> None:
        with patch("baseline_agent.SOURCE_SPECIFIC_PROMPTS", True):
            prompt = _build_prompt(self._ts_question("yfinance"))
        assert "financial analyst" in prompt
        assert "random walk" in prompt
        assert "volatility" in prompt

    def test_dbnomics_gets_statistical_prompt(self) -> None:
        with patch("baseline_agent.SOURCE_SPECIFIC_PROMPTS", True):
            prompt = _build_prompt(self._ts_question("dbnomics"))
        assert "data analyst" in prompt
        assert "seasonal" in prompt
        assert "publication" in prompt

    def test_source_specific_disabled_uses_generic(self) -> None:
        with patch("baseline_agent.SOURCE_SPECIFIC_PROMPTS", False):
            for source in ["fred", "yfinance", "dbnomics"]:
                prompt = _build_prompt(self._ts_question(source))
                assert "macroeconomic forecaster" not in prompt
                assert "financial analyst" not in prompt
                assert "data analyst specializing in statistical" not in prompt
                assert "superforecaster" in prompt

    def test_acled_uses_generic(self) -> None:
        q = _make_question(
            source="acled",
            freeze="2024-06-15",
            freeze_datetime_value=50.0,
            freeze_datetime_value_explanation="Count",
            resolution_dates=["2024-07-01"],
        )
        with patch("baseline_agent.SOURCE_SPECIFIC_PROMPTS", True):
            prompt = _build_prompt(q)
        assert "superforecaster" in prompt
        assert "macroeconomic forecaster" not in prompt
        assert "financial analyst" not in prompt
        assert "data analyst specializing in statistical" not in prompt

    def test_source_specific_preserves_placeholders(self) -> None:
        with patch("baseline_agent.SOURCE_SPECIFIC_PROMPTS", True):
            for source in ["fred", "yfinance", "dbnomics"]:
                prompt = _build_prompt(self._ts_question(source))
                assert "3.5" in prompt
                assert "Current value" in prompt
                assert "2024-06-15" in prompt
                assert "2024-07-01" in prompt
                assert "asterisk" in prompt.lower()

    def test_source_specific_with_scratchpad_variant(self) -> None:
        with patch("baseline_agent.SOURCE_SPECIFIC_PROMPTS", True):
            prompt = _build_prompt(self._ts_question("fred"), prompt_variant="scratchpad")
        assert "macroeconomic forecaster" in prompt

    def test_wikipedia_uses_generic(self) -> None:
        q = _make_question(
            source="wikipedia",
            freeze="2024-06-15",
            freeze_datetime_value=100.0,
            freeze_datetime_value_explanation="Page views",
            resolution_dates=["2024-07-01"],
        )
        with patch("baseline_agent.SOURCE_SPECIFIC_PROMPTS", True):
            prompt = _build_prompt(q)
        assert "superforecaster" in prompt

    def test_market_sources_unaffected(self) -> None:
        q = _make_question(source="metaculus")
        with patch("baseline_agent.SOURCE_SPECIFIC_PROMPTS", True):
            prompt = _build_prompt(q)
        assert "macroeconomic forecaster" not in prompt
        assert "financial analyst" not in prompt
        assert "data analyst specializing in statistical" not in prompt


class TestComputeStatisticsContext:
    """Tests for _compute_statistics_context and FORECAST_STATS_CONTEXT integration."""

    def _ts_question(
        self,
        source: str = "fred",
        freeze_value: float | None = 4.5,
        question_text: str = "Will the rate exceed 5.0 by the resolution date?",
        resolution_criteria: str = "Resolves YES if the rate exceeds 5.0.",
    ) -> Question:
        return Question(
            id="stats1",
            source=source,
            question=question_text,
            background="Economic indicator",
            resolution_criteria=resolution_criteria,
            freeze_datetime="2024-06-15",
            forecast_due_date="2024-06-15",
            freeze_datetime_value=freeze_value,
            resolution_dates=["2024-07-15", "2024-08-15"],
        )

    def test_returns_context_for_timeseries_with_threshold(self) -> None:
        from baseline_agent import _compute_statistics_context
        q = self._ts_question(freeze_value=4.5)
        ctx = _compute_statistics_context(q, "2024-07-15", "2024-06-15")
        assert ctx is not None
        assert "Current value: 4.5" in ctx
        assert "Threshold: 5.0" in ctx
        assert "below the threshold" in ctx
        assert "Days to resolution: 30" in ctx

    def test_returns_correct_above_position(self) -> None:
        from baseline_agent import _compute_statistics_context
        q = self._ts_question(freeze_value=6.0)
        ctx = _compute_statistics_context(q, "2024-07-15", "2024-06-15")
        assert ctx is not None
        assert "above the threshold" in ctx

    def test_computes_pct_distance(self) -> None:
        from baseline_agent import _compute_statistics_context
        q = self._ts_question(freeze_value=4.0)
        ctx = _compute_statistics_context(q, "2024-07-15", "2024-06-15")
        assert ctx is not None
        assert "25.0% below the threshold" in ctx

    def test_returns_none_for_missing_freeze_value(self) -> None:
        from baseline_agent import _compute_statistics_context
        q = self._ts_question(freeze_value=None)
        ctx = _compute_statistics_context(q, "2024-07-15", "2024-06-15")
        assert ctx is None

    def test_returns_none_for_non_timeseries_source(self) -> None:
        from baseline_agent import _compute_statistics_context
        q = self._ts_question(source="metaculus")
        ctx = _compute_statistics_context(q, "2024-07-15", "2024-06-15")
        assert ctx is None

    def test_returns_none_for_unparseable_threshold(self) -> None:
        from baseline_agent import _compute_statistics_context
        q = self._ts_question(question_text="Will something happen?", resolution_criteria="Resolves YES if yes.")
        ctx = _compute_statistics_context(q, "2024-07-15", "2024-06-15")
        assert ctx is None

    def test_all_timeseries_sources_work(self) -> None:
        from baseline_agent import _compute_statistics_context
        for source in ["fred", "dbnomics", "yfinance"]:
            q = self._ts_question(source=source)
            ctx = _compute_statistics_context(q, "2024-07-15", "2024-06-15")
            assert ctx is not None, f"Source {source} returned None"

    def test_zero_current_value_handles_pct(self) -> None:
        from baseline_agent import _compute_statistics_context
        q = self._ts_question(freeze_value=0.0)
        ctx = _compute_statistics_context(q, "2024-07-15", "2024-06-15")
        assert ctx is not None
        assert "0.0% below the threshold" in ctx

    def test_stats_context_disabled_by_default(self) -> None:
        q = self._ts_question()
        prompt = _build_prompt(q, resolution_date="2024-07-15")
        assert "Statistical Context:" not in prompt

    def test_stats_context_enabled_injects_into_single_date(self) -> None:
        q = self._ts_question()
        with patch("baseline_agent.STATS_CONTEXT", True):
            prompt = _build_prompt(q, resolution_date="2024-07-15")
        assert "Statistical Context:" in prompt
        assert "Current value: 4.5" in prompt
        assert "Threshold: 5.0" in prompt
        assert prompt.index("Statistical Context:") < prompt.index("Output your answer")

    def test_stats_context_enabled_injects_into_multi_date(self) -> None:
        q = self._ts_question()
        with patch("baseline_agent.STATS_CONTEXT", True):
            prompt = _build_prompt(q, prompt_variant="dataset")
        assert "Statistical Context:" in prompt
        assert "Current value: 4.5" in prompt

    def test_stats_context_false_no_change(self) -> None:
        q = self._ts_question()
        with patch("baseline_agent.STATS_CONTEXT", False):
            prompt = _build_prompt(q, resolution_date="2024-07-15")
        assert "Statistical Context:" not in prompt

    def test_stats_context_market_source_not_injected(self) -> None:
        q = _make_question(source="metaculus", freeze_datetime_value=0.5)
        with patch("baseline_agent.STATS_CONTEXT", True):
            prompt = _build_prompt(q)
        assert "Statistical Context:" not in prompt

    def test_stats_context_no_threshold_no_injection(self) -> None:
        q = self._ts_question(question_text="Will something happen?", resolution_criteria="Resolves YES if yes.")
        with patch("baseline_agent.STATS_CONTEXT", True):
            prompt = _build_prompt(q, resolution_date="2024-07-15")
        assert "Statistical Context:" not in prompt

    def test_stats_context_scratchpad_fallback_path(self) -> None:
        q = self._ts_question()
        with patch("baseline_agent.STATS_CONTEXT", True):
            prompt = _build_prompt(q, prompt_variant="scratchpad")
        assert "Statistical Context:" in prompt
