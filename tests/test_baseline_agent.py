"""Tests for baseline LLM forecaster (mocked litellm)."""

from __future__ import annotations

import time
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
        assert "@" in MODEL

    @patch.dict("os.environ", {"FORECAST_MODEL": "gpt-4o"})
    def test_model_configurable_via_env(self) -> None:
        import importlib
        import baseline_agent

        importlib.reload(baseline_agent)
        assert baseline_agent.MODEL == "gpt-4o"
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
