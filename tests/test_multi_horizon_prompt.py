"""Tests for multi-horizon single-prompt forecasting and LLM parsing fallback."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from baseline_agent import (
    FORECAST_EXTRACTION_PROMPT,
    _extract_probabilities,
    _extract_with_llm,
    aforecast_multi_horizon,
)
from fetch_data import Question


def _make_question(
    source: str = "fred",
    freeze: str | None = "2024-06-15",
    freeze_datetime_value: float | None = 3.5,
    resolution_dates: list[str] | None = None,
    freeze_datetime_value_explanation: str | None = None,
) -> Question:
    return Question(
        id="q1",
        source=source,
        question="Will X exceed threshold?",
        background="Some background",
        resolution_criteria="Resolves YES if X > 100.",
        freeze_datetime=freeze,
        freeze_datetime_value=freeze_datetime_value,
        freeze_datetime_value_explanation=freeze_datetime_value_explanation,
        resolution_dates=resolution_dates or ["2024-07-01", "2024-08-01", "2024-09-01"],
    )


def _mock_response(content: str) -> MagicMock:
    msg = MagicMock()
    msg.content = content
    choice = MagicMock()
    choice.message = msg
    resp = MagicMock()
    resp.choices = [choice]
    return resp


class TestExtractProbabilities:
    def test_asterisk_format(self) -> None:
        result = _extract_probabilities("*0.3* *0.5* *0.7*", 3)
        assert result is not None
        assert result == pytest.approx([0.3, 0.5, 0.7])

    def test_asterisk_format_with_surrounding_text(self) -> None:
        text = "For the three dates, I estimate *0.2* for the first, *0.4* for the second, and *0.6* for the third."
        result = _extract_probabilities(text, 3)
        assert result is not None
        assert result == pytest.approx([0.2, 0.4, 0.6])

    def test_wrong_count_returns_none(self) -> None:
        result = _extract_probabilities("*0.3* *0.5*", 3)
        assert result is None

    def test_double_count_takes_last_n(self) -> None:
        text = "Reasoning: *0.1* *0.2* *0.3*\nAnswer: { *0.4*, *0.5*, *0.6* }"
        result = _extract_probabilities(text, 3)
        assert result is not None
        assert result == pytest.approx([0.4, 0.5, 0.6])

    def test_triple_count_takes_last_n(self) -> None:
        text = "Draft: *0.1* *0.2*\nRevised: *0.3* *0.4*\nFinal: *0.5* *0.6*"
        result = _extract_probabilities(text, 2)
        assert result is not None
        assert result == pytest.approx([0.5, 0.6])

    def test_decimal_overflow_takes_last_n(self) -> None:
        text = "Step 1: 0.3, 0.4. Step 2: 0.5, 0.6. Final: 0.7, 0.8."
        result = _extract_probabilities(text, 2)
        assert result is not None
        assert result == pytest.approx([0.7, 0.8])

    def test_decimal_format(self) -> None:
        text = "My estimates are 0.3, 0.5, and 0.7 for the three dates."
        result = _extract_probabilities(text, 3)
        assert result is not None
        assert result == pytest.approx([0.3, 0.5, 0.7])

    def test_clamps_to_bounds(self) -> None:
        result = _extract_probabilities("*0.001* *0.999* *0.5*", 3)
        assert result is not None
        assert result[0] == pytest.approx(0.01)
        assert result[1] == pytest.approx(0.99)
        assert result[2] == pytest.approx(0.5)

    def test_zero_clamped_to_min(self) -> None:
        result = _extract_probabilities("*0* *0.5* *1.0*", 3)
        assert result is not None
        assert result[0] == pytest.approx(0.01)
        assert result[2] == pytest.approx(0.99)

    def test_no_matches_returns_none(self) -> None:
        result = _extract_probabilities("I cannot determine probabilities", 3)
        assert result is None

    def test_decimal_fallback_wrong_count(self) -> None:
        text = "Estimates: 0.3 and 0.5"
        result = _extract_probabilities(text, 3)
        assert result is None

    def test_single_horizon(self) -> None:
        result = _extract_probabilities("*0.75*", 1)
        assert result is not None
        assert result == pytest.approx([0.75])


class TestExtractionPrompt:
    def test_prompt_has_placeholders(self) -> None:
        assert "{n_horizons}" in FORECAST_EXTRACTION_PROMPT
        assert "{model_response}" in FORECAST_EXTRACTION_PROMPT

    def test_prompt_formats_correctly(self) -> None:
        formatted = FORECAST_EXTRACTION_PROMPT.format(
            n_horizons=3,
            model_response="*0.3* *0.5* *0.7*",
        )
        assert "3" in formatted
        assert "*0.3* *0.5* *0.7*" in formatted
        assert "Extract only probabilities" in formatted

    def test_extraction_model_defaults_to_forecast_model(self) -> None:
        import baseline_agent
        assert baseline_agent.EXTRACTION_MODEL == baseline_agent.MODEL


class TestExtractWithLlm:
    @patch("baseline_agent.litellm")
    async def test_successful_extraction(self, mock_litellm: MagicMock) -> None:
        mock_litellm.acompletion = AsyncMock(
            return_value=_mock_response("[0.3, 0.5, 0.7]")
        )
        result = await _extract_with_llm("some model output", 3)
        assert result is not None
        assert result == pytest.approx([0.3, 0.5, 0.7])
        mock_litellm.acompletion.assert_called_once()
        call_kwargs = mock_litellm.acompletion.call_args.kwargs
        import baseline_agent
        assert call_kwargs["model"] == baseline_agent.EXTRACTION_MODEL
        assert call_kwargs["temperature"] == 0.0

    @patch("baseline_agent.litellm")
    async def test_wrong_count_returns_none(self, mock_litellm: MagicMock) -> None:
        mock_litellm.acompletion = AsyncMock(
            return_value=_mock_response("[0.3, 0.5]")
        )
        result = await _extract_with_llm("some model output", 3)
        assert result is None

    @patch("baseline_agent.litellm")
    async def test_invalid_values_returns_none(self, mock_litellm: MagicMock) -> None:
        mock_litellm.acompletion = AsyncMock(
            return_value=_mock_response("[0.3, 1.5, 0.7]")
        )
        result = await _extract_with_llm("some model output", 3)
        assert result is None

    @patch("baseline_agent.litellm")
    async def test_api_error_returns_none(self, mock_litellm: MagicMock) -> None:
        mock_litellm.acompletion = AsyncMock(side_effect=RuntimeError("API error"))
        result = await _extract_with_llm("some model output", 3)
        assert result is None

    @patch("baseline_agent.litellm")
    async def test_non_list_returns_none(self, mock_litellm: MagicMock) -> None:
        mock_litellm.acompletion = AsyncMock(
            return_value=_mock_response("0.5")
        )
        result = await _extract_with_llm("some model output", 3)
        assert result is None

    @patch("baseline_agent.litellm")
    async def test_clamps_values(self, mock_litellm: MagicMock) -> None:
        mock_litellm.acompletion = AsyncMock(
            return_value=_mock_response("[0.001, 0.5, 0.999]")
        )
        result = await _extract_with_llm("some model output", 3)
        assert result is not None
        assert result[0] == pytest.approx(0.01)
        assert result[2] == pytest.approx(0.99)


class TestAforecastMultiHorizon:
    @patch("baseline_agent.litellm")
    async def test_single_llm_call(self, mock_litellm: MagicMock) -> None:
        mock_litellm.acompletion = AsyncMock(
            return_value=_mock_response("*0.3* *0.5* *0.7*")
        )
        q = _make_question()
        dates = ["2024-07-01", "2024-08-01", "2024-09-01"]

        result = await aforecast_multi_horizon(q, dates, source="fred")

        assert len(result) == 3
        assert result == pytest.approx([0.3, 0.5, 0.7])
        mock_litellm.acompletion.assert_called_once()

    @patch("baseline_agent.litellm")
    async def test_extraction_fallback_called(self, mock_litellm: MagicMock) -> None:
        """When regex fails, the LLM extraction fallback is invoked."""
        main_response = _mock_response("I think the probabilities are roughly moderate across all dates.")
        extraction_response = _mock_response("[0.4, 0.5, 0.6]")
        mock_litellm.acompletion = AsyncMock(
            side_effect=[main_response, extraction_response]
        )

        q = _make_question()
        dates = ["2024-07-01", "2024-08-01", "2024-09-01"]

        result = await aforecast_multi_horizon(q, dates, source="fred")

        assert len(result) == 3
        assert result == pytest.approx([0.4, 0.5, 0.6])
        assert mock_litellm.acompletion.call_count == 2

    @patch("baseline_agent.litellm")
    async def test_fallback_to_default(self, mock_litellm: MagicMock) -> None:
        """When both regex and LLM extraction fail, returns [0.5] * n."""
        main_response = _mock_response("I cannot determine the probabilities.")
        extraction_response = _mock_response("I don't know")
        mock_litellm.acompletion = AsyncMock(
            side_effect=[main_response, extraction_response]
        )

        q = _make_question()
        dates = ["2024-07-01", "2024-08-01", "2024-09-01"]

        result = await aforecast_multi_horizon(q, dates, source="fred")

        assert result == [0.5, 0.5, 0.5]

    @patch("baseline_agent.litellm")
    async def test_api_error_returns_default(self, mock_litellm: MagicMock) -> None:
        mock_litellm.acompletion = AsyncMock(side_effect=RuntimeError("API down"))

        q = _make_question()
        dates = ["2024-07-01", "2024-08-01"]

        result = await aforecast_multi_horizon(q, dates, source="fred")

        assert result == [0.5, 0.5]

    @patch("baseline_agent.litellm")
    async def test_prompt_contains_all_dates(self, mock_litellm: MagicMock) -> None:
        mock_litellm.acompletion = AsyncMock(
            return_value=_mock_response("*0.3* *0.5* *0.7*")
        )
        q = _make_question()
        dates = ["2024-07-01", "2024-08-01", "2024-09-01"]

        await aforecast_multi_horizon(q, dates, source="fred", prompt_variant="dataset")

        call_kwargs = mock_litellm.acompletion.call_args.kwargs
        prompt = call_kwargs["messages"][0]["content"]
        for d in dates:
            assert d in prompt

    @patch("baseline_agent.litellm")
    async def test_uses_dataset_prompt_variant_by_default(self, mock_litellm: MagicMock) -> None:
        mock_litellm.acompletion = AsyncMock(
            return_value=_mock_response("*0.3* *0.5*")
        )
        q = _make_question()
        dates = ["2024-07-01", "2024-08-01"]

        await aforecast_multi_horizon(q, dates, source="fred")

        call_kwargs = mock_litellm.acompletion.call_args.kwargs
        prompt = call_kwargs["messages"][0]["content"]
        assert "Resolution dates:" in prompt or "resolution dates:" in prompt.lower()
