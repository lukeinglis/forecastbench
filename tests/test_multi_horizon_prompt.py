"""Tests for multi-horizon single-prompt forecasting and LLM parsing fallback."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from baseline_agent import (
    FORECAST_EXTRACTION_PROMPT,
    _asterisk_extract,
    _decimal_extract,
    _extract_answer_block,
    _extract_probabilities,
    _extract_with_llm,
    _parse_probs_from_text,
    _tokenize_and_extract,
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
        text = "Draft: 0.3, 0.4, 0.5\n\nFinal: 0.7, 0.8"
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


class TestExtractAnswerBlock:
    def test_finds_answer_marker(self) -> None:
        text = "Some reasoning here.\n\nAnswer: *0.3* *0.5* *0.7*"
        block = _extract_answer_block(text)
        assert block is not None
        assert "*0.3*" in block
        assert "reasoning" not in block

    def test_case_insensitive(self) -> None:
        text = "Reasoning.\n\nanswer: *0.4*"
        block = _extract_answer_block(text)
        assert block is not None
        assert "*0.4*" in block

    def test_answer_with_curly_braces(self) -> None:
        text = "Reasoning.\n\nAnswer: {\n*0.3*\n*0.5*\n*0.7*\n}"
        block = _extract_answer_block(text)
        assert block is not None
        assert "*0.3*" in block

    def test_falls_back_to_last_paragraph(self) -> None:
        text = "First paragraph reasoning.\n\n*0.3* *0.5* *0.7*"
        block = _extract_answer_block(text)
        assert block is not None
        assert "*0.3*" in block

    def test_single_paragraph_returns_none(self) -> None:
        text = "Just one paragraph with *0.3* in it."
        block = _extract_answer_block(text)
        assert block is None


class TestParseFromAnswerBlock:
    def test_asterisk_format(self) -> None:
        result = _parse_probs_from_text("*0.3* *0.5* *0.7*", 3)
        assert result == pytest.approx([0.3, 0.5, 0.7])

    def test_decimal_format(self) -> None:
        result = _parse_probs_from_text("0.3 0.5 0.7", 3)
        assert result == pytest.approx([0.3, 0.5, 0.7])

    def test_wrong_count_returns_none(self) -> None:
        result = _parse_probs_from_text("*0.3* *0.5*", 3)
        assert result is None


class TestTokenizeAndExtract:
    def test_asterisk_tokens(self) -> None:
        result = _tokenize_and_extract("*0.3* *0.5* *0.7*", 3)
        assert result == pytest.approx([0.3, 0.5, 0.7])

    def test_comma_separated(self) -> None:
        result = _tokenize_and_extract("*0.3*, *0.5*, *0.7*", 3)
        assert result == pytest.approx([0.3, 0.5, 0.7])

    def test_curly_braces(self) -> None:
        result = _tokenize_and_extract("{ *0.3* *0.5* *0.7* }", 3)
        assert result == pytest.approx([0.3, 0.5, 0.7])

    def test_overflow_takes_last_n(self) -> None:
        text = "0.1 0.2 0.3 0.4 0.5 0.6"
        result = _tokenize_and_extract(text, 3)
        assert result == pytest.approx([0.4, 0.5, 0.6])

    def test_non_prob_tokens_skipped(self) -> None:
        text = "The value is *0.42* and also *0.58*"
        result = _tokenize_and_extract(text, 2)
        assert result == pytest.approx([0.42, 0.58])


class TestAsteriskExtract:
    def test_exact_count(self) -> None:
        result = _asterisk_extract("*0.3* *0.5* *0.7*", 3)
        assert result == pytest.approx([0.3, 0.5, 0.7])

    def test_non_multiple_overflow(self) -> None:
        text = " ".join(f"*0.{i}*" for i in range(1, 10))
        result = _asterisk_extract(text, 3)
        assert result is not None
        assert len(result) == 3
        assert result == pytest.approx([0.7, 0.8, 0.9])


class TestDecimalExtract:
    def test_exact_count(self) -> None:
        result = _decimal_extract("values: 0.3 0.5 0.7", 3)
        assert result == pytest.approx([0.3, 0.5, 0.7])

    def test_overflow_takes_last_n(self) -> None:
        text = "0.1, 0.2, 0.3, 0.4, 0.5"
        result = _decimal_extract(text, 3)
        assert result == pytest.approx([0.3, 0.4, 0.5])


class TestExtractProbabilitiesIntegration:
    def test_answer_block_with_reasoning(self) -> None:
        text = (
            "I need to estimate the probability that...\n\n"
            "Key considerations:\n"
            "- Current value: 19,500\n"
            "- Target: 20,000\n\n"
            "Estimates based on drift-diffusion model:\n\n"
            "Answer: {\n"
            "2026-07-12: *0.42*\n"
            "2026-08-04: *0.58*\n"
            "2026-10-03: *0.72*\n"
            "2027-01-01: *0.82*\n"
            "2027-07-05: *0.90*\n"
            "2029-07-04: *0.96*\n"
            "2031-07-04: *0.98*\n"
            "2036-07-02: *0.99*\n"
            "}"
        )
        result = _extract_probabilities(text, 8)
        assert result is not None
        assert len(result) == 8
        assert result == pytest.approx([0.42, 0.58, 0.72, 0.82, 0.90, 0.96, 0.98, 0.99])

    def test_inline_reasoning_then_answer(self) -> None:
        text = (
            "... reasoning with probabilities mentioned inline like *0.35* ...\n\n"
            "Answer:\n"
            "*0.35*\n"
            "*0.40*\n"
            "*0.30*\n"
            "*0.25*\n"
            "*0.45*\n"
            "*0.55*\n"
            "*0.62*\n"
            "*0.70*"
        )
        result = _extract_probabilities(text, 8)
        assert result is not None
        assert len(result) == 8
        assert result == pytest.approx([0.35, 0.40, 0.30, 0.25, 0.45, 0.55, 0.62, 0.70])

    def test_non_multiple_overflow_11_for_8(self) -> None:
        text = (
            "Draft: *0.1* *0.2* *0.3*\n"
            "Final: *0.42* *0.58* *0.72* *0.82* *0.90* *0.96* *0.98* *0.99*"
        )
        result = _extract_probabilities(text, 8)
        assert result is not None
        assert len(result) == 8
        assert result == pytest.approx([0.42, 0.58, 0.72, 0.82, 0.90, 0.96, 0.98, 0.99])

    def test_strategies_priority_order(self) -> None:
        text = "Answer: *0.4* *0.5* *0.6*"
        result = _extract_probabilities(text, 3)
        assert result == pytest.approx([0.4, 0.5, 0.6])


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
        """When both regex and LLM extraction fail, returns None (caller uses 0.5)."""
        main_response = _mock_response("I cannot determine the probabilities.")
        extraction_response = _mock_response("I don't know")
        mock_litellm.acompletion = AsyncMock(
            side_effect=[main_response, extraction_response]
        )

        q = _make_question()
        dates = ["2024-07-01", "2024-08-01", "2024-09-01"]

        result = await aforecast_multi_horizon(q, dates, source="fred")

        assert result is None

    @patch("baseline_agent.litellm")
    async def test_api_error_returns_default(self, mock_litellm: MagicMock) -> None:
        mock_litellm.acompletion = AsyncMock(side_effect=RuntimeError("API down"))

        q = _make_question()
        dates = ["2024-07-01", "2024-08-01"]

        result = await aforecast_multi_horizon(q, dates, source="fred")

        assert result is None

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
