"""Tests for sub-question decomposition forecaster."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from decompose import (
    DECOMPOSITION_PROMPT,
    decomposed_forecast,
    decomposed_forecast_multi_horizon,
    is_complex_question,
)
from fetch_data import Question


def _make_question(**kwargs: object) -> Question:
    defaults: dict[str, object] = {
        "id": "test-q1",
        "source": "metaculus",
        "question": "Will X happen by 2025?",
        "background": "Some background info.",
        "resolution_criteria": "Resolves YES if X happens.",
    }
    defaults.update(kwargs)
    return Question(**defaults)  # type: ignore[arg-type]


class TestDecompositionPrompt:
    def test_prompt_contains_placeholders(self) -> None:
        assert "{question}" in DECOMPOSITION_PROMPT
        assert "{background}" in DECOMPOSITION_PROMPT
        assert "{resolution_criteria}" in DECOMPOSITION_PROMPT

    def test_prompt_renders_with_question_fields(self) -> None:
        rendered = DECOMPOSITION_PROMPT.format(
            question="Will inflation exceed 3%?",
            background="CPI data shows rising trend.",
            resolution_criteria="Resolves YES if CPI > 3%.",
        )
        assert "Will inflation exceed 3%?" in rendered
        assert "CPI data shows rising trend." in rendered
        assert "Resolves YES if CPI > 3%." in rendered
        assert "FACTOR:" in rendered
        assert "DIRECTION:" in rendered
        assert "WEIGHT:" in rendered


class TestIsComplexQuestion:
    def test_simple_question_returns_false(self) -> None:
        q = _make_question(resolution_criteria="Resolves YES if X happens.")
        assert is_complex_question(q) is False

    def test_long_criteria_returns_true(self) -> None:
        q = _make_question(resolution_criteria="A" * 201)
        assert is_complex_question(q) is True

    def test_multiple_conditions_returns_true(self) -> None:
        q = _make_question(
            resolution_criteria="Resolves YES if A happens and B is true or C occurs."
        )
        assert is_complex_question(q) is True

    def test_combination_of_returns_true(self) -> None:
        q = _make_question(combination_of=["q1", "q2"])
        assert is_complex_question(q) is True

    def test_empty_criteria_returns_false(self) -> None:
        q = _make_question(resolution_criteria="")
        assert is_complex_question(q) is False

    def test_single_condition_word_returns_false(self) -> None:
        q = _make_question(resolution_criteria="Resolves YES if X happens.")
        assert is_complex_question(q) is False


class TestDecomposedForecast:
    def test_calls_llm_and_parses_probability(self) -> None:
        q = _make_question()
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = (
            "FACTOR: Economic growth\n"
            "DIRECTION: increases\n"
            "WEIGHT: high\n\n"
            "FACTOR: Political stability\n"
            "DIRECTION: neutral\n"
            "WEIGHT: medium\n\n"
            "*0.72*"
        )

        with patch("decompose.litellm.acompletion", new_callable=AsyncMock, return_value=mock_response) as mock_llm:
            with patch("decompose._ensure_vertex_credentials"):
                result = asyncio.run(decomposed_forecast(q))

        assert result == pytest.approx(0.72)
        mock_llm.assert_called_once()
        call_kwargs = mock_llm.call_args
        prompt_content = call_kwargs.kwargs["messages"][0]["content"]
        assert "Will X happen by 2025?" in prompt_content

    def test_uses_decomposition_prompt_not_baseline(self) -> None:
        q = _make_question()
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "*0.65*"

        with patch("decompose.litellm.acompletion", new_callable=AsyncMock, return_value=mock_response):
            with patch("decompose._ensure_vertex_credentials"):
                result = asyncio.run(decomposed_forecast(q))

        assert result == pytest.approx(0.65)

    def test_api_error_propagates(self) -> None:
        q = _make_question()

        with patch("decompose.litellm.acompletion", new_callable=AsyncMock, side_effect=RuntimeError("API down")):
            with patch("decompose._ensure_vertex_credentials"):
                with pytest.raises(RuntimeError, match="API down"):
                    asyncio.run(decomposed_forecast(q))


class TestDecomposedMultiHorizon:
    def test_extracts_multiple_probabilities(self) -> None:
        q = _make_question(
            source="acled",
            resolution_dates=["2025-01-01", "2025-06-01", "2025-12-01"],
            freeze_datetime="2024-06-01",
        )
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = (
            "FACTOR: Trend analysis\nDIRECTION: increases\nWEIGHT: high\n\n"
            "*0.30* *0.55* *0.75*"
        )

        with patch("decompose.litellm.acompletion", new_callable=AsyncMock, return_value=mock_response):
            with patch("decompose._ensure_vertex_credentials"):
                result = asyncio.run(
                    decomposed_forecast_multi_horizon(q, ["2025-01-01", "2025-06-01", "2025-12-01"])
                )

        assert result is not None
        assert len(result) == 3
        assert result == pytest.approx([0.30, 0.55, 0.75])

    def test_returns_none_on_api_error(self) -> None:
        q = _make_question(source="acled", freeze_datetime="2024-06-01")

        with patch("decompose.litellm.acompletion", new_callable=AsyncMock, side_effect=RuntimeError("timeout")):
            with patch("decompose._ensure_vertex_credentials"):
                result = asyncio.run(
                    decomposed_forecast_multi_horizon(q, ["2025-01-01"])
                )

        assert result is None


class TestEvalIntegration:
    def test_eval_accepts_decompose_agent(self) -> None:
        """Verify eval.py argparser accepts --agent decompose and runs."""
        import sys

        from eval import main as _main

        old_argv = sys.argv
        try:
            sys.argv = ["eval.py", "--agent", "decompose", "--list-rounds"]
            with patch("eval.list_rounds", return_value=[]):
                with patch("eval.configure_logging"):
                    with patch("eval.generate_run_id", return_value="test"):
                        _main()
        finally:
            sys.argv = old_argv
