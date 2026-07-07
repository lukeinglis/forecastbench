"""Tests for baseline LLM forecasting agent."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

from fetch_data import Question
from cutoff import CutoffEnvironment
from baseline_agent import _parse_probability, aforecast, forecast


class TestParseProbability:
    def test_simple_decimal(self) -> None:
        assert _parse_probability("0.73") == 0.73

    def test_embedded_in_text(self) -> None:
        assert _parse_probability("The probability is 0.85") == 0.85

    def test_empty_string(self) -> None:
        assert _parse_probability("") == 0.5

    def test_nan_string(self) -> None:
        assert _parse_probability("nan") == 0.5

    def test_negative_clamped(self) -> None:
        assert _parse_probability("-0.5") == 0.5

    def test_above_one_clamped(self) -> None:
        assert _parse_probability("1.5") == 1.0

    def test_no_number(self) -> None:
        assert _parse_probability("no number here") == 0.5

    def test_integer(self) -> None:
        assert _parse_probability("1") == 1.0

    def test_zero(self) -> None:
        assert _parse_probability("0") == 0.0

    def test_zero_point_zero(self) -> None:
        assert _parse_probability("0.0") == 0.0


def _mock_response(content: str) -> MagicMock:
    resp = MagicMock()
    resp.choices = [MagicMock()]
    resp.choices[0].message.content = content
    return resp


class TestAforecast:
    def test_returns_float_in_range(self) -> None:
        with patch("baseline_agent.litellm") as mock_litellm:
            mock_litellm.acompletion = AsyncMock(return_value=_mock_response("0.42"))
            q = Question(id="q1", source="acled", question="Will X happen?")
            result = asyncio.run(aforecast(q))

        assert isinstance(result, float)
        assert 0.0 <= result <= 1.0
        assert result == 0.42

    def test_prompt_includes_cutoff(self) -> None:
        with patch("baseline_agent.litellm") as mock_litellm:
            mock_litellm.acompletion = AsyncMock(return_value=_mock_response("0.65"))
            q = Question(id="q1", source="acled", question="Will X happen?")
            cutoff = CutoffEnvironment(freeze_datetime="2024-03-15")
            asyncio.run(aforecast(q, cutoff=cutoff))

            call_args = mock_litellm.acompletion.call_args
            messages = call_args[1]["messages"]
            user_msg = messages[1]["content"]
            assert "2024-03-15" in user_msg

    def test_prompt_without_cutoff(self) -> None:
        with patch("baseline_agent.litellm") as mock_litellm:
            mock_litellm.acompletion = AsyncMock(return_value=_mock_response("0.50"))
            q = Question(id="q1", source="acled", question="Will X happen?")
            asyncio.run(aforecast(q))

            call_args = mock_litellm.acompletion.call_args
            messages = call_args[1]["messages"]
            user_msg = messages[1]["content"]
            assert "knowledge cutoff" not in user_msg.lower()

    def test_includes_background_and_criteria(self) -> None:
        with patch("baseline_agent.litellm") as mock_litellm:
            mock_litellm.acompletion = AsyncMock(return_value=_mock_response("0.70"))
            q = Question(
                id="q1",
                source="acled",
                question="Will X happen?",
                background="Some context",
                resolution_criteria="Resolves YES if ...",
            )
            asyncio.run(aforecast(q))

            call_args = mock_litellm.acompletion.call_args
            messages = call_args[1]["messages"]
            user_msg = messages[1]["content"]
            assert "Some context" in user_msg
            assert "Resolves YES if" in user_msg


class TestForecast:
    def test_sync_wrapper_returns_float(self) -> None:
        with patch("baseline_agent.litellm") as mock_litellm:
            mock_litellm.acompletion = AsyncMock(return_value=_mock_response("0.55"))
            q = Question(id="q1", source="acled", question="Will X happen?")
            result = forecast(q)

        assert isinstance(result, float)
        assert 0.0 <= result <= 1.0
