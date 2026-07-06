"""Tests for LLM baseline forecaster."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from fetch_data import Question
from baseline_agent import build_prompt, parse_probability, forecast


def _make_question(**kwargs) -> Question:
    defaults = {
        "id": "q1",
        "source": "acled",
        "question": "Will X happen by end of 2024?",
        "background": "X has been trending upward.",
        "resolution_criteria": "Resolves YES if X exceeds threshold.",
    }
    defaults.update(kwargs)
    return Question(**defaults)


class TestBuildPrompt:
    def test_includes_question_fields(self) -> None:
        q = _make_question()
        prompt = build_prompt(q, "January 15, 2024")
        assert "Will X happen by end of 2024?" in prompt
        assert "X has been trending upward." in prompt
        assert "Resolves YES if X exceeds threshold." in prompt
        assert "January 15, 2024" in prompt

    def test_includes_superforecaster_persona(self) -> None:
        q = _make_question()
        prompt = build_prompt(q, "January 15, 2024")
        assert "superforecaster" in prompt
        assert "Tetlock" in prompt

    def test_includes_freeze_datetime_value(self) -> None:
        q = _make_question(
            freeze_datetime="2024-06-15T12:00:00",
            freeze_datetime_value=42.5,
        )
        prompt = build_prompt(q, "June 15, 2024")
        assert "Current value on 2024-06-15T12:00:00: 42.5" in prompt

    def test_excludes_freeze_value_when_none(self) -> None:
        q = _make_question(freeze_datetime_value=None)
        prompt = build_prompt(q, "January 15, 2024")
        assert "Current value" not in prompt

    def test_output_format_instruction(self) -> None:
        q = _make_question()
        prompt = build_prompt(q, "January 15, 2024")
        assert "asterisk" in prompt


class TestParseProbability:
    @pytest.mark.parametrize(
        "text,expected",
        [
            ("*0.75*", 0.75),
            ("*0.1*", 0.1),
            ("* 0.85 *", 0.85),
            ("*0.00*", 0.0),
            ("*1.0*", 1.0),
        ],
    )
    def test_asterisk_pattern(self, text: str, expected: float) -> None:
        assert parse_probability(text) == pytest.approx(expected)

    def test_forecast_pattern(self) -> None:
        assert parse_probability("Forecast: 0.65") == pytest.approx(0.65)

    def test_bare_decimal(self) -> None:
        assert parse_probability("The answer is 0.8") == pytest.approx(0.8)

    def test_percentage(self) -> None:
        assert parse_probability("75%") == pytest.approx(0.75)

    def test_garbage_returns_fallback(self) -> None:
        assert parse_probability("I don't know") == pytest.approx(0.5)

    def test_empty_returns_fallback(self) -> None:
        assert parse_probability("") == pytest.approx(0.5)

    def test_none_returns_fallback(self) -> None:
        assert parse_probability(None) == pytest.approx(0.5)

    def test_clamps_high(self) -> None:
        assert parse_probability("*1.5*") == pytest.approx(1.0)

    def test_clamps_low(self) -> None:
        assert parse_probability("*-0.1*") == pytest.approx(0.0)


class TestForecast:
    @patch("baseline_agent.litellm")
    def test_returns_parsed_probability(self, mock_litellm: MagicMock) -> None:
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "*0.72*"
        mock_litellm.completion.return_value = mock_response

        q = _make_question()
        result = forecast(q, model="test/model", today_date="January 15, 2024")
        assert result == pytest.approx(0.72)
        mock_litellm.completion.assert_called_once()

    @patch("baseline_agent.litellm")
    def test_uses_temperature_zero(self, mock_litellm: MagicMock) -> None:
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "*0.5*"
        mock_litellm.completion.return_value = mock_response

        q = _make_question()
        forecast(q, model="test/model", today_date="January 15, 2024")
        call_kwargs = mock_litellm.completion.call_args[1]
        assert call_kwargs["temperature"] == 0
        assert call_kwargs["max_tokens"] == 2000

    @patch("baseline_agent.litellm")
    def test_returns_fallback_on_exception(self, mock_litellm: MagicMock) -> None:
        mock_litellm.completion.side_effect = RuntimeError("API error")

        q = _make_question()
        result = forecast(q, model="test/model", today_date="January 15, 2024")
        assert result == pytest.approx(0.5)

    @patch("baseline_agent.litellm")
    def test_uses_freeze_datetime_as_today_when_none(self, mock_litellm: MagicMock) -> None:
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "*0.6*"
        mock_litellm.completion.return_value = mock_response

        q = _make_question(freeze_datetime="2024-06-15T12:00:00")
        forecast(q, model="test/model")
        call_args = mock_litellm.completion.call_args
        prompt = call_args[1]["messages"][0]["content"]
        assert "2024-06-15T12:00:00" in prompt
