"""Tests for the belief state tracking forecaster."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from belief_forecaster import (
    BeliefState,
    _aggregate_forecasts,
    _parse_belief_state,
    _single_trial,
    belief_forecast,
    belief_forecast_multi_horizon,
)
from fetch_data import Question


def _make_question(**overrides: object) -> Question:
    defaults = {
        "id": "test-q-1",
        "source": "metaculus",
        "question": "Will X happen by 2026?",
        "background": "Some background info.",
        "resolution_criteria": "Resolves YES if X happens.",
        "freeze_datetime": "2026-01-01",
        "forecast_due_date": "2026-02-01",
    }
    defaults.update(overrides)
    return Question(**defaults)  # type: ignore[arg-type]


def _make_belief_response(
    probability: float = 0.7,
    confidence: str = "medium",
    evidence_for: list[str] | None = None,
    evidence_against: list[str] | None = None,
    reasoning: str = "Updated based on evidence.",
) -> str:
    return json.dumps({
        "probability": probability,
        "confidence": confidence,
        "evidence_for": evidence_for or ["Point A"],
        "evidence_against": evidence_against or ["Point B"],
        "reasoning": reasoning,
    })


class TestBeliefState:
    def test_default_initialization(self) -> None:
        state = BeliefState()
        assert state.probability == 0.5
        assert state.confidence == "low"
        assert state.evidence_for == []
        assert state.evidence_against == []
        assert state.reasoning == ""

    def test_custom_initialization(self) -> None:
        state = BeliefState(
            probability=0.8,
            confidence="high",
            evidence_for=["A"],
            evidence_against=["B"],
            reasoning="Strong evidence.",
        )
        assert state.probability == 0.8
        assert state.confidence == "high"
        assert state.evidence_for == ["A"]
        assert state.evidence_against == ["B"]


class TestParseBeliefState:
    def test_valid_json(self) -> None:
        text = _make_belief_response(0.65, "medium")
        state = _parse_belief_state(text)
        assert state.probability == 0.65
        assert state.confidence == "medium"
        assert len(state.evidence_for) == 1
        assert len(state.evidence_against) == 1

    def test_probability_clamped_above_1(self) -> None:
        text = json.dumps({"probability": 1.5, "confidence": "high"})
        state = _parse_belief_state(text)
        assert state.probability == 1.0

    def test_probability_clamped_below_0(self) -> None:
        text = json.dumps({"probability": -0.3, "confidence": "low"})
        state = _parse_belief_state(text)
        assert state.probability == 0.0

    def test_invalid_confidence_defaults_to_low(self) -> None:
        text = json.dumps({"probability": 0.5, "confidence": "extreme"})
        state = _parse_belief_state(text)
        assert state.confidence == "low"

    def test_json_embedded_in_text(self) -> None:
        text = 'Here is my analysis:\n\n' + _make_belief_response(0.3) + '\n\nDone.'
        state = _parse_belief_state(text)
        assert state.probability == 0.3

    def test_no_json_raises(self) -> None:
        with pytest.raises(ValueError, match="No JSON"):
            _parse_belief_state("Just some plain text with no JSON.")

    def test_invalid_json_raises(self) -> None:
        with pytest.raises((ValueError, json.JSONDecodeError)):
            _parse_belief_state("{not valid json}")

    def test_missing_fields_use_defaults(self) -> None:
        text = json.dumps({"probability": 0.6})
        state = _parse_belief_state(text)
        assert state.probability == 0.6
        assert state.confidence == "low"
        assert state.evidence_for == []
        assert state.evidence_against == []

    def test_evidence_not_list_coerced(self) -> None:
        text = json.dumps({
            "probability": 0.5,
            "evidence_for": "single item",
            "evidence_against": 42,
        })
        state = _parse_belief_state(text)
        assert state.evidence_for == ["single item"]
        assert state.evidence_against == ["42"]


class TestAggregateForecast:
    def test_single_value(self) -> None:
        result = _aggregate_forecasts([0.7])
        assert abs(result - 0.7) < 0.01

    def test_symmetric_values(self) -> None:
        result = _aggregate_forecasts([0.3, 0.7])
        assert abs(result - 0.5) < 0.01

    def test_all_same(self) -> None:
        result = _aggregate_forecasts([0.6, 0.6, 0.6])
        assert abs(result - 0.6) < 0.01

    def test_extreme_values_handled(self) -> None:
        result = _aggregate_forecasts([0.01, 0.99])
        assert 0.0 < result < 1.0

    def test_output_in_range(self) -> None:
        result = _aggregate_forecasts([0.1, 0.2, 0.9, 0.95])
        assert 0.0 < result < 1.0


def _mock_acompletion(response_text: str) -> AsyncMock:
    mock = AsyncMock()
    message = MagicMock()
    message.content = response_text
    choice = MagicMock()
    choice.message = message
    mock.return_value = MagicMock(choices=[choice])
    return mock


class TestSingleTrial:
    @pytest.mark.asyncio
    async def test_single_trial_returns_final_probability(self) -> None:
        responses = [
            _make_belief_response(0.6, "low"),
            _make_belief_response(0.65, "medium"),
            _make_belief_response(0.7, "high"),
        ]
        call_count = 0

        async def mock_acompletion(**kwargs: object) -> MagicMock:
            nonlocal call_count
            resp = responses[min(call_count, len(responses) - 1)]
            call_count += 1
            message = MagicMock()
            message.content = resp
            choice = MagicMock()
            choice.message = message
            return MagicMock(choices=[choice])

        q = _make_question()
        with patch("belief_forecaster.litellm.acompletion", side_effect=mock_acompletion), \
             patch("belief_forecaster._ensure_vertex_credentials"):
            result = await _single_trial(q, n_iterations=3)

        assert result == 0.7
        assert call_count == 3

    @pytest.mark.asyncio
    async def test_single_trial_handles_parse_error_gracefully(self) -> None:
        responses = [
            _make_belief_response(0.6, "low"),
            "This is not JSON at all",
        ]
        call_count = 0

        async def mock_acompletion(**kwargs: object) -> MagicMock:
            nonlocal call_count
            resp = responses[min(call_count, len(responses) - 1)]
            call_count += 1
            message = MagicMock()
            message.content = resp
            choice = MagicMock()
            choice.message = message
            return MagicMock(choices=[choice])

        q = _make_question()
        with patch("belief_forecaster.litellm.acompletion", side_effect=mock_acompletion), \
             patch("belief_forecaster._ensure_vertex_credentials"):
            result = await _single_trial(q, n_iterations=3)

        assert result == 0.6
        assert call_count == 2

    @pytest.mark.asyncio
    async def test_single_trial_handles_api_error(self) -> None:
        async def mock_acompletion(**kwargs: object) -> None:
            raise Exception("API error")

        q = _make_question()
        with patch("belief_forecaster.litellm.acompletion", side_effect=mock_acompletion), \
             patch("belief_forecaster._ensure_vertex_credentials"):
            result = await _single_trial(q, n_iterations=2)

        assert result == 0.5


class TestBeliefForecast:
    @pytest.mark.asyncio
    async def test_multi_trial_aggregation(self) -> None:
        trial_results = [0.6, 0.7, 0.65, 0.72, 0.68]
        call_idx = 0

        async def mock_single_trial(
            question: Question, **kwargs: object,
        ) -> float:
            nonlocal call_idx
            result = trial_results[call_idx % len(trial_results)]
            call_idx += 1
            return result

        q = _make_question()
        with patch("belief_forecaster._single_trial", side_effect=mock_single_trial):
            result = await belief_forecast(q, n_trials=5, n_iterations=3)

        assert 0.5 < result < 0.8
        expected = _aggregate_forecasts(trial_results)
        assert abs(result - expected) < 0.01

    @pytest.mark.asyncio
    async def test_single_trial_returns_directly(self) -> None:
        async def mock_single_trial(
            question: Question, **kwargs: object,
        ) -> float:
            return 0.73

        q = _make_question()
        with patch("belief_forecaster._single_trial", side_effect=mock_single_trial):
            result = await belief_forecast(q, n_trials=1, n_iterations=3)

        assert result == 0.73

    @pytest.mark.asyncio
    async def test_result_in_valid_range(self) -> None:
        async def mock_single_trial(
            question: Question, **kwargs: object,
        ) -> float:
            return 0.9

        q = _make_question()
        with patch("belief_forecaster._single_trial", side_effect=mock_single_trial):
            result = await belief_forecast(q, n_trials=3)

        assert 0.0 <= result <= 1.0


class TestBeliefForecastMultiHorizon:
    @pytest.mark.asyncio
    async def test_multi_horizon_returns_list(self) -> None:
        call_count = 0

        async def mock_belief_forecast(
            question: Question, **kwargs: object,
        ) -> float:
            nonlocal call_count
            call_count += 1
            return 0.5 + call_count * 0.05

        q = _make_question(
            source="fred",
            resolution_dates=["2026-03-01", "2026-06-01", "2026-09-01"],
        )
        with patch("belief_forecaster.belief_forecast", side_effect=mock_belief_forecast):
            result = await belief_forecast_multi_horizon(
                q,
                resolution_dates=["2026-03-01", "2026-06-01", "2026-09-01"],
            )

        assert result is not None
        assert len(result) == 3
        assert all(0.0 <= p <= 1.0 for p in result)
