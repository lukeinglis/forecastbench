"""Tests for its_hub ensemble integration in baseline_agent."""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


@pytest.fixture(autouse=True)
def _reset_ensemble_config(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("baseline_agent.ENSEMBLE_N", 3)
    monkeypatch.setattr("baseline_agent.ENSEMBLE_TEMP", 0.7)
    monkeypatch.setattr("baseline_agent.MODEL", "vertex_ai/claude-sonnet-4@20250514")
    monkeypatch.setattr("baseline_agent.MAX_TOKENS", 16384)
    monkeypatch.setattr("baseline_agent.VERTEX_LOCATION", "europe-west1")


def _make_completion_response(content: str) -> MagicMock:
    resp = MagicMock()
    resp.choices = [MagicMock()]
    resp.choices[0].message.content = content
    return resp


class TestLiteLLMAdapter:
    @pytest.mark.asyncio
    async def test_returns_correct_dict_format(self) -> None:
        from baseline_agent import LiteLLMAdapter

        adapter = LiteLLMAdapter("test-model", 1024, "us-central1")
        mock_resp = _make_completion_response("*0.75*")

        with patch("baseline_agent.litellm") as mock_litellm, \
             patch("baseline_agent._ensure_vertex_credentials"):
            mock_litellm.acompletion = AsyncMock(return_value=mock_resp)
            result = await adapter.agenerate_single(
                [{"role": "user", "content": "test"}],
                temperature=0.7,
            )

        assert isinstance(result, dict)
        assert result["role"] == "assistant"
        assert result["content"] == "*0.75*"

    @pytest.mark.asyncio
    async def test_does_not_include_thinking_key(self) -> None:
        from baseline_agent import LiteLLMAdapter

        adapter = LiteLLMAdapter("test-model", 1024, "us-central1")
        mock_resp = _make_completion_response("*0.5*")

        with patch("baseline_agent.litellm") as mock_litellm, \
             patch("baseline_agent._ensure_vertex_credentials"):
            mock_litellm.acompletion = AsyncMock(return_value=mock_resp)
            await adapter.agenerate_single(
                [{"role": "user", "content": "test"}],
                temperature=0.7,
            )

            call_kwargs = mock_litellm.acompletion.call_args[1]
            assert "thinking" not in call_kwargs

    @pytest.mark.asyncio
    async def test_passes_temperature_from_kwargs(self) -> None:
        from baseline_agent import LiteLLMAdapter

        adapter = LiteLLMAdapter("test-model", 1024, "us-central1")
        mock_resp = _make_completion_response("*0.5*")

        with patch("baseline_agent.litellm") as mock_litellm, \
             patch("baseline_agent._ensure_vertex_credentials"):
            mock_litellm.acompletion = AsyncMock(return_value=mock_resp)
            await adapter.agenerate_single(
                [{"role": "user", "content": "test"}],
                temperature=0.9,
            )

            call_kwargs = mock_litellm.acompletion.call_args[1]
            assert call_kwargs["temperature"] == 0.9

    @pytest.mark.asyncio
    async def test_uses_default_temp_when_not_in_kwargs(self) -> None:
        from baseline_agent import LiteLLMAdapter

        adapter = LiteLLMAdapter("test-model", 1024, "us-central1")
        mock_resp = _make_completion_response("*0.5*")

        with patch("baseline_agent.litellm") as mock_litellm, \
             patch("baseline_agent._ensure_vertex_credentials"):
            mock_litellm.acompletion = AsyncMock(return_value=mock_resp)
            await adapter.agenerate_single(
                [{"role": "user", "content": "test"}],
            )

            call_kwargs = mock_litellm.acompletion.call_args[1]
            assert call_kwargs["temperature"] == 0.7


class TestEnsembleForecast:
    @pytest.mark.asyncio
    async def test_averages_probabilities(self) -> None:
        from baseline_agent import _ensemble_forecast

        async def mock_agenerate(lm: Any, batch: Any, **kwargs: Any) -> list[dict[str, str]]:
            return [
                {"role": "assistant", "content": "*0.3*"},
                {"role": "assistant", "content": "*0.5*"},
                {"role": "assistant", "content": "*0.7*"},
            ]

        with patch("baseline_agent.LiteLLMAdapter"), \
             patch("its_hub.core.orchestrator.LMOrchestrator") as MockOrch:
            instance = MockOrch.return_value
            instance.agenerate = AsyncMock(side_effect=mock_agenerate)

            result = await _ensemble_forecast("test prompt")

        assert result is not None
        assert abs(result - 0.5) < 1e-9

    @pytest.mark.asyncio
    async def test_partial_failure_averages_successful(self) -> None:
        from baseline_agent import _ensemble_forecast

        async def mock_agenerate(lm: Any, batch: Any, **kwargs: Any) -> list[dict[str, str]]:
            return [
                {"role": "assistant", "content": "*0.3*"},
                {"role": "assistant", "content": "I cannot provide a probability"},
                {"role": "assistant", "content": "*0.7*"},
            ]

        with patch("baseline_agent.LiteLLMAdapter"), \
             patch("its_hub.core.orchestrator.LMOrchestrator") as MockOrch:
            instance = MockOrch.return_value
            instance.agenerate = AsyncMock(side_effect=mock_agenerate)

            result = await _ensemble_forecast("test prompt")

        assert result is not None
        assert abs(result - 0.5) < 1e-9

    @pytest.mark.asyncio
    async def test_total_failure_returns_none(self) -> None:
        from baseline_agent import _ensemble_forecast

        async def mock_agenerate(lm: Any, batch: Any, **kwargs: Any) -> list[dict[str, str]]:
            return [
                {"role": "assistant", "content": "no numbers here"},
                {"role": "assistant", "content": "still nothing"},
                {"role": "assistant", "content": "nope"},
            ]

        with patch("baseline_agent.LiteLLMAdapter"), \
             patch("its_hub.core.orchestrator.LMOrchestrator") as MockOrch:
            instance = MockOrch.return_value
            instance.agenerate = AsyncMock(side_effect=mock_agenerate)

            result = await _ensemble_forecast("test prompt")

        assert result is None


class TestEnsembleN1Bypass:
    @pytest.mark.asyncio
    async def test_ensemble_n1_skips_ensemble_path(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setattr("baseline_agent.ENSEMBLE_N", 1)

        from baseline_agent import aforecast
        from fetch_data import Question

        q = Question(
            id="test-q",
            question="Will it rain?",
            source="metaculus",
            resolution_criteria="Yes if rain",
            background="",
            freeze_datetime="2024-01-01",
        )

        mock_resp = _make_completion_response("*0.65*")

        with patch("baseline_agent._ensemble_forecast") as mock_ensemble, \
             patch("baseline_agent.litellm") as mock_litellm, \
             patch("baseline_agent._ensure_vertex_credentials"):
            mock_litellm.acompletion = AsyncMock(return_value=mock_resp)
            result = await aforecast(q)

        mock_ensemble.assert_not_called()
        assert abs(result - 0.65) < 1e-9


class TestEnsembleForecastMultiHorizon:
    @pytest.mark.asyncio
    async def test_averages_per_horizon(self) -> None:
        from baseline_agent import _ensemble_forecast_multi_horizon

        async def mock_agenerate(lm: Any, batch: Any, **kwargs: Any) -> list[dict[str, str]]:
            return [
                {"role": "assistant", "content": "*0.2* *0.4* *0.6*"},
                {"role": "assistant", "content": "*0.4* *0.6* *0.8*"},
                {"role": "assistant", "content": "*0.6* *0.8* *1.0*"},
            ]

        with patch("baseline_agent.LiteLLMAdapter"), \
             patch("its_hub.core.orchestrator.LMOrchestrator") as MockOrch:
            instance = MockOrch.return_value
            instance.agenerate = AsyncMock(side_effect=mock_agenerate)

            result = await _ensemble_forecast_multi_horizon(
                "test prompt", n_horizons=3, question_id="q1",
            )

        assert result is not None
        assert len(result) == 3
        assert abs(result[0] - 0.4) < 1e-9
        assert abs(result[1] - 0.6) < 1e-9
        assert abs(result[2] - 0.8) < 1e-9

    @pytest.mark.asyncio
    async def test_partial_failure_uses_successful_members(self) -> None:
        from baseline_agent import _ensemble_forecast_multi_horizon

        async def mock_agenerate(lm: Any, batch: Any, **kwargs: Any) -> list[dict[str, str]]:
            return [
                {"role": "assistant", "content": "*0.2* *0.4*"},
                {"role": "assistant", "content": "garbage response"},
                {"role": "assistant", "content": "*0.6* *0.8*"},
            ]

        with patch("baseline_agent.LiteLLMAdapter"), \
             patch("its_hub.core.orchestrator.LMOrchestrator") as MockOrch:
            instance = MockOrch.return_value
            instance.agenerate = AsyncMock(side_effect=mock_agenerate)

            result = await _ensemble_forecast_multi_horizon(
                "test prompt", n_horizons=2, question_id="q1",
            )

        assert result is not None
        assert len(result) == 2
        assert abs(result[0] - 0.4) < 1e-9
        assert abs(result[1] - 0.6) < 1e-9

    @pytest.mark.asyncio
    async def test_total_failure_returns_none(self) -> None:
        from baseline_agent import _ensemble_forecast_multi_horizon

        async def mock_agenerate(lm: Any, batch: Any, **kwargs: Any) -> list[dict[str, str]]:
            return [
                {"role": "assistant", "content": "no probs"},
                {"role": "assistant", "content": "still nothing"},
            ]

        with patch("baseline_agent.LiteLLMAdapter"), \
             patch("its_hub.core.orchestrator.LMOrchestrator") as MockOrch:
            instance = MockOrch.return_value
            instance.agenerate = AsyncMock(side_effect=mock_agenerate)

            result = await _ensemble_forecast_multi_horizon(
                "test prompt", n_horizons=3, question_id="q1",
            )

        assert result is None
