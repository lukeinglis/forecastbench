"""Tests for ensemble forecasting (multi-model and its_hub self-consistency)."""

from __future__ import annotations

import asyncio
import math
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from ensemble import (
    _aggregate_forecasts,
    _arithmetic_mean,
    _geometric_mean_odds,
    _trimmed_mean,
    aggregate_predictions,
)
from fetch_data import Question


# ---------------------------------------------------------------------------
# Helpers for multi-model ensemble tests
# ---------------------------------------------------------------------------

def _make_question(
    source: str = "metaculus",
    question_id: str = "q1",
    resolution_dates: list[str] | None = None,
) -> Question:
    return Question(
        id=question_id,
        source=source,
        question="Will X happen?",
        background="Background info",
        resolution_criteria="Resolves YES if X.",
        freeze_datetime="2024-06-15",
        resolution_dates=resolution_dates,
    )


def _mock_response(content: str) -> MagicMock:
    resp = MagicMock()
    resp.choices = [MagicMock()]
    resp.choices[0].message.content = content
    return resp


# ---------------------------------------------------------------------------
# Helpers / fixtures for its_hub ensemble tests
# ---------------------------------------------------------------------------

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


# ===================================================================
# Multi-model ensemble tests (from ensemble.py)
# ===================================================================

class TestEnsembleModelsEnvVar:
    def test_default_models(self) -> None:
        import ensemble
        assert len(ensemble.ENSEMBLE_MODELS) >= 2

    def test_env_var_single_model(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("FORECAST_ENSEMBLE_MODELS", "openai/gpt-4o")
        models = "openai/gpt-4o".split(",")
        assert models == ["openai/gpt-4o"]

    def test_env_var_multiple_models(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("FORECAST_ENSEMBLE_MODELS", "openai/gpt-4o,anthropic/claude-3-opus,google/gemini-pro")
        models = "openai/gpt-4o,anthropic/claude-3-opus,google/gemini-pro".split(",")
        assert models == ["openai/gpt-4o", "anthropic/claude-3-opus", "google/gemini-pro"]


class TestMultiModelEnsembleForecast:
    def test_mean_of_two_models(self) -> None:
        from ensemble import ensemble_forecast

        q = _make_question(source="metaculus")
        resp1 = _mock_response("*0.7*")
        resp2 = _mock_response("*0.3*")

        with patch("ensemble.ENSEMBLE_MODELS", ["model_a", "model_b"]), \
             patch("ensemble.litellm") as mock_litellm:
            mock_litellm.acompletion = AsyncMock(side_effect=[resp1, resp2])
            result = asyncio.run(ensemble_forecast(q))

        assert result == pytest.approx(0.5, abs=1e-6)

    def test_single_model_fallback_on_failure(self) -> None:
        from ensemble import ensemble_forecast

        q = _make_question(source="polymarket")
        resp_ok = _mock_response("*0.8*")

        with patch("ensemble.ENSEMBLE_MODELS", ["model_a", "model_b"]), \
             patch("ensemble.litellm") as mock_litellm:
            mock_litellm.acompletion = AsyncMock(
                side_effect=[resp_ok, RuntimeError("API timeout")]
            )
            result = asyncio.run(ensemble_forecast(q))

        assert result == pytest.approx(0.8, abs=1e-6)

    def test_all_models_fail_raises(self) -> None:
        from ensemble import ensemble_forecast

        q = _make_question(source="manifold")

        with patch("ensemble.ENSEMBLE_MODELS", ["model_a", "model_b"]), \
             patch("ensemble.litellm") as mock_litellm:
            mock_litellm.acompletion = AsyncMock(
                side_effect=[RuntimeError("fail"), RuntimeError("fail")]
            )
            with pytest.raises(ValueError, match="All .* ensemble models failed"):
                asyncio.run(ensemble_forecast(q))

    def test_three_model_aggregated(self) -> None:
        from ensemble import ensemble_forecast

        q = _make_question(source="infer")
        resp1 = _mock_response("*0.3*")
        resp2 = _mock_response("*0.6*")
        resp3 = _mock_response("*0.9*")

        expected = _aggregate_forecasts([0.3, 0.6, 0.9])
        with patch("ensemble.ENSEMBLE_MODELS", ["m1", "m2", "m3"]), \
             patch("ensemble.litellm") as mock_litellm:
            mock_litellm.acompletion = AsyncMock(side_effect=[resp1, resp2, resp3])
            result = asyncio.run(ensemble_forecast(q))

        assert result == pytest.approx(expected, abs=1e-6)
        assert result == pytest.approx(0.6, abs=1e-6)


class TestEventSourceSingleModel:
    def test_acled_uses_primary_model_only(self) -> None:
        from ensemble import ensemble_forecast

        q = _make_question(source="acled")
        resp = _mock_response("*0.65*")

        with patch("ensemble.ENSEMBLE_MODELS", ["model_a", "model_b"]), \
             patch("ensemble.litellm") as mock_litellm, \
             patch("ensemble._ensure_vertex_credentials"):
            mock_litellm.acompletion = AsyncMock(return_value=resp)
            result = asyncio.run(ensemble_forecast(q))

        assert result == pytest.approx(0.65, abs=1e-6)
        mock_litellm.acompletion.assert_called_once()

    def test_wikipedia_uses_primary_model_only(self) -> None:
        from ensemble import ensemble_forecast

        q = _make_question(source="wikipedia")
        resp = _mock_response("*0.42*")

        with patch("ensemble.ENSEMBLE_MODELS", ["model_a", "model_b"]), \
             patch("ensemble.litellm") as mock_litellm, \
             patch("ensemble._ensure_vertex_credentials"):
            mock_litellm.acompletion = AsyncMock(return_value=resp)
            result = asyncio.run(ensemble_forecast(q))

        assert result == pytest.approx(0.42, abs=1e-6)
        mock_litellm.acompletion.assert_called_once()

    def test_market_source_uses_all_models(self) -> None:
        from ensemble import ensemble_forecast

        q = _make_question(source="metaculus")
        resp1 = _mock_response("*0.4*")
        resp2 = _mock_response("*0.6*")

        with patch("ensemble.ENSEMBLE_MODELS", ["model_a", "model_b"]), \
             patch("ensemble.litellm") as mock_litellm:
            mock_litellm.acompletion = AsyncMock(side_effect=[resp1, resp2])
            result = asyncio.run(ensemble_forecast(q))

        assert mock_litellm.acompletion.call_count == 2
        assert result == pytest.approx(0.5, abs=1e-6)

    def test_source_kwarg_overrides_question_source(self) -> None:
        from ensemble import ensemble_forecast

        q = _make_question(source="metaculus")
        resp = _mock_response("*0.55*")

        with patch("ensemble.ENSEMBLE_MODELS", ["model_a", "model_b"]), \
             patch("ensemble.litellm") as mock_litellm, \
             patch("ensemble._ensure_vertex_credentials"):
            mock_litellm.acompletion = AsyncMock(return_value=resp)
            asyncio.run(ensemble_forecast(q, source="acled"))

        mock_litellm.acompletion.assert_called_once()


class TestMultiModelEnsembleMultiHorizon:
    def test_aggregates_per_horizon(self) -> None:
        from ensemble import ensemble_forecast_multi_horizon

        q = _make_question(source="fred", resolution_dates=["2024-07-01", "2024-08-01"])

        resp1 = _mock_response("*0.3* *0.5*")
        resp2 = _mock_response("*0.7* *0.9*")

        expected_h0 = _aggregate_forecasts([0.3, 0.7])
        expected_h1 = _aggregate_forecasts([0.5, 0.9])

        with patch("ensemble.ENSEMBLE_MODELS", ["model_a", "model_b"]), \
             patch("ensemble.litellm") as mock_litellm:
            mock_litellm.acompletion = AsyncMock(side_effect=[resp1, resp2])
            result = asyncio.run(
                ensemble_forecast_multi_horizon(q, resolution_dates=["2024-07-01", "2024-08-01"])
            )

        assert result is not None
        assert len(result) == 2
        assert result[0] == pytest.approx(expected_h0, abs=1e-6)
        assert result[1] == pytest.approx(expected_h1, abs=1e-6)

    def test_single_model_failure_uses_other(self) -> None:
        from ensemble import ensemble_forecast_multi_horizon

        q = _make_question(source="dbnomics", resolution_dates=["2024-07-01", "2024-08-01"])

        resp_ok = _mock_response("*0.4* *0.6*")

        with patch("ensemble.ENSEMBLE_MODELS", ["model_a", "model_b"]), \
             patch("ensemble.litellm") as mock_litellm:
            mock_litellm.acompletion = AsyncMock(
                side_effect=[resp_ok, RuntimeError("timeout")]
            )
            result = asyncio.run(
                ensemble_forecast_multi_horizon(q, resolution_dates=["2024-07-01", "2024-08-01"])
            )

        assert result is not None
        assert result[0] == pytest.approx(0.4, abs=1e-6)
        assert result[1] == pytest.approx(0.6, abs=1e-6)

    def test_all_models_fail_returns_none(self) -> None:
        from ensemble import ensemble_forecast_multi_horizon

        q = _make_question(source="fred", resolution_dates=["2024-07-01"])

        with patch("ensemble.ENSEMBLE_MODELS", ["model_a", "model_b"]), \
             patch("ensemble.litellm") as mock_litellm:
            mock_litellm.acompletion = AsyncMock(
                side_effect=[RuntimeError("fail"), RuntimeError("fail")]
            )
            result = asyncio.run(
                ensemble_forecast_multi_horizon(q, resolution_dates=["2024-07-01"])
            )

        assert result is None

    def test_event_source_uses_primary_only(self) -> None:
        from ensemble import ensemble_forecast_multi_horizon

        q = _make_question(source="acled", resolution_dates=["2024-07-01", "2024-08-01"])
        resp = _mock_response("*0.3* *0.7*")

        with patch("ensemble.ENSEMBLE_MODELS", ["model_a", "model_b"]), \
             patch("ensemble.litellm") as mock_litellm, \
             patch("ensemble._ensure_vertex_credentials"):
            mock_litellm.acompletion = AsyncMock(return_value=resp)
            result = asyncio.run(
                ensemble_forecast_multi_horizon(q, resolution_dates=["2024-07-01", "2024-08-01"])
            )

        assert result is not None
        assert len(result) == 2
        mock_litellm.acompletion.assert_called_once()


# ===================================================================
# its_hub ensemble tests (from baseline_agent)
# ===================================================================

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


class TestItshubEnsembleForecast:
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


class TestItshubEnsembleForecastMultiHorizon:
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


# ===================================================================
# Aggregation tests (multi-model)
# ===================================================================

class TestAggregateForecasts:
    def test_default_is_arithmetic_mean(self) -> None:
        preds = [0.3, 0.7]
        result = _aggregate_forecasts(preds)
        assert result == pytest.approx(0.5, abs=1e-6)

    def test_single_prediction_returned_unchanged(self) -> None:
        assert _aggregate_forecasts([0.73]) == 0.73
        assert _aggregate_forecasts([0.01]) == 0.01
        assert _aggregate_forecasts([0.99]) == 0.99

    def test_extremized_mode_gamma_1_is_geometric_mean_of_odds(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setattr("ensemble.ENSEMBLE_AGGREGATION", "extremized")
        preds = [0.3, 0.7, 0.8]
        result = _aggregate_forecasts(preds, gamma=1.0)
        odds = [p / (1 - p) for p in preds]
        geo_mean_odds = math.exp(sum(math.log(o) for o in odds) / len(odds))
        expected = geo_mean_odds / (1 + geo_mean_odds)
        assert result == pytest.approx(expected, abs=1e-6)

    def test_extremized_mode_pushes_away_from_half(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setattr("ensemble.ENSEMBLE_AGGREGATION", "extremized")
        preds = [0.6, 0.7, 0.8]
        result_neutral = _aggregate_forecasts(preds, gamma=1.0)
        result_extremized = _aggregate_forecasts(preds, gamma=1.5)
        assert result_neutral > 0.5
        assert result_extremized > result_neutral

    def test_extremized_mode_higher_gamma_more_extreme(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setattr("ensemble.ENSEMBLE_AGGREGATION", "extremized")
        preds = [0.6, 0.7, 0.8]
        result_1 = _aggregate_forecasts(preds, gamma=1.0)
        result_2 = _aggregate_forecasts(preds, gamma=2.0)
        assert result_2 > result_1
        assert result_2 > 0.5

    def test_extremized_mode_below_half_pushed_lower(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setattr("ensemble.ENSEMBLE_AGGREGATION", "extremized")
        preds = [0.2, 0.3, 0.4]
        result_1 = _aggregate_forecasts(preds, gamma=1.0)
        result_2 = _aggregate_forecasts(preds, gamma=2.0)
        assert result_2 < result_1
        assert result_2 < 0.5

    def test_geometric_mean_odds_via_env(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setattr("ensemble.ENSEMBLE_AGGREGATION", "geometric_mean_odds")
        preds = [0.3, 0.7, 0.8]
        result = _aggregate_forecasts(preds)
        expected = _geometric_mean_odds(preds)
        assert result == pytest.approx(expected, abs=1e-6)

    def test_trimmed_mean_via_env(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setattr("ensemble.ENSEMBLE_AGGREGATION", "trimmed_mean")
        preds = [0.1, 0.4, 0.6, 0.9]
        result = _aggregate_forecasts(preds)
        expected = _trimmed_mean(preds)
        assert result == pytest.approx(expected, abs=1e-6)


# ===================================================================
# Geometric mean of odds tests
# ===================================================================

class TestGeometricMeanOdds:
    def test_identity_at_half(self) -> None:
        assert _geometric_mean_odds([0.5, 0.5, 0.5]) == pytest.approx(0.5, abs=1e-6)

    def test_known_values(self) -> None:
        result = _geometric_mean_odds([0.8, 0.8, 0.8])
        assert result == pytest.approx(0.8, abs=1e-6)

    def test_emphasizes_agreement(self) -> None:
        agreeing = [0.8, 0.8, 0.8]
        mixed = [0.6, 0.8, 1.0]
        result_agree = _geometric_mean_odds(agreeing)
        result_mixed = _geometric_mean_odds(mixed)
        mean_agree = _arithmetic_mean(agreeing)
        mean_mixed = _arithmetic_mean(mixed)
        assert result_agree >= mean_agree - 1e-6
        assert abs(result_agree - mean_agree) < abs(result_mixed - mean_mixed) + 0.05

    def test_symmetric_around_half(self) -> None:
        preds_high = [0.7, 0.8]
        preds_low = [0.3, 0.2]
        result_high = _geometric_mean_odds(preds_high)
        result_low = _geometric_mean_odds(preds_low)
        assert result_high == pytest.approx(1.0 - result_low, abs=1e-6)

    def test_extreme_probabilities_clamped(self) -> None:
        result = _geometric_mean_odds([0.001, 0.999])
        assert 0.02 <= result <= 0.98

    def test_single_prediction(self) -> None:
        result = _geometric_mean_odds([0.7])
        assert 0.02 <= result <= 0.98

    def test_output_bounded(self) -> None:
        result_low = _geometric_mean_odds([0.01, 0.01, 0.01])
        result_high = _geometric_mean_odds([0.99, 0.99, 0.99])
        assert result_low >= 0.02
        assert result_high <= 0.98

    def test_two_predictions(self) -> None:
        result = _geometric_mean_odds([0.3, 0.7])
        assert result == pytest.approx(0.5, abs=0.05)


# ===================================================================
# Trimmed mean tests
# ===================================================================

class TestTrimmedMean:
    def test_drops_extremes_with_four_values(self) -> None:
        preds = [0.1, 0.4, 0.6, 0.9]
        result = _trimmed_mean(preds)
        expected = (0.4 + 0.6) / 2
        assert result == pytest.approx(expected, abs=1e-6)

    def test_drops_extremes_with_five_values(self) -> None:
        preds = [0.1, 0.3, 0.5, 0.7, 0.9]
        result = _trimmed_mean(preds)
        expected = (0.3 + 0.5 + 0.7) / 3
        assert result == pytest.approx(expected, abs=1e-6)

    def test_fallback_to_mean_with_two(self) -> None:
        preds = [0.3, 0.7]
        result = _trimmed_mean(preds)
        expected = 0.5
        assert result == pytest.approx(expected, abs=1e-6)

    def test_fallback_to_mean_with_three(self) -> None:
        preds = [0.2, 0.5, 0.8]
        result = _trimmed_mean(preds)
        expected = 0.5
        assert result == pytest.approx(expected, abs=1e-6)

    def test_single_prediction(self) -> None:
        result = _trimmed_mean([0.65])
        assert result == pytest.approx(0.65, abs=1e-6)

    def test_unsorted_input(self) -> None:
        preds = [0.9, 0.1, 0.6, 0.4]
        result = _trimmed_mean(preds)
        expected = (0.4 + 0.6) / 2
        assert result == pytest.approx(expected, abs=1e-6)

    def test_extreme_probabilities(self) -> None:
        preds = [0.01, 0.4, 0.6, 0.99]
        result = _trimmed_mean(preds)
        expected = (0.4 + 0.6) / 2
        assert result == pytest.approx(expected, abs=1e-6)


# ===================================================================
# aggregate_predictions dispatch tests
# ===================================================================

class TestAggregatePredictions:
    def test_single_prediction_passthrough(self) -> None:
        for method in ["mean", "geometric_mean_odds", "trimmed_mean", "extremized"]:
            assert aggregate_predictions([0.73], method=method) == 0.73

    def test_mean_method(self) -> None:
        result = aggregate_predictions([0.3, 0.7], method="mean")
        assert result == pytest.approx(0.5, abs=1e-6)

    def test_geometric_mean_odds_method(self) -> None:
        result = aggregate_predictions([0.5, 0.5, 0.5], method="geometric_mean_odds")
        assert result == pytest.approx(0.5, abs=1e-6)

    def test_trimmed_mean_method(self) -> None:
        result = aggregate_predictions([0.1, 0.4, 0.6, 0.9], method="trimmed_mean")
        assert result == pytest.approx(0.5, abs=1e-6)

    def test_extremized_method(self) -> None:
        result = aggregate_predictions([0.6, 0.8], method="extremized", gamma=1.5)
        assert result > 0.7

    def test_unknown_method_falls_back_to_mean(self) -> None:
        result = aggregate_predictions([0.3, 0.7], method="unknown")
        assert result == pytest.approx(0.5, abs=1e-6)

    def test_multi_horizon_independent_aggregation(self) -> None:
        horizons = [[0.3, 0.7], [0.2, 0.8], [0.4, 0.6]]
        n_horizons = len(horizons[0])
        n_models = len(horizons)
        for h in range(n_horizons):
            preds = [horizons[m][h] for m in range(n_models)]
            result = aggregate_predictions(preds, method="geometric_mean_odds")
            assert 0.02 <= result <= 0.98
