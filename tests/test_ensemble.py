"""Tests for multi-model ensemble forecaster."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from fetch_data import Question


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


class TestEnsembleForecast:
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

    def test_three_model_mean(self) -> None:
        from ensemble import ensemble_forecast

        q = _make_question(source="infer")
        resp1 = _mock_response("*0.3*")
        resp2 = _mock_response("*0.6*")
        resp3 = _mock_response("*0.9*")

        with patch("ensemble.ENSEMBLE_MODELS", ["m1", "m2", "m3"]), \
             patch("ensemble.litellm") as mock_litellm:
            mock_litellm.acompletion = AsyncMock(side_effect=[resp1, resp2, resp3])
            result = asyncio.run(ensemble_forecast(q))

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
            result = asyncio.run(ensemble_forecast(q, source="acled"))

        mock_litellm.acompletion.assert_called_once()


class TestEnsembleMultiHorizon:
    def test_aggregates_per_horizon(self) -> None:
        from ensemble import ensemble_forecast_multi_horizon

        q = _make_question(source="fred", resolution_dates=["2024-07-01", "2024-08-01"])

        resp1 = _mock_response("*0.3* *0.5*")
        resp2 = _mock_response("*0.7* *0.9*")

        with patch("ensemble.ENSEMBLE_MODELS", ["model_a", "model_b"]), \
             patch("ensemble.litellm") as mock_litellm:
            mock_litellm.acompletion = AsyncMock(side_effect=[resp1, resp2])
            result = asyncio.run(
                ensemble_forecast_multi_horizon(q, resolution_dates=["2024-07-01", "2024-08-01"])
            )

        assert result is not None
        assert len(result) == 2
        assert result[0] == pytest.approx(0.5, abs=1e-6)
        assert result[1] == pytest.approx(0.7, abs=1e-6)

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
