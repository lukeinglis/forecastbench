"""Tests for hybrid forecaster routing logic."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, patch  # noqa: F811 — patch used in fixture and directly

import pytest

from fetch_data import Question


def _make_question(source: str, qid: str = "q1") -> Question:
    return Question(
        id=qid,
        source=source,
        question="Will X happen?",
        background="",
        resolution_criteria="",
    )


@pytest.fixture
def _mock_forecasters():
    with (
        patch("hybrid_forecaster.belief_forecast", new_callable=AsyncMock, return_value=0.7) as belief,
        patch("hybrid_forecaster.aforecast", new_callable=AsyncMock, return_value=0.3) as baseline,
        patch("hybrid_forecaster.belief_forecast_multi_horizon", new_callable=AsyncMock, return_value=[0.7, 0.8]) as belief_multi,
        patch("hybrid_forecaster.aforecast_multi_horizon", new_callable=AsyncMock, return_value=[0.3, 0.4]) as baseline_multi,
    ):
        yield {
            "belief": belief,
            "baseline": baseline,
            "belief_multi": belief_multi,
            "baseline_multi": baseline_multi,
        }


class TestRouting:
    def test_acled_routes_to_belief(self, _mock_forecasters):
        from hybrid_forecaster import hybrid_forecast

        result = asyncio.run(hybrid_forecast(_make_question("acled")))
        assert result == 0.7
        _mock_forecasters["belief"].assert_awaited_once()
        _mock_forecasters["baseline"].assert_not_awaited()

    def test_fred_routes_to_belief(self, _mock_forecasters):
        from hybrid_forecaster import hybrid_forecast

        result = asyncio.run(hybrid_forecast(_make_question("fred")))
        assert result == 0.7
        _mock_forecasters["belief"].assert_awaited_once()

    def test_polymarket_routes_to_baseline(self, _mock_forecasters):
        from hybrid_forecaster import hybrid_forecast

        result = asyncio.run(hybrid_forecast(_make_question("polymarket")))
        assert result == 0.3
        _mock_forecasters["baseline"].assert_awaited_once()
        _mock_forecasters["belief"].assert_not_awaited()

    def test_dbnomics_routes_to_baseline(self, _mock_forecasters):
        from hybrid_forecaster import hybrid_forecast

        result = asyncio.run(hybrid_forecast(_make_question("dbnomics")))
        assert result == 0.3
        _mock_forecasters["baseline"].assert_awaited_once()
        _mock_forecasters["belief"].assert_not_awaited()

    def test_case_insensitive(self, _mock_forecasters):
        from hybrid_forecaster import hybrid_forecast

        result = asyncio.run(hybrid_forecast(_make_question("ACLED")))
        assert result == 0.7
        _mock_forecasters["belief"].assert_awaited_once()

    def test_source_override_via_kwarg(self, _mock_forecasters):
        from hybrid_forecaster import hybrid_forecast

        result = asyncio.run(hybrid_forecast(_make_question("polymarket"), source="fred"))
        assert result == 0.7
        _mock_forecasters["belief"].assert_awaited_once()


class TestMultiHorizon:
    def test_belief_multi_horizon(self, _mock_forecasters):
        from hybrid_forecaster import hybrid_forecast_multi_horizon

        result = asyncio.run(
            hybrid_forecast_multi_horizon(_make_question("acled"), ["2026-01-01", "2026-02-01"])
        )
        assert result == [0.7, 0.8]
        _mock_forecasters["belief_multi"].assert_awaited_once()

    def test_baseline_multi_horizon(self, _mock_forecasters):
        from hybrid_forecaster import hybrid_forecast_multi_horizon

        result = asyncio.run(
            hybrid_forecast_multi_horizon(_make_question("manifold"), ["2026-01-01", "2026-02-01"])
        )
        assert result == [0.3, 0.4]
        _mock_forecasters["baseline_multi"].assert_awaited_once()


class TestEnvOverride:
    def test_env_override_belief_sources(self, monkeypatch):
        monkeypatch.setenv("FORECAST_BELIEF_SOURCES", "polymarket,infer")
        import importlib
        import hybrid_forecaster

        importlib.reload(hybrid_forecaster)
        try:
            with patch.object(hybrid_forecaster, "belief_forecast", new_callable=AsyncMock, return_value=0.7) as belief:
                result = asyncio.run(hybrid_forecaster.hybrid_forecast(_make_question("polymarket")))
                assert result == 0.7
                belief.assert_awaited_once()
        finally:
            monkeypatch.delenv("FORECAST_BELIEF_SOURCES")
            importlib.reload(hybrid_forecaster)
