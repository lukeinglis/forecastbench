"""Tests for statistical blend integration in baseline_agent."""

from __future__ import annotations

import os
from unittest.mock import patch, MagicMock

from fetch_data import Question


def _make_timeseries_question(
    source: str = "fred",
    freeze_value: float = 5.0,
) -> Question:
    return Question(
        id="FEDFUNDS",
        source=source,
        question="Will the federal funds rate exceed 6.0 by the resolution date?",
        background="Interest rate question",
        resolution_criteria="Resolves YES if the rate goes above 6.0.",
        freeze_datetime="2024-06-15",
        forecast_due_date="2024-06-15",
        freeze_datetime_value=freeze_value,
        freeze_datetime_value_explanation="Current rate",
        resolution_dates=["2024-07-15", "2024-08-15", "2024-09-15"],
    )


class TestStatisticalBlendDisabled:
    """FORECAST_STATISTICAL_BLEND=0 produces identical output (no regression)."""

    def test_blend_zero_returns_llm_prob(self) -> None:
        import baseline_agent

        with patch.object(baseline_agent, "STATISTICAL_BLEND", 0):
            q = _make_timeseries_question()
            result = baseline_agent._compute_statistical_blend(
                0.7, q, "2024-07-15", "2024-06-15",
            )
            assert result == 0.7

    def test_blend_default_is_zero(self) -> None:
        with patch.dict(os.environ, {}, clear=False):
            env = os.environ.copy()
            env.pop("FORECAST_STATISTICAL_BLEND", None)
            with patch.dict(os.environ, env, clear=True):
                import importlib
                import baseline_agent

                importlib.reload(baseline_agent)
                assert baseline_agent.STATISTICAL_BLEND == 0


class TestStatisticalBlendMath:
    """Test that blending math is correct."""

    @patch("statistical_baseline.get_statistical_forecast")
    @patch("timeseries_rag.get_raw_historical_data")
    def test_blend_60_40(
        self, mock_hist: MagicMock, mock_stat: MagicMock,
    ) -> None:
        mock_hist.return_value = {"2024-06-01": 5.0, "2024-06-10": 5.1}
        mock_stat.return_value = 0.3

        import baseline_agent

        with patch.object(baseline_agent, "STATISTICAL_BLEND", 0.6):
            q = _make_timeseries_question()
            result = baseline_agent._compute_statistical_blend(
                0.7, q, "2024-07-15", "2024-06-15",
            )
            expected = 0.6 * 0.3 + 0.4 * 0.7
            assert abs(result - expected) < 1e-9

    @patch("statistical_baseline.get_statistical_forecast")
    @patch("timeseries_rag.get_raw_historical_data")
    def test_pure_statistical(
        self, mock_hist: MagicMock, mock_stat: MagicMock,
    ) -> None:
        mock_hist.return_value = {"2024-06-01": 5.0}
        mock_stat.return_value = 0.4

        import baseline_agent

        with patch.object(baseline_agent, "STATISTICAL_BLEND", 1.0):
            q = _make_timeseries_question()
            result = baseline_agent._compute_statistical_blend(
                0.8, q, "2024-07-15", "2024-06-15",
            )
            assert abs(result - 0.4) < 1e-9

    @patch("statistical_baseline.get_statistical_forecast")
    @patch("timeseries_rag.get_raw_historical_data")
    def test_pure_llm(
        self, mock_hist: MagicMock, mock_stat: MagicMock,
    ) -> None:
        mock_hist.return_value = {"2024-06-01": 5.0}
        mock_stat.return_value = 0.4

        import baseline_agent

        with patch.object(baseline_agent, "STATISTICAL_BLEND", 0.0):
            q = _make_timeseries_question()
            result = baseline_agent._compute_statistical_blend(
                0.8, q, "2024-07-15", "2024-06-15",
            )
            assert result == 0.8


class TestStatisticalBlendFallback:
    """When statistical forecast returns None, use LLM only."""

    @patch("statistical_baseline.get_statistical_forecast")
    @patch("timeseries_rag.get_raw_historical_data")
    def test_stat_none_uses_llm(
        self, mock_hist: MagicMock, mock_stat: MagicMock,
    ) -> None:
        mock_hist.return_value = None
        mock_stat.return_value = None

        import baseline_agent

        with patch.object(baseline_agent, "STATISTICAL_BLEND", 0.6):
            q = _make_timeseries_question()
            result = baseline_agent._compute_statistical_blend(
                0.75, q, "2024-07-15", "2024-06-15",
            )
            assert result == 0.75

    @patch("statistical_baseline.get_statistical_forecast")
    @patch("timeseries_rag.get_raw_historical_data")
    def test_stat_none_with_historical_data(
        self, mock_hist: MagicMock, mock_stat: MagicMock,
    ) -> None:
        mock_hist.return_value = {"2024-06-01": 5.0, "2024-06-10": 5.1}
        mock_stat.return_value = None

        import baseline_agent

        with patch.object(baseline_agent, "STATISTICAL_BLEND", 0.6):
            q = _make_timeseries_question()
            result = baseline_agent._compute_statistical_blend(
                0.65, q, "2024-07-15", "2024-06-15",
            )
            assert result == 0.65


class TestStatisticalBlendHorizon:
    """Each resolution date gets its own horizon_days and statistical forecast."""

    @patch("statistical_baseline.get_statistical_forecast")
    @patch("timeseries_rag.get_raw_historical_data")
    def test_different_horizons_different_forecasts(
        self, mock_hist: MagicMock, mock_stat: MagicMock,
    ) -> None:
        mock_hist.return_value = {"2024-06-01": 5.0, "2024-06-10": 5.1}

        call_horizons: list[int] = []

        def capture_horizon(question: Question, historical_data: dict | None, horizon_days: int) -> float:
            call_horizons.append(horizon_days)
            return 0.3 + 0.01 * horizon_days

        mock_stat.side_effect = capture_horizon

        import baseline_agent

        with patch.object(baseline_agent, "STATISTICAL_BLEND", 0.5):
            q = _make_timeseries_question()
            results = []
            for rd in ["2024-07-15", "2024-08-15", "2024-09-15"]:
                r = baseline_agent._compute_statistical_blend(
                    0.5, q, rd, "2024-06-15",
                )
                results.append(r)

            assert call_horizons == [30, 61, 92]
            assert len(set(results)) == 3

    @patch("statistical_baseline.get_statistical_forecast")
    @patch("timeseries_rag.get_raw_historical_data")
    def test_invalid_date_returns_llm(
        self, mock_hist: MagicMock, mock_stat: MagicMock,
    ) -> None:
        mock_hist.return_value = {}

        import baseline_agent

        with patch.object(baseline_agent, "STATISTICAL_BLEND", 0.6):
            q = _make_timeseries_question()
            result = baseline_agent._compute_statistical_blend(
                0.7, q, "invalid-date", "2024-06-15",
            )
            assert result == 0.7

    @patch("statistical_baseline.get_statistical_forecast")
    @patch("timeseries_rag.get_raw_historical_data")
    def test_no_forecast_due_date_uses_freeze(
        self, mock_hist: MagicMock, mock_stat: MagicMock,
    ) -> None:
        mock_hist.return_value = {"2024-06-01": 5.0}
        mock_stat.return_value = 0.4

        import baseline_agent

        with patch.object(baseline_agent, "STATISTICAL_BLEND", 0.5):
            q = _make_timeseries_question()
            result = baseline_agent._compute_statistical_blend(
                0.6, q, "2024-07-15", None,
            )
            expected = 0.5 * 0.4 + 0.5 * 0.6
            assert abs(result - expected) < 1e-9


class TestNonTimeseriesUnaffected:
    """Market/event sources are gated at the call site (not passed to blend)."""

    def test_market_source_not_in_timeseries_sources(self) -> None:
        from baseline_agent import TIMESERIES_SOURCES
        assert "metaculus" not in TIMESERIES_SOURCES
        assert "polymarket" not in TIMESERIES_SOURCES

    def test_timeseries_sources_are_in_set(self) -> None:
        from baseline_agent import TIMESERIES_SOURCES
        assert "fred" in TIMESERIES_SOURCES
        assert "dbnomics" in TIMESERIES_SOURCES
        assert "yfinance" in TIMESERIES_SOURCES
