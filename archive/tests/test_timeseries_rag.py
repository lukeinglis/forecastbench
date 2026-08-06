"""Tests for timeseries_rag.py — historical data fetching, caching, formatting."""

from __future__ import annotations

import os
from datetime import date
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from timeseries_rag import (
    LOOKBACK_DAYS,
    _fetch_dbnomics,
    _fetch_fred,
    _fetch_yfinance,
    _read_cache,
    _write_cache,
    fetch_historical_context,
    format_historical_context,
    parse_dbnomics_id,
)


class TestParseDbnomicsId:
    def test_valid_three_parts(self) -> None:
        result = parse_dbnomics_id("meteofrance_TEMPERATURE_celsius.07434.D")
        assert result == ("meteofrance", "TEMPERATURE", "celsius.07434.D")

    def test_valid_with_underscores_in_series(self) -> None:
        result = parse_dbnomics_id("provider_dataset_series_with_underscores")
        assert result == ("provider", "dataset", "series_with_underscores")

    def test_too_few_parts(self) -> None:
        assert parse_dbnomics_id("only_two") is None

    def test_single_part(self) -> None:
        assert parse_dbnomics_id("single") is None


class TestFormatHistoricalContext:
    def test_basic_format(self) -> None:
        values = {
            "2024-01-01": 100.0,
            "2024-01-15": 105.0,
            "2024-02-01": 110.0,
            "2024-02-15": 108.0,
            "2024-03-01": 115.0,
        }
        result = format_historical_context(values, "2024-03-01")
        assert "Current value: 115.0" in result
        assert "Range: 100.0" in result
        assert "115.0" in result
        assert f"last {LOOKBACK_DAYS} days" in result

    def test_includes_change_percentage(self) -> None:
        values = {
            "2024-01-01": 100.0,
            "2024-01-15": 105.0,
            "2024-02-01": 110.0,
            "2024-02-15": 108.0,
            "2024-03-01": 115.0,
        }
        result = format_historical_context(values, "2024-03-01")
        assert "30-day change:" in result

    def test_last_10_values(self) -> None:
        values = {f"2024-01-{d:02d}": float(d) for d in range(1, 16)}
        result = format_historical_context(values, "2024-01-15")
        assert "Last 10 values:" in result
        assert "2024-01-15: 15.0" in result

    def test_empty_values(self) -> None:
        assert format_historical_context({}, "2024-01-01") == ""

    def test_single_value(self) -> None:
        values = {"2024-01-01": 42.0}
        result = format_historical_context(values, "2024-01-01")
        assert "Current value: 42.0" in result
        assert "Range: 42.0 – 42.0" in result


class TestCache:
    def test_round_trip(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        import timeseries_rag

        monkeypatch.setattr(timeseries_rag, "RAG_CACHE_DIR", tmp_path / "rag")

        values = {"2024-01-01": 100.0, "2024-01-02": 101.0}
        _write_cache("fred", "GDP", "2024-01-02", values)
        result = _read_cache("fred", "GDP", "2024-01-02")
        assert result == values

    def test_cache_miss(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        import timeseries_rag

        monkeypatch.setattr(timeseries_rag, "RAG_CACHE_DIR", tmp_path / "rag")

        result = _read_cache("fred", "NONEXISTENT", "2024-01-01")
        assert result is None

    def test_corrupt_cache(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        import timeseries_rag

        monkeypatch.setattr(timeseries_rag, "RAG_CACHE_DIR", tmp_path / "rag")

        path = tmp_path / "rag" / "fred" / "BAD_2024-01-01.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("not json")
        result = _read_cache("fred", "BAD", "2024-01-01")
        assert result is None

    def test_slashes_in_id(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        import timeseries_rag

        monkeypatch.setattr(timeseries_rag, "RAG_CACHE_DIR", tmp_path / "rag")

        values = {"2024-01-01": 50.0}
        _write_cache("dbnomics", "a/b/c", "2024-01-01", values)
        result = _read_cache("dbnomics", "a/b/c", "2024-01-01")
        assert result == values


class TestFetchFred:
    def test_missing_api_key(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import timeseries_rag

        monkeypatch.setattr(timeseries_rag, "FRED_API_KEY", "")
        result = _fetch_fred("GDP", date(2024, 1, 1))
        assert result is None

    def test_successful_fetch(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import sys
        import timeseries_rag
        import pandas as pd

        monkeypatch.setattr(timeseries_rag, "FRED_API_KEY", "test-key")

        mock_series = pd.Series(
            [100.0, 101.0, 102.0],
            index=pd.to_datetime(["2024-01-01", "2024-01-15", "2024-02-01"]),
        )
        mock_fred_instance = MagicMock()
        mock_fred_instance.get_series.return_value = mock_series

        mock_fredapi = MagicMock()
        mock_fredapi.Fred.return_value = mock_fred_instance
        monkeypatch.setitem(sys.modules, "fredapi", mock_fredapi)

        result = _fetch_fred("GDP", date(2024, 3, 1))

        assert result is not None
        assert len(result) == 3
        assert result["2024-02-01"] == 102.0

    def test_api_error(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import sys
        import timeseries_rag

        monkeypatch.setattr(timeseries_rag, "FRED_API_KEY", "test-key")

        mock_fredapi = MagicMock()
        mock_fredapi.Fred.return_value.get_series.side_effect = Exception("API error")
        monkeypatch.setitem(sys.modules, "fredapi", mock_fredapi)

        result = _fetch_fred("BAD_SERIES", date(2024, 1, 1))
        assert result is None


class TestFetchYfinance:
    def test_successful_fetch(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import sys
        import pandas as pd

        mock_hist = pd.DataFrame(
            {"Close": [150.0, 151.0, 152.0]},
            index=pd.to_datetime(["2024-01-01", "2024-01-15", "2024-02-01"]),
        )
        mock_ticker = MagicMock()
        mock_ticker.history.return_value = mock_hist

        mock_yf = MagicMock()
        mock_yf.Ticker.return_value = mock_ticker
        monkeypatch.setitem(sys.modules, "yfinance", mock_yf)

        result = _fetch_yfinance("AAPL", date(2024, 3, 1))

        assert result is not None
        assert len(result) == 3

    def test_empty_history(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import sys
        import pandas as pd

        mock_ticker = MagicMock()
        mock_ticker.history.return_value = pd.DataFrame()

        mock_yf = MagicMock()
        mock_yf.Ticker.return_value = mock_ticker
        monkeypatch.setitem(sys.modules, "yfinance", mock_yf)

        result = _fetch_yfinance("NONEXISTENT", date(2024, 1, 1))
        assert result is None

    def test_api_error(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import sys

        mock_yf = MagicMock()
        mock_yf.Ticker.side_effect = Exception("Network error")
        monkeypatch.setitem(sys.modules, "yfinance", mock_yf)

        result = _fetch_yfinance("AAPL", date(2024, 1, 1))
        assert result is None


class TestFetchDbnomics:
    def test_successful_fetch(self) -> None:
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "series": {
                "docs": [{
                    "period": ["2024-01-01", "2024-01-15", "2024-02-01", "2024-06-01"],
                    "value": [10.0, 11.0, 12.0, 20.0],
                }]
            }
        }
        mock_response.raise_for_status = MagicMock()

        with patch("timeseries_rag.requests.get", return_value=mock_response):
            result = _fetch_dbnomics("prov_ds_series", date(2024, 3, 1))

        assert result is not None
        assert len(result) == 3
        assert "2024-06-01" not in result

    def test_invalid_id(self) -> None:
        result = _fetch_dbnomics("invalid", date(2024, 1, 1))
        assert result is None

    def test_na_values_filtered(self) -> None:
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "series": {
                "docs": [{
                    "period": ["2024-01-01", "2024-01-15"],
                    "value": [10.0, "NA"],
                }]
            }
        }
        mock_response.raise_for_status = MagicMock()

        with patch("timeseries_rag.requests.get", return_value=mock_response):
            result = _fetch_dbnomics("prov_ds_series", date(2024, 3, 1))

        assert result is not None
        assert len(result) == 1

    def test_network_error(self) -> None:
        with patch("timeseries_rag.requests.get", side_effect=Exception("timeout")):
            result = _fetch_dbnomics("prov_ds_series", date(2024, 1, 1))

        assert result is None


class TestFetchHistoricalContext:
    def _make_question(
        self,
        source: str = "fred",
        question_id: str = "GDP",
        freeze_value: float | None = None,
        forecast_due_date: str = "2024-03-01",
    ) -> MagicMock:
        q = MagicMock()
        q.source = source
        q.id = question_id
        q.freeze_datetime_value = freeze_value
        q.forecast_due_date = forecast_due_date
        q.freeze_datetime = None
        return q

    def test_rag_disabled(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import timeseries_rag

        monkeypatch.setattr(timeseries_rag, "FORECAST_RAG", False)
        result = fetch_historical_context(self._make_question())
        assert result == ""

    def test_non_timeseries_source(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import timeseries_rag

        monkeypatch.setattr(timeseries_rag, "FORECAST_RAG", True)
        result = fetch_historical_context(self._make_question(source="metaculus"))
        assert result == ""

    def test_fetcher_returns_none(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import timeseries_rag

        monkeypatch.setattr(timeseries_rag, "FORECAST_RAG", True)
        with patch.dict(timeseries_rag._FETCHERS, {"fred": lambda *a: None}):
            result = fetch_historical_context(self._make_question())
        assert result == ""

    def test_fetcher_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import timeseries_rag

        monkeypatch.setattr(timeseries_rag, "FORECAST_RAG", True)

        def exploding_fetcher(*args: object) -> None:
            raise RuntimeError("boom")

        with patch.dict(timeseries_rag._FETCHERS, {"fred": exploding_fetcher}):
            result = fetch_historical_context(self._make_question())
        assert result == ""

    def test_successful_fetch_and_format(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
    ) -> None:
        import timeseries_rag

        monkeypatch.setattr(timeseries_rag, "FORECAST_RAG", True)
        monkeypatch.setattr(timeseries_rag, "RAG_CACHE_DIR", tmp_path / "rag")

        mock_values = {"2024-01-01": 100.0, "2024-02-01": 110.0, "2024-03-01": 115.0}

        with patch.dict(timeseries_rag._FETCHERS, {"fred": lambda *a: mock_values}):
            result = fetch_historical_context(self._make_question())

        assert "Current value: 115.0" in result
        assert "Range: 100.0" in result
        cache_file = tmp_path / "rag" / "fred" / "GDP_2024-03-01.json"
        assert cache_file.exists()

    def test_uses_cache(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        import timeseries_rag

        monkeypatch.setattr(timeseries_rag, "FORECAST_RAG", True)
        monkeypatch.setattr(timeseries_rag, "RAG_CACHE_DIR", tmp_path / "rag")

        cached_values = {"2024-01-01": 50.0, "2024-02-01": 55.0}
        _write_cache("fred", "GDP", "2024-03-01", cached_values)

        call_count = 0

        def counting_fetcher(*args: object) -> dict[str, float]:
            nonlocal call_count
            call_count += 1
            return {"2024-01-01": 999.0}

        with patch.dict(timeseries_rag._FETCHERS, {"fred": counting_fetcher}):
            result = fetch_historical_context(self._make_question())

        assert call_count == 0
        assert "Current value: 55.0" in result

    def test_no_question_id(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import timeseries_rag

        monkeypatch.setattr(timeseries_rag, "FORECAST_RAG", True)
        q = self._make_question()
        q.id = ""
        assert fetch_historical_context(q) == ""


@pytest.mark.integration
class TestIntegration:
    def test_fred_real_fetch(self) -> None:
        if not os.getenv("FRED_API_KEY"):
            pytest.skip("FRED_API_KEY not set")
        result = _fetch_fred("GDP", date(2024, 6, 1))
        assert result is not None
        assert len(result) > 0

    def test_yfinance_real_fetch(self) -> None:
        result = _fetch_yfinance("AAPL", date(2024, 6, 1))
        assert result is not None
        assert len(result) > 0

    def test_dbnomics_real_fetch(self) -> None:
        result = _fetch_dbnomics("IMF_WEO:2024-04_USA.NGDPD", date(2024, 6, 1))
        if result is not None:
            assert len(result) > 0
