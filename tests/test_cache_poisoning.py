"""Tests for cache poisoning prevention.

Verifies that fallback 0.5 values are NOT cached, so re-running retries the API call.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, patch

from fetch_data import Question
from eval import _run_sync, _read_cache


def _raising_forecaster(question: Question, **kwargs: object) -> float:
    raise RuntimeError("API timeout")


def _good_forecaster(question: Question, **kwargs: object) -> float:
    return 0.73


class TestSyncFallbackNotCached:
    def test_multi_horizon_exception_not_cached(self, tmp_path: Path) -> None:
        """When forecaster raises on a multi-horizon question, 0.5 should NOT be cached."""
        import eval as eval_mod
        original = eval_mod.CACHE_DIR
        eval_mod.CACHE_DIR = tmp_path / "cache"
        try:
            q = Question(
                id="mh1", source="acled", question="MH?",
                resolution_dates=["2024-01-01", "2024-06-01"],
            )
            forecasts = _run_sync(_raising_forecaster, [q], "test_slug")
            assert forecasts["mh1_2024-01-01"] == 0.5
            assert forecasts["mh1_2024-06-01"] == 0.5
            assert _read_cache("test_slug", "mh1_2024-01-01") is None
            assert _read_cache("test_slug", "mh1_2024-06-01") is None
        finally:
            eval_mod.CACHE_DIR = original

    def test_multi_horizon_success_is_cached(self, tmp_path: Path) -> None:
        """When forecaster succeeds, the result SHOULD be cached."""
        import eval as eval_mod
        original = eval_mod.CACHE_DIR
        eval_mod.CACHE_DIR = tmp_path / "cache"
        try:
            q = Question(
                id="mh2", source="acled", question="MH?",
                resolution_dates=["2024-01-01"],
            )
            forecasts = _run_sync(_good_forecaster, [q], "test_slug")
            assert forecasts["mh2_2024-01-01"] == 0.73
            assert _read_cache("test_slug", "mh2_2024-01-01") == 0.73
        finally:
            eval_mod.CACHE_DIR = original

    def test_rerun_after_failure_retries(self, tmp_path: Path) -> None:
        """After a failed run (no cache), re-running should call the forecaster again."""
        import eval as eval_mod
        original = eval_mod.CACHE_DIR
        eval_mod.CACHE_DIR = tmp_path / "cache"
        try:
            q = Question(
                id="mh3", source="acled", question="MH?",
                resolution_dates=["2024-01-01"],
            )
            # First run: forecaster raises, result is 0.5, NOT cached
            forecasts1 = _run_sync(_raising_forecaster, [q], "test_slug")
            assert forecasts1["mh3_2024-01-01"] == 0.5

            # Second run: forecaster succeeds, result is 0.73, IS cached
            forecasts2 = _run_sync(_good_forecaster, [q], "test_slug")
            assert forecasts2["mh3_2024-01-01"] == 0.73
            assert _read_cache("test_slug", "mh3_2024-01-01") == 0.73
        finally:
            eval_mod.CACHE_DIR = original


class TestAsyncMultiHorizonFallbackNotCached:
    def test_multi_horizon_none_return_not_cached(self, tmp_path: Path) -> None:
        """When aforecast_multi_horizon returns None, results should NOT be cached."""
        import eval as eval_mod
        original = eval_mod.CACHE_DIR
        eval_mod.CACHE_DIR = tmp_path / "cache"
        try:
            q = Question(
                id="amh1", source="acled", question="Async MH?",
                resolution_dates=["2024-01-01", "2024-06-01"],
            )

            mock_multi = AsyncMock(return_value=None)
            with patch("eval.is_async_forecaster", return_value=True):
                from eval import _run_async
                with patch("baseline_agent.aforecast_multi_horizon", mock_multi):
                    async_forecaster = AsyncMock(return_value=0.5)
                    forecasts = asyncio.run(
                        _run_async(async_forecaster, [q], "test_slug", multi_horizon=True)
                    )

            assert forecasts["amh1_2024-01-01"] == 0.5
            assert forecasts["amh1_2024-06-01"] == 0.5
            assert _read_cache("test_slug", "amh1_2024-01-01") is None
            assert _read_cache("test_slug", "amh1_2024-06-01") is None
        finally:
            eval_mod.CACHE_DIR = original

    def test_multi_horizon_success_cached(self, tmp_path: Path) -> None:
        """When aforecast_multi_horizon returns real values, results SHOULD be cached."""
        import eval as eval_mod
        original = eval_mod.CACHE_DIR
        eval_mod.CACHE_DIR = tmp_path / "cache"
        try:
            q = Question(
                id="amh2", source="acled", question="Async MH?",
                resolution_dates=["2024-01-01", "2024-06-01"],
            )

            mock_multi = AsyncMock(return_value=[0.3, 0.7])
            with patch("eval.is_async_forecaster", return_value=True):
                from eval import _run_async
                with patch("baseline_agent.aforecast_multi_horizon", mock_multi):
                    async_forecaster = AsyncMock(return_value=0.5)
                    forecasts = asyncio.run(
                        _run_async(async_forecaster, [q], "test_slug", multi_horizon=True)
                    )

            assert forecasts["amh2_2024-01-01"] == 0.3
            assert forecasts["amh2_2024-06-01"] == 0.7
            assert _read_cache("test_slug", "amh2_2024-01-01") == 0.3
            assert _read_cache("test_slug", "amh2_2024-06-01") == 0.7
        finally:
            eval_mod.CACHE_DIR = original

    def test_multi_horizon_exception_not_cached(self, tmp_path: Path) -> None:
        """When aforecast_multi_horizon raises, results should NOT be cached."""
        import eval as eval_mod
        original = eval_mod.CACHE_DIR
        eval_mod.CACHE_DIR = tmp_path / "cache"
        try:
            q = Question(
                id="amh3", source="acled", question="Async MH?",
                resolution_dates=["2024-01-01", "2024-06-01"],
            )

            mock_multi = AsyncMock(side_effect=RuntimeError("API error"))
            with patch("eval.is_async_forecaster", return_value=True):
                from eval import _run_async
                with patch("baseline_agent.aforecast_multi_horizon", mock_multi):
                    async_forecaster = AsyncMock(return_value=0.5)
                    forecasts = asyncio.run(
                        _run_async(async_forecaster, [q], "test_slug", multi_horizon=True)
                    )

            assert forecasts["amh3_2024-01-01"] == 0.5
            assert forecasts["amh3_2024-06-01"] == 0.5
            assert _read_cache("test_slug", "amh3_2024-01-01") is None
            assert _read_cache("test_slug", "amh3_2024-06-01") is None
        finally:
            eval_mod.CACHE_DIR = original

    def test_rerun_after_failure_retries_async(self, tmp_path: Path) -> None:
        """After async multi-horizon failure (None), re-running should retry."""
        import eval as eval_mod
        original = eval_mod.CACHE_DIR
        eval_mod.CACHE_DIR = tmp_path / "cache"
        try:
            q = Question(
                id="amh4", source="acled", question="Async MH?",
                resolution_dates=["2024-01-01"],
            )
            async_forecaster = AsyncMock(return_value=0.5)

            # First run: returns None (fallback), NOT cached
            mock_fail = AsyncMock(return_value=None)
            from eval import _run_async
            with patch("baseline_agent.aforecast_multi_horizon", mock_fail):
                forecasts1 = asyncio.run(
                    _run_async(async_forecaster, [q], "test_slug", multi_horizon=True)
                )
            assert forecasts1["amh4_2024-01-01"] == 0.5
            assert _read_cache("test_slug", "amh4_2024-01-01") is None

            # Second run: returns real value, IS cached
            mock_ok = AsyncMock(return_value=[0.82])
            with patch("baseline_agent.aforecast_multi_horizon", mock_ok):
                forecasts2 = asyncio.run(
                    _run_async(async_forecaster, [q], "test_slug", multi_horizon=True)
                )
            assert forecasts2["amh4_2024-01-01"] == 0.82
            assert _read_cache("test_slug", "amh4_2024-01-01") == 0.82
        finally:
            eval_mod.CACHE_DIR = original
