"""Tests for cache poisoning prevention.

Verifies that fallback 0.5 values are NOT cached, so re-running retries the API call.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from unittest.mock import AsyncMock

from fetch_data import Question
from eval import _run_sync, _run_async, _read_cache


def _raising_forecaster(question: Question, **kwargs: object) -> float:
    raise RuntimeError("API timeout")


def _good_forecaster(question: Question, **kwargs: object) -> float:
    return 0.73


class TestSyncFallbackNotCached:
    def test_exception_skips_question(self, tmp_path: Path) -> None:
        """When forecaster raises, question is skipped (not cached)."""
        import eval as eval_mod
        original = eval_mod.CACHE_DIR
        eval_mod.CACHE_DIR = tmp_path / "cache"
        try:
            q = Question(
                id="mh1", source="acled", question="MH?",
                resolution_dates=["2024-01-01", "2024-06-01"],
            )
            forecasts = _run_sync(_raising_forecaster, [q], "test_slug")
            assert "mh1" not in forecasts
            assert _read_cache("test_slug", "mh1") is None
        finally:
            eval_mod.CACHE_DIR = original

    def test_success_is_cached(self, tmp_path: Path) -> None:
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
            assert forecasts["mh2"] == 0.73
            assert _read_cache("test_slug", "mh2") == 0.73
        finally:
            eval_mod.CACHE_DIR = original

    def test_rerun_after_failure_retries(self, tmp_path: Path) -> None:
        """After a failed run (skipped), re-running should call the forecaster again."""
        import eval as eval_mod
        original = eval_mod.CACHE_DIR
        eval_mod.CACHE_DIR = tmp_path / "cache"
        try:
            q = Question(
                id="mh3", source="acled", question="MH?",
                resolution_dates=["2024-01-01"],
            )
            forecasts1 = _run_sync(_raising_forecaster, [q], "test_slug")
            assert "mh3" not in forecasts1

            forecasts2 = _run_sync(_good_forecaster, [q], "test_slug")
            assert forecasts2["mh3"] == 0.73
            assert _read_cache("test_slug", "mh3") == 0.73
        finally:
            eval_mod.CACHE_DIR = original


class TestAsyncFallbackNotCached:
    def test_async_success_cached(self, tmp_path: Path) -> None:
        """When async forecaster succeeds, the result SHOULD be cached by base ID."""
        import eval as eval_mod
        original = eval_mod.CACHE_DIR
        eval_mod.CACHE_DIR = tmp_path / "cache"
        try:
            q = Question(
                id="amh2", source="acled", question="Async MH?",
                resolution_dates=["2024-01-01", "2024-06-01"],
            )

            async_forecaster = AsyncMock(return_value=0.7)
            forecasts = asyncio.run(
                _run_async(async_forecaster, [q], "test_slug")
            )

            assert forecasts["amh2"] == 0.7
            assert _read_cache("test_slug", "amh2") == 0.7
        finally:
            eval_mod.CACHE_DIR = original

    def test_async_failure_not_cached(self, tmp_path: Path) -> None:
        """When async forecaster raises, question is skipped and NOT cached."""
        import eval as eval_mod
        original = eval_mod.CACHE_DIR
        eval_mod.CACHE_DIR = tmp_path / "cache"
        try:
            q = Question(
                id="amh3", source="acled", question="Async MH?",
                resolution_dates=["2024-01-01", "2024-06-01"],
            )

            async_forecaster_fail = AsyncMock(side_effect=RuntimeError("API down"))
            forecasts = asyncio.run(
                _run_async(async_forecaster_fail, [q], "test_slug")
            )

            assert "amh3" not in forecasts
            assert _read_cache("test_slug", "amh3") is None
        finally:
            eval_mod.CACHE_DIR = original

    def test_rerun_after_failure_retries_async(self, tmp_path: Path) -> None:
        """After async failure, re-running should retry and cache on success."""
        import eval as eval_mod
        original = eval_mod.CACHE_DIR
        eval_mod.CACHE_DIR = tmp_path / "cache"
        try:
            q = Question(
                id="amh4", source="acled", question="Async MH?",
                resolution_dates=["2024-01-01"],
            )

            async_forecaster_fail = AsyncMock(side_effect=RuntimeError("API down"))
            forecasts1 = asyncio.run(
                _run_async(async_forecaster_fail, [q], "test_slug")
            )
            assert "amh4" not in forecasts1
            assert _read_cache("test_slug", "amh4") is None

            async_forecaster_ok = AsyncMock(return_value=0.82)
            forecasts2 = asyncio.run(
                _run_async(async_forecaster_ok, [q], "test_slug")
            )
            assert forecasts2["amh4"] == 0.82
            assert _read_cache("test_slug", "amh4") == 0.82
        finally:
            eval_mod.CACHE_DIR = original
