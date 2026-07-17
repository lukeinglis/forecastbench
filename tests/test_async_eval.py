"""Tests for async eval path in eval.py."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from unittest.mock import patch

import pytest

from fetch_data import Question
from eval import (
    is_async_forecaster,
    _read_cache,
    _write_cache,
    _run_async,
    _run_sync,
)


def _make_question(qid: str = "q1") -> Question:
    return Question(id=qid, source="acled", question=f"Test question {qid}")


class TestForecasterDetection:
    def test_sync_function_detected(self) -> None:
        def sync_fn(q: Question) -> float:
            return 0.5

        assert not is_async_forecaster(sync_fn)

    def test_async_function_detected(self) -> None:
        async def async_fn(q: Question) -> float:
            return 0.5

        assert is_async_forecaster(async_fn)

    def test_lambda_detected_as_sync(self) -> None:
        fn = lambda q: 0.5  # noqa: E731
        assert not is_async_forecaster(fn)


class TestCaching:
    def test_cache_miss_returns_none(self, tmp_path: Path) -> None:
        with patch("eval.CACHE_DIR", tmp_path):
            assert _read_cache("test_model", "nonexistent_q") is None

    def test_cache_write_then_read(self, tmp_path: Path) -> None:
        with patch("eval.CACHE_DIR", tmp_path):
            _write_cache("test_model", "q1", 0.73)
            result = _read_cache("test_model", "q1")
            assert result == pytest.approx(0.73)

    def test_cache_creates_subdirectory(self, tmp_path: Path) -> None:
        with patch("eval.CACHE_DIR", tmp_path):
            _write_cache("my_model", "q1", 0.5)
            assert (tmp_path / "my_model" / "q1.json").exists()

    def test_cache_stores_correct_json(self, tmp_path: Path) -> None:
        with patch("eval.CACHE_DIR", tmp_path):
            _write_cache("mdl", "q42", 0.85)
            data = json.loads((tmp_path / "mdl" / "q42.json").read_text())
            assert data["probability"] == 0.85
            assert data["model"] == "mdl"
            assert data["question_id"] == "q42"

    def test_corrupt_cache_returns_none(self, tmp_path: Path) -> None:
        with patch("eval.CACHE_DIR", tmp_path):
            cache_dir = tmp_path / "mdl"
            cache_dir.mkdir()
            (cache_dir / "q1.json").write_text("not json")
            assert _read_cache("mdl", "q1") is None


class TestSyncPath:
    def test_sync_forecaster_backward_compatible(self, tmp_path: Path) -> None:
        def dummy(q: Question, **kwargs: object) -> float:
            return 0.5

        questions = [_make_question(f"q{i}") for i in range(3)]
        with patch("eval.CACHE_DIR", tmp_path):
            forecasts = _run_sync(dummy, questions, "test")
        assert len(forecasts) == 3
        assert all(v == 0.5 for v in forecasts.values())

    def test_sync_uses_cache(self, tmp_path: Path) -> None:
        call_count = 0

        def counting_fn(q: Question, **kwargs: object) -> float:
            nonlocal call_count
            call_count += 1
            return 0.7

        with patch("eval.CACHE_DIR", tmp_path):
            _write_cache("test", "q0", 0.99)
            questions = [_make_question("q0"), _make_question("q1")]
            forecasts = _run_sync(counting_fn, questions, "test")

        assert forecasts["q0"] == pytest.approx(0.99)
        assert forecasts["q1"] == pytest.approx(0.7)
        assert call_count == 1


class TestAsyncPath:
    async def test_async_forecaster_runs(self, tmp_path: Path) -> None:
        async def async_fn(q: Question, **kwargs: object) -> float:
            return 0.6

        questions = [_make_question(f"q{i}") for i in range(3)]
        with patch("eval.CACHE_DIR", tmp_path):
            forecasts = await _run_async(async_fn, questions, "test")
        assert len(forecasts) == 3
        assert all(v == pytest.approx(0.6) for v in forecasts.values())

    async def test_semaphore_limits_concurrency(self, tmp_path: Path) -> None:
        max_concurrent = 0
        current = 0
        lock = asyncio.Lock()

        async def tracking_fn(q: Question, **kwargs: object) -> float:
            nonlocal max_concurrent, current
            async with lock:
                current += 1
                if current > max_concurrent:
                    max_concurrent = current
            await asyncio.sleep(0.01)
            async with lock:
                current -= 1
            return 0.5

        questions = [_make_question(f"q{i}") for i in range(20)]
        with (
            patch("eval.CACHE_DIR", tmp_path),
            patch.dict("os.environ", {"FORECAST_CONCURRENCY": "3"}),
        ):
            await _run_async(tracking_fn, questions, "test")

        assert max_concurrent <= 3

    async def test_cache_hit_skips_api_call(self, tmp_path: Path) -> None:
        call_count = 0

        async def counting_fn(q: Question, **kwargs: object) -> float:
            nonlocal call_count
            call_count += 1
            return 0.7

        with patch("eval.CACHE_DIR", tmp_path):
            _write_cache("test", "q0", 0.99)
            _write_cache("test", "q1", 0.88)
            questions = [_make_question("q0"), _make_question("q1"), _make_question("q2")]
            forecasts = await _run_async(counting_fn, questions, "test")

        assert forecasts["q0"] == pytest.approx(0.99)
        assert forecasts["q1"] == pytest.approx(0.88)
        assert forecasts["q2"] == pytest.approx(0.7)
        assert call_count == 1

    async def test_cache_miss_writes_cache(self, tmp_path: Path) -> None:
        async def fn(q: Question, **kwargs: object) -> float:
            return 0.65

        with patch("eval.CACHE_DIR", tmp_path):
            await _run_async(fn, [_make_question("q1")], "mdl")
            cached = _read_cache("mdl", "q1")

        assert cached == pytest.approx(0.65)

    async def test_failed_forecast_returns_fallback(self, tmp_path: Path) -> None:
        async def failing_fn(q: Question, **kwargs: object) -> float:
            raise RuntimeError("API error")

        with patch("eval.CACHE_DIR", tmp_path):
            forecasts = await _run_async(failing_fn, [_make_question("q1")], "test")
            assert _read_cache("test", "q1") is None

        assert forecasts["q1"] == pytest.approx(0.5)

    async def test_empty_question_list(self, tmp_path: Path) -> None:
        async def fn(q: Question, **kwargs: object) -> float:
            return 0.5

        with patch("eval.CACHE_DIR", tmp_path):
            forecasts = await _run_async(fn, [], "test")

        assert forecasts == {}
