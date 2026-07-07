"""Tests for async eval runner: caching, progress, and run_baseline_eval."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from unittest.mock import AsyncMock

import pytest

from fetch_data import Question, ResolvedQuestion
from eval import (
    _format_duration,
    _load_cached_forecast,
    _save_cached_forecast,
    run_baseline_eval,
)
from score import ScoringResult


class TestFormatDuration:
    def test_seconds(self) -> None:
        assert _format_duration(45) == "45s"

    def test_minutes(self) -> None:
        assert _format_duration(252) == "4m12s"

    def test_hours(self) -> None:
        assert _format_duration(4200) == "1h10m"

    def test_zero(self) -> None:
        assert _format_duration(0) == "0s"


class TestForecastCaching:
    def test_cache_miss_returns_none(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr("eval.Path", lambda *a: tmp_path / "nonexistent")
        result = _load_cached_forecast("test-model", "nonexistent-q")
        assert result is None

    def test_cache_roundtrip(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr("eval._cache_dir", lambda model: tmp_path / "forecasts" / model)
        _save_cached_forecast("test-model", "q1", 0.73)

        cache_file = tmp_path / "forecasts" / "test-model" / "q1.json"
        assert cache_file.exists()
        data = json.loads(cache_file.read_text())
        assert data["probability"] == 0.73
        assert data["model"] == "test-model"
        assert "timestamp" in data

    def test_cache_hit_skips_forecaster(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr("eval._cache_dir", lambda model: tmp_path / "forecasts" / model)

        _save_cached_forecast("test-model", "q0", 0.42)

        mock_forecaster = AsyncMock(return_value=0.99)

        resolved = [
            ResolvedQuestion(
                id="q0",
                source="acled",
                question="Cached question",
                outcome=1,
                forecast_due_date="2024-01-01",
            ),
            ResolvedQuestion(
                id="q1",
                source="acled",
                question="Uncached question",
                outcome=0,
                forecast_due_date="2024-01-01",
            ),
        ]

        monkeypatch.setattr(
            "eval.load_data",
            lambda: (
                [__import__("fetch_data").QuestionSet(
                    forecast_due_date="2024-01-01",
                    questions=[
                        Question(id=q.id, source=q.source, question=q.question)
                        for q in resolved
                    ],
                )],
                resolved,
            ),
        )

        asyncio.run(
            run_baseline_eval(mock_forecaster, model="test-model", n_held_out=0)
        )

        assert mock_forecaster.call_count == 1
        called_q = mock_forecaster.call_args[0][0]
        assert called_q.id == "q1"

    def test_cache_file_structure(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr("eval._cache_dir", lambda model: tmp_path / "forecasts" / model)
        _save_cached_forecast("claude-sonnet-4-20250514", "abc-123", 0.65)

        path = tmp_path / "forecasts" / "claude-sonnet-4-20250514" / "abc-123.json"
        assert path.exists()
        data = json.loads(path.read_text())
        assert set(data.keys()) == {"probability", "model", "timestamp"}


class TestRunBaselineEval:
    def test_produces_valid_scoring_result(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr("eval._cache_dir", lambda model: tmp_path / "forecasts" / model)

        resolved = [
            ResolvedQuestion(
                id=f"q{i}",
                source="acled",
                question=f"Question {i}",
                outcome=outcome,
                forecast_due_date="2024-01-01",
            )
            for i, outcome in enumerate([1, 0, 1, 0, 1])
        ]

        mock_forecaster = AsyncMock(return_value=0.5)

        monkeypatch.setattr(
            "eval.load_data",
            lambda: (
                [__import__("fetch_data").QuestionSet(
                    forecast_due_date="2024-01-01",
                    questions=[
                        Question(id=q.id, source=q.source, question=q.question)
                        for q in resolved
                    ],
                )],
                resolved,
            ),
        )

        result = asyncio.run(
            run_baseline_eval(mock_forecaster, model="test-model", n_held_out=0)
        )

        assert isinstance(result, ScoringResult)
        assert result.n_dataset == 5
        assert result.n_missing == 0
        assert 0.0 <= result.dataset_brier <= 1.0

    def test_progress_logging(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        monkeypatch.setattr("eval._cache_dir", lambda model: tmp_path / "forecasts" / model)

        resolved = [
            ResolvedQuestion(
                id=f"q{i}",
                source="acled",
                question=f"Question {i}",
                outcome=i % 2,
                forecast_due_date="2024-01-01",
            )
            for i in range(100)
        ]

        mock_forecaster = AsyncMock(return_value=0.5)

        monkeypatch.setattr(
            "eval.load_data",
            lambda: (
                [__import__("fetch_data").QuestionSet(
                    forecast_due_date="2024-01-01",
                    questions=[
                        Question(id=q.id, source=q.source, question=q.question)
                        for q in resolved
                    ],
                )],
                resolved,
            ),
        )

        asyncio.run(
            run_baseline_eval(mock_forecaster, model="test-model", n_held_out=0)
        )

        captured = capsys.readouterr().out
        assert "%" in captured
        assert "elapsed" in captured
        assert "ETA" in captured
