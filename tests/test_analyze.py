"""Tests for error analysis module."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from fetch_data import ResolvedQuestion
from analyze import analyze_by_source, analyze_calibration, analyze_biases, save_analysis


def _rq(qid: str, source: str, outcome: int) -> ResolvedQuestion:
    return ResolvedQuestion(
        id=qid,
        source=source,
        question=f"Q {qid}",
        outcome=outcome,
        forecast_due_date="2024-01-01",
    )


class TestAnalyzeBySource:
    def test_groups_by_source(self) -> None:
        resolved = [
            _rq("q1", "metaculus", 1),
            _rq("q2", "metaculus", 0),
            _rq("q3", "polymarket", 1),
        ]
        forecasts = {"q1": 0.9, "q2": 0.1, "q3": 0.5}
        result = analyze_by_source(forecasts, resolved)

        assert "metaculus" in result
        assert "polymarket" in result
        assert result["metaculus"]["count"] == 2
        assert result["polymarket"]["count"] == 1

    def test_brier_scores_correct(self) -> None:
        resolved = [_rq("q1", "acled", 1)]
        forecasts = {"q1": 0.9}
        result = analyze_by_source(forecasts, resolved)
        assert result["acled"]["brier"] == pytest.approx(0.01)

    def test_missing_forecast_defaults_to_half(self) -> None:
        resolved = [_rq("q1", "acled", 1)]
        result = analyze_by_source({}, resolved)
        assert result["acled"]["brier"] == pytest.approx(0.25)

    def test_index_computed(self) -> None:
        resolved = [_rq("q1", "acled", 1)]
        forecasts = {"q1": 0.9}
        result = analyze_by_source(forecasts, resolved)
        assert result["acled"]["index"] > 0


class TestAnalyzeCalibration:
    def test_bins_created(self) -> None:
        resolved = [_rq(f"q{i}", "acled", i % 2) for i in range(20)]
        forecasts = {f"q{i}": i / 20.0 for i in range(20)}
        bins = analyze_calibration(forecasts, resolved, n_bins=5)
        assert len(bins) > 0

    def test_bin_structure(self) -> None:
        resolved = [_rq("q1", "acled", 1)]
        forecasts = {"q1": 0.55}
        bins = analyze_calibration(forecasts, resolved, n_bins=10)
        assert len(bins) == 1
        b = bins[0]
        assert "bin_low" in b
        assert "bin_high" in b
        assert "mean_predicted" in b
        assert "mean_observed" in b
        assert "count" in b
        assert b["mean_predicted"] == pytest.approx(0.55)
        assert b["mean_observed"] == pytest.approx(1.0)

    def test_empty_data(self) -> None:
        assert analyze_calibration({}, [], n_bins=10) == []

    def test_perfect_calibration(self) -> None:
        resolved = [_rq(f"q{i}", "acled", 1) for i in range(10)]
        forecasts = {f"q{i}": 0.95 for i in range(10)}
        bins = analyze_calibration(forecasts, resolved, n_bins=10)
        for b in bins:
            assert b["mean_observed"] == pytest.approx(1.0)


class TestAnalyzeBiases:
    def test_optimistic_bias(self) -> None:
        resolved = [_rq(f"q{i}", "acled", 0) for i in range(10)]
        forecasts = {f"q{i}": 0.8 for i in range(10)}
        result = analyze_biases(forecasts, resolved)
        assert result["bias"] > 0
        assert result["mean_forecast"] == pytest.approx(0.8)
        assert result["mean_outcome"] == pytest.approx(0.0)

    def test_pessimistic_bias(self) -> None:
        resolved = [_rq(f"q{i}", "acled", 1) for i in range(10)]
        forecasts = {f"q{i}": 0.2 for i in range(10)}
        result = analyze_biases(forecasts, resolved)
        assert result["bias"] < 0

    def test_low_bin_stats(self) -> None:
        resolved = [_rq(f"q{i}", "acled", 1) for i in range(5)]
        forecasts = {f"q{i}": 0.1 for i in range(5)}
        result = analyze_biases(forecasts, resolved)
        assert result["low_bin"]["count"] == 5
        assert result["low_bin"]["mean_predicted"] == pytest.approx(0.1)
        assert result["low_bin"]["mean_observed"] == pytest.approx(1.0)

    def test_high_bin_stats(self) -> None:
        resolved = [_rq(f"q{i}", "acled", 0) for i in range(5)]
        forecasts = {f"q{i}": 0.9 for i in range(5)}
        result = analyze_biases(forecasts, resolved)
        assert result["high_bin"]["count"] == 5
        assert result["high_bin"]["mean_predicted"] == pytest.approx(0.9)

    def test_empty_data(self) -> None:
        result = analyze_biases({}, [])
        assert result["bias"] == 0.0


class TestSaveAnalysis:
    def test_saves_json(self, tmp_path: Path) -> None:
        analysis = {"by_source": {"acled": {"brier": 0.25, "count": 10}}}
        out = tmp_path / "sub" / "analysis.json"
        save_analysis(analysis, out)
        loaded = json.loads(out.read_text())
        assert loaded == analysis
