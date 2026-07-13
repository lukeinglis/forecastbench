"""Tests for error analysis module."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from fetch_data import ResolvedQuestion
from analyze import (
    analyze_by_source,
    analyze_calibration,
    analyze_biases,
    analyze_worst_questions,
    analyze_by_horizon,
    brier_decomposition,
    compare_paired,
    save_analysis,
)


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


class TestAnalyzeWorstQuestions:
    def test_finds_worst(self) -> None:
        resolved = [_rq("q1", "acled", 1), _rq("q2", "acled", 0), _rq("q3", "metaculus", 1)]
        forecasts = {"q1": 0.1, "q2": 0.9, "q3": 0.9}
        worst = analyze_worst_questions(forecasts, resolved, top_n=2)
        assert len(worst) == 2
        assert worst[0]["brier"] >= worst[1]["brier"]

    def test_categorizes_confident_wrong(self) -> None:
        resolved = [_rq("q1", "acled", 0)]
        forecasts = {"q1": 0.95}
        worst = analyze_worst_questions(forecasts, resolved, top_n=1)
        assert worst[0]["category"] == "confident_wrong_positive"

    def test_categorizes_uncertain(self) -> None:
        resolved = [_rq("q1", "acled", 1)]
        forecasts = {"q1": 0.5}
        worst = analyze_worst_questions(forecasts, resolved, top_n=1)
        assert worst[0]["category"] == "uncertain"

    def test_missing_defaults_to_half(self) -> None:
        resolved = [_rq("q1", "acled", 1)]
        worst = analyze_worst_questions({}, resolved, top_n=1)
        assert worst[0]["forecast"] == 0.5


class TestAnalyzeByHorizon:
    def test_groups_by_date(self) -> None:
        resolved = [
            ResolvedQuestion(id="q1_2024-02-01", source="acled", question="Q1",
                             outcome=1, forecast_due_date="2024-01-01", resolution_date="2024-02-01"),
            ResolvedQuestion(id="q2_2024-02-01", source="fred", question="Q2",
                             outcome=0, forecast_due_date="2024-01-01", resolution_date="2024-02-01"),
            ResolvedQuestion(id="q3_2024-06-01", source="acled", question="Q3",
                             outcome=1, forecast_due_date="2024-01-01", resolution_date="2024-06-01"),
        ]
        forecasts = {"q1_2024-02-01": 0.8, "q2_2024-02-01": 0.2, "q3_2024-06-01": 0.7}
        result = analyze_by_horizon(forecasts, resolved)
        assert "2024-02-01" in result
        assert "2024-06-01" in result
        assert result["2024-02-01"]["count"] == 2
        assert result["2024-06-01"]["count"] == 1

    def test_non_horizon_questions_excluded(self) -> None:
        resolved = [_rq("q1", "metaculus", 1)]
        result = analyze_by_horizon({"q1": 0.5}, resolved)
        assert result == {}


class TestBrierDecomposition:
    def test_components_sum(self) -> None:
        resolved = [_rq(f"q{i}", "acled", i % 2) for i in range(100)]
        forecasts = {f"q{i}": (i % 2) * 0.8 + (1 - i % 2) * 0.2 for i in range(100)}
        decomp = brier_decomposition(forecasts, resolved)
        expected = decomp["reliability"] - decomp["resolution"] + decomp["uncertainty"]
        assert decomp["brier_from_decomposition"] == pytest.approx(expected)

    def test_perfect_calibration_zero_reliability(self) -> None:
        resolved = [_rq(f"q{i}", "acled", 1) for i in range(10)]
        forecasts = {f"q{i}": 1.0 for i in range(10)}
        decomp = brier_decomposition(forecasts, resolved)
        assert decomp["reliability"] == pytest.approx(0.0, abs=1e-10)
        assert decomp["ece"] == pytest.approx(0.0, abs=1e-10)

    def test_empty_data(self) -> None:
        decomp = brier_decomposition({}, [])
        assert decomp["ece"] == 0.0


class TestComparePaired:
    def test_paired_comparison(self, tmp_path: Path) -> None:
        result_a = {
            "model_slug": "model_a",
            "forecasts": {"q1": 0.9, "q2": 0.1, "q3": 0.5},
            "scoring_result": {"overall_brier": 0.20},
        }
        result_b = {
            "model_slug": "model_b",
            "forecasts": {"q1": 0.5, "q2": 0.5, "q3": 0.5},
            "scoring_result": {"overall_brier": 0.25},
        }
        path_a = tmp_path / "a.json"
        path_b = tmp_path / "b.json"
        path_a.write_text(json.dumps(result_a))
        path_b.write_text(json.dumps(result_b))

        result = compare_paired(path_a, path_b)
        assert result["model_a"] == "model_a"
        assert result["model_b"] == "model_b"
        assert result["n_shared"] == 3

    def test_no_shared_questions(self, tmp_path: Path) -> None:
        result_a = {"model_slug": "a", "forecasts": {"q1": 0.5}, "scoring_result": {"overall_brier": 0.25}}
        result_b = {"model_slug": "b", "forecasts": {"q2": 0.5}, "scoring_result": {"overall_brier": 0.25}}
        path_a = tmp_path / "a.json"
        path_b = tmp_path / "b.json"
        path_a.write_text(json.dumps(result_a))
        path_b.write_text(json.dumps(result_b))
        result = compare_paired(path_a, path_b)
        assert result["n_shared"] == 0


class TestSaveAnalysis:
    def test_saves_json(self, tmp_path: Path) -> None:
        analysis = {"by_source": {"acled": {"brier": 0.25, "count": 10}}}
        out = tmp_path / "sub" / "analysis.json"
        save_analysis(analysis, out)
        loaded = json.loads(out.read_text())
        assert loaded == analysis
