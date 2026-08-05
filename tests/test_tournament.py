"""Tests for tournament analysis module."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tournament import (
    ModelResult,
    _bootstrap_ci_brier,
    cost_accuracy_summary,
    load_tournament_results,
    model_source_matrix,
    paired_bootstrap_test,
    pairwise_comparison_table,
    tournament_report,
)


def _make_result(
    model_slug: str = "test_model",
    forecasts: dict[str, float] | None = None,
    outcomes: dict[str, int] | None = None,
    sources: dict[str, str] | None = None,
    costs: dict[str, float] | None = None,
) -> ModelResult:
    if forecasts is None:
        forecasts = {"q1": 0.7, "q2": 0.3, "q3": 0.8}
    if outcomes is None:
        outcomes = {"q1": 1, "q2": 0, "q3": 1}
    if sources is None:
        sources = {"q1": "fred", "q2": "metaculus", "q3": "fred"}
    return ModelResult(
        model_slug=model_slug,
        forecasts=forecasts,
        outcomes=outcomes,
        sources=sources or {},
        costs=costs or {},
        scoring_result={"overall_brier": 0.1, "overall_index": 68.4},
        metadata={},
    )


class TestBootstrapCI:
    def test_basic_ci(self) -> None:
        pairs = [(0.7, 1), (0.3, 0), (0.8, 1), (0.2, 0)]
        lo, hi = _bootstrap_ci_brier(pairs, n_bootstrap=500, seed=42)
        assert lo <= hi
        assert 0.0 <= lo <= 1.0
        assert 0.0 <= hi <= 1.0

    def test_empty_pairs(self) -> None:
        lo, hi = _bootstrap_ci_brier([], n_bootstrap=100)
        assert lo == 0.0
        assert hi == 0.0

    def test_deterministic_with_seed(self) -> None:
        pairs = [(0.5, 1), (0.5, 0)] * 20
        lo1, hi1 = _bootstrap_ci_brier(pairs, seed=123)
        lo2, hi2 = _bootstrap_ci_brier(pairs, seed=123)
        assert lo1 == lo2
        assert hi1 == hi2

    def test_perfect_forecasts_narrow_ci(self) -> None:
        pairs = [(1.0, 1), (0.0, 0)] * 50
        lo, hi = _bootstrap_ci_brier(pairs, n_bootstrap=1000, seed=42)
        assert lo == 0.0
        assert hi == 0.0

    def test_ci_contains_point_estimate(self) -> None:
        pairs = [(0.6, 1), (0.4, 0), (0.7, 1), (0.3, 0)] * 10
        lo, hi = _bootstrap_ci_brier(pairs, n_bootstrap=2000, seed=42)
        point = sum((f - o) ** 2 for f, o in pairs) / len(pairs)
        assert lo <= point <= hi


class TestPairedBootstrapTest:
    def test_identical_forecasts(self) -> None:
        fa = {"q1": 0.7, "q2": 0.3}
        fb = {"q1": 0.7, "q2": 0.3}
        outcomes = {"q1": 1, "q2": 0}
        result = paired_bootstrap_test(fa, fb, outcomes, n_bootstrap=500)
        assert result.mean_diff == 0.0
        assert result.p_value >= 0.5

    def test_clearly_better_forecasts(self) -> None:
        fa = {f"q{i}": 0.9 for i in range(100)}
        fb = {f"q{i}": 0.5 for i in range(100)}
        outcomes = {f"q{i}": 1 for i in range(100)}
        result = paired_bootstrap_test(fa, fb, outcomes, n_bootstrap=1000, seed=42)
        assert result.mean_diff < 0
        assert result.p_value < 0.05

    def test_no_shared_questions(self) -> None:
        fa = {"q1": 0.5}
        fb = {"q2": 0.5}
        outcomes = {"q1": 1, "q2": 0}
        result = paired_bootstrap_test(fa, fb, outcomes)
        assert result.n_questions == 0
        assert result.p_value == 1.0

    def test_returns_correct_n(self) -> None:
        fa = {"q1": 0.7, "q2": 0.3, "q3": 0.5}
        fb = {"q1": 0.6, "q2": 0.4}
        outcomes = {"q1": 1, "q2": 0, "q3": 1}
        result = paired_bootstrap_test(fa, fb, outcomes)
        assert result.n_questions == 2


class TestModelSourceMatrix:
    def test_basic_matrix(self) -> None:
        r = _make_result()
        matrix = model_source_matrix([r], n_bootstrap=100)
        assert "test_model" in matrix
        row = matrix["test_model"]
        assert "fred" in row
        assert "overall" in row
        assert row["fred"].count == 2
        assert isinstance(row["fred"].brier_index, float)

    def test_small_n_flag(self) -> None:
        forecasts = {f"q{i}": 0.5 for i in range(50)}
        outcomes = {f"q{i}": i % 2 for i in range(50)}
        sources = {f"q{i}": "tiny" for i in range(50)}
        r = _make_result(forecasts=forecasts, outcomes=outcomes, sources=sources)
        matrix = model_source_matrix([r], n_bootstrap=100)
        assert matrix["test_model"]["tiny"].small_n is True

    def test_large_n_not_flagged(self) -> None:
        forecasts = {f"q{i}": 0.5 for i in range(200)}
        outcomes = {f"q{i}": i % 2 for i in range(200)}
        sources = {f"q{i}": "big" for i in range(200)}
        r = _make_result(forecasts=forecasts, outcomes=outcomes, sources=sources)
        matrix = model_source_matrix([r], n_bootstrap=100)
        assert matrix["test_model"]["big"].small_n is False

    def test_multiple_models(self) -> None:
        r1 = _make_result(model_slug="model_a")
        r2 = _make_result(model_slug="model_b")
        matrix = model_source_matrix([r1, r2], n_bootstrap=100)
        assert "model_a" in matrix
        assert "model_b" in matrix

    def test_ci_ordering(self) -> None:
        r = _make_result()
        matrix = model_source_matrix([r], n_bootstrap=500)
        overall = matrix["test_model"]["overall"]
        assert overall.ci_low <= overall.ci_high


class TestPairwiseComparisonTable:
    def test_two_models(self) -> None:
        r1 = _make_result(model_slug="a")
        r2 = _make_result(model_slug="b")
        table = pairwise_comparison_table([r1, r2], n_bootstrap=500, seed=42)
        assert len(table) == 1
        assert table[0].model_a == "a"
        assert table[0].model_b == "b"
        assert isinstance(table[0].significant, bool)

    def test_three_models(self) -> None:
        r1 = _make_result(model_slug="a")
        r2 = _make_result(model_slug="b")
        r3 = _make_result(model_slug="c")
        table = pairwise_comparison_table([r1, r2, r3], n_bootstrap=100)
        assert len(table) == 3

    def test_single_model_no_comparisons(self) -> None:
        r1 = _make_result()
        table = pairwise_comparison_table([r1])
        assert len(table) == 0


class TestCostAccuracySummary:
    def test_with_costs(self) -> None:
        r = _make_result(costs={"q1": 0.01, "q2": 0.02, "q3": 0.015})
        entries = cost_accuracy_summary([r])
        assert len(entries) == 1
        assert entries[0].total_cost == pytest.approx(0.045)
        assert entries[0].mean_cost == pytest.approx(0.015)
        assert entries[0].n_costed == 3

    def test_without_costs(self) -> None:
        r = _make_result(costs={})
        entries = cost_accuracy_summary([r])
        assert len(entries) == 0

    def test_cost_by_source(self) -> None:
        r = _make_result(
            costs={"q1": 0.01, "q2": 0.02, "q3": 0.015},
            sources={"q1": "fred", "q2": "metaculus", "q3": "fred"},
        )
        entries = cost_accuracy_summary([r])
        assert "fred" in entries[0].by_source
        assert "metaculus" in entries[0].by_source
        assert entries[0].by_source["fred"]["count"] == 2


class TestTournamentReport:
    def test_generates_report(self) -> None:
        r1 = _make_result(model_slug="model_a")
        r2 = _make_result(model_slug="model_b")
        report = tournament_report([r1, r2])
        assert "TOURNAMENT REPORT" in report
        assert "OVERALL RANKINGS" in report
        assert "MODEL x SOURCE MATRIX" in report
        assert "PAIRWISE COMPARISONS" in report
        assert "model_a" in report
        assert "model_b" in report

    def test_empty_results(self) -> None:
        report = tournament_report([])
        assert "No results to report" in report

    def test_single_model(self) -> None:
        r = _make_result()
        report = tournament_report([r])
        assert "OVERALL RANKINGS" in report
        assert "Need 2+ models" in report

    def test_report_with_costs(self) -> None:
        r = _make_result(costs={"q1": 0.01, "q2": 0.02})
        report = tournament_report([r])
        assert "COST SUMMARY" in report


class TestLoadTournamentResults:
    def test_loads_from_directory(self, tmp_path: Path) -> None:
        data = {
            "model_slug": "test",
            "forecasts": {"q1": 0.5},
            "outcomes": {"q1": 1},
            "sources": {"q1": "fred"},
            "scoring_result": {"overall_brier": 0.25},
            "metadata": {},
            "timestamp": "20260101T000000Z",
        }
        (tmp_path / "result.json").write_text(json.dumps(data))
        results = load_tournament_results(tmp_path)
        assert len(results) == 1
        assert results[0].model_slug == "test"

    def test_skips_invalid_json(self, tmp_path: Path) -> None:
        (tmp_path / "bad.json").write_text("not json")
        results = load_tournament_results(tmp_path)
        assert len(results) == 0

    def test_skips_missing_fields(self, tmp_path: Path) -> None:
        (tmp_path / "partial.json").write_text(json.dumps({"model_slug": "x"}))
        results = load_tournament_results(tmp_path)
        assert len(results) == 0

    def test_backward_compatible_no_costs(self, tmp_path: Path) -> None:
        data = {
            "model_slug": "old",
            "forecasts": {"q1": 0.5},
            "outcomes": {"q1": 1},
            "scoring_result": {"overall_brier": 0.25},
            "metadata": {},
        }
        (tmp_path / "old.json").write_text(json.dumps(data))
        results = load_tournament_results(tmp_path)
        assert len(results) == 1
        assert results[0].costs == {}

    def test_empty_dir(self, tmp_path: Path) -> None:
        results = load_tournament_results(tmp_path)
        assert len(results) == 0

    def test_nonexistent_dir(self) -> None:
        results = load_tournament_results("/nonexistent/path")
        assert len(results) == 0


class TestRoundsFilter:
    def test_rounds_filtering(self) -> None:
        from fetch_data import QuestionSet, Question

        question_sets = []
        for i, date in enumerate(["2026-01-01", "2026-02-01", "2026-03-01", "2026-04-01", "2026-05-01"]):
            qs = QuestionSet(
                forecast_due_date=date,
                questions=[Question(id=f"q{i}", source="fred", question="test")],
            )
            question_sets.append(qs)

        sorted_qs = sorted(question_sets, key=lambda qs: qs.forecast_due_date, reverse=True)
        filtered = sorted_qs[:3]
        assert len(filtered) == 3
        dates = [qs.forecast_due_date for qs in filtered]
        assert "2026-05-01" in dates
        assert "2026-04-01" in dates
        assert "2026-03-01" in dates
        assert "2026-01-01" not in dates
