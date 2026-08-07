"""Tests for investigate.py diagnostic analysis functions."""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from investigate import (
    brier_index,
    brier_score,
    classify_source,
    compare_round_sizes,
    diagnose_id_mismatch,
    extract_date_suffix,
    extract_round_date,
    format_report,
    is_market_source,
    knowledge_cutoff_analysis,
    load_all_results,
    per_round_breakdown,
    run_investigation,
    stratify_by_source,
    superforecaster_gap,
)


def _make_result(
    forecasts: dict[str, float],
    outcomes: dict[str, int],
    sources: dict[str, str],
    scoring_result: dict | None = None,
    round_name: str | None = None,
) -> dict:
    sr = scoring_result or {
        "dataset_brier": 0.25,
        "dataset_index": 50.0,
        "market_brier": 0.1,
        "market_index": 68.4,
        "overall_brier": 0.175,
        "overall_index": 58.2,
        "n_dataset": len([s for s in sources.values() if s.lower() not in {"metaculus", "polymarket", "manifold", "infer"}]),
        "n_market": len([s for s in sources.values() if s.lower() in {"metaculus", "polymarket", "manifold", "infer"}]),
        "n_missing": len(set(outcomes) - set(forecasts)),
        "difficulty_adjusted": False,
    }
    meta: dict = {"question_sets_used": ["2026-01-01"]}
    if round_name:
        meta["round"] = round_name
    return {
        "timestamp": "20260801T120000Z",
        "model_slug": "test_model",
        "scoring_result": sr,
        "forecasts": forecasts,
        "outcomes": outcomes,
        "sources": sources,
        "metadata": meta,
    }


class TestClassification:
    def test_market_sources(self):
        assert is_market_source("metaculus")
        assert is_market_source("polymarket")
        assert is_market_source("manifold")
        assert is_market_source("infer")

    def test_dataset_sources(self):
        assert not is_market_source("acled")
        assert not is_market_source("fred")
        assert not is_market_source("wikipedia")

    def test_classify_source(self):
        assert classify_source("metaculus") == "market"
        assert classify_source("acled") == "dataset"
        assert classify_source("FRED") == "dataset"

    def test_case_insensitive(self):
        assert is_market_source("Metaculus")
        assert not is_market_source("ACLED")


class TestBrierCalculations:
    def test_perfect_forecast(self):
        assert brier_score(1.0, 1) == pytest.approx(0.0)
        assert brier_score(0.0, 0) == pytest.approx(0.0)

    def test_worst_forecast(self):
        assert brier_score(0.0, 1) == pytest.approx(1.0)
        assert brier_score(1.0, 0) == pytest.approx(1.0)

    def test_uncertain_forecast(self):
        assert brier_score(0.5, 1) == pytest.approx(0.25)
        assert brier_score(0.5, 0) == pytest.approx(0.25)

    def test_brier_index(self):
        assert brier_index(0.25) == pytest.approx((1.0 - math.sqrt(0.25)) * 100.0)

    def test_brier_index_perfect(self):
        assert brier_index(0.0) == pytest.approx(100.0)

    def test_brier_index_uninformed(self):
        assert brier_index(0.25) == pytest.approx(50.0)


class TestExtractDateSuffix:
    def test_composite_id(self):
        assert extract_date_suffix("dq123_2026-03-16") == "2026-03-16"

    def test_plain_id(self):
        assert extract_date_suffix("q123") is None

    def test_multiple_dates(self):
        assert extract_date_suffix("dq_2026-01-01_2026-03-16") == "2026-03-16"

    def test_round_name(self):
        assert extract_date_suffix("2026-03-01-llm") == "2026-03-01"


class TestDiagnoseIdMismatch:
    def test_fully_matched(self):
        result = _make_result(
            forecasts={"q1": 0.7, "q2": 0.3},
            outcomes={"q1": 1, "q2": 0},
            sources={"q1": "metaculus", "q2": "metaculus"},
        )
        diag = diagnose_id_mismatch(result)
        assert diag["matched"] == 2
        assert diag["forecast_only"] == 0
        assert diag["outcome_only"] == 0

    def test_forecast_only_keys(self):
        result = _make_result(
            forecasts={"dq1_2026-08-09": 0.6, "q2": 0.3},
            outcomes={"dq1_2026-03-16": 1, "q2": 0},
            sources={"dq1_2026-08-09": "acled", "dq1_2026-03-16": "acled", "q2": "metaculus"},
        )
        diag = diagnose_id_mismatch(result)
        assert diag["matched"] == 1
        assert diag["forecast_only"] == 1
        assert diag["outcome_only"] == 1

    def test_date_mismatch_detected(self):
        result = _make_result(
            forecasts={"dq1_2026-08-09": 0.6},
            outcomes={"dq1_2026-03-16": 1},
            sources={"dq1_2026-08-09": "acled", "dq1_2026-03-16": "acled"},
        )
        diag = diagnose_id_mismatch(result)
        assert diag["date_mismatch_count"] >= 1
        dm = diag["date_mismatches"][0]
        assert dm["base_id"] == "dq1"
        assert dm["forecast_date"] == "2026-08-09"
        assert dm["outcome_date"] == "2026-03-16"

    def test_empty_result(self):
        result = _make_result(forecasts={}, outcomes={}, sources={})
        diag = diagnose_id_mismatch(result)
        assert diag["matched"] == 0
        assert diag["forecast_only"] == 0
        assert diag["outcome_only"] == 0


class TestStratifyBySource:
    def test_basic_stratification(self):
        result = _make_result(
            forecasts={"q1": 0.9, "q2": 0.1, "q3": 0.7},
            outcomes={"q1": 1, "q2": 0, "q3": 1},
            sources={"q1": "acled", "q2": "metaculus", "q3": "acled"},
        )
        table = stratify_by_source(result)
        assert "acled" in table
        assert "metaculus" in table
        assert table["acled"]["category"] == "dataset"
        assert table["metaculus"]["category"] == "market"
        assert table["acled"]["outcome_count"] == 2
        assert table["metaculus"]["outcome_count"] == 1

    def test_missing_forecast(self):
        result = _make_result(
            forecasts={"q1": 0.9},
            outcomes={"q1": 1, "q2": 0},
            sources={"q1": "acled", "q2": "acled"},
        )
        table = stratify_by_source(result)
        assert table["acled"]["missing_count"] == 1
        assert table["acled"]["missing_rate"] == pytest.approx(0.5)

    def test_no_missing(self):
        result = _make_result(
            forecasts={"q1": 0.5},
            outcomes={"q1": 1},
            sources={"q1": "fred"},
        )
        table = stratify_by_source(result)
        assert table["fred"]["missing_rate"] == pytest.approx(0.0)


class TestExtractRoundDate:
    def test_with_round(self):
        result = _make_result(
            forecasts={}, outcomes={}, sources={}, round_name="2026-03-01-llm",
        )
        assert extract_round_date(result) == "2026-03-01"

    def test_without_round(self):
        result = _make_result(forecasts={}, outcomes={}, sources={})
        assert extract_round_date(result) is None


class TestPerRoundBreakdown:
    def test_basic_round(self):
        result = _make_result(
            forecasts={"q1": 0.7, "m1": 0.6},
            outcomes={"q1": 1, "m1": 0},
            sources={"q1": "acled", "m1": "metaculus"},
            round_name="2026-03-01-llm",
        )
        rounds = per_round_breakdown([result])
        assert len(rounds) == 1
        r = rounds[0]
        assert r["round_date"] == "2026-03-01"
        assert r["n_dataset"] == 1
        assert r["n_market"] == 1
        assert r["size_category"] == "500q"

    def test_large_round_detected(self):
        result = _make_result(
            forecasts={"q1": 0.5},
            outcomes={"q1": 1},
            sources={"q1": "acled"},
            round_name="2025-08-17-llm",
        )
        rounds = per_round_breakdown([result])
        assert rounds[0]["size_category"] == "1000q"

    def test_no_round(self):
        result = _make_result(forecasts={}, outcomes={}, sources={})
        assert per_round_breakdown([result]) == []


class TestKnowledgeCutoffAnalysis:
    def test_pre_and_post_cutoff(self):
        result = _make_result(
            forecasts={
                "q1_2025-01-15": 0.8,
                "q2_2025-06-01": 0.6,
            },
            outcomes={
                "q1_2025-01-15": 1,
                "q2_2025-06-01": 0,
            },
            sources={
                "q1_2025-01-15": "metaculus",
                "q2_2025-06-01": "metaculus",
            },
        )
        analysis = knowledge_cutoff_analysis(result)
        assert analysis["pre_cutoff_all"]["count"] == 1
        assert analysis["post_cutoff_all"]["count"] == 1
        assert analysis["pre_cutoff_market"]["count"] == 1
        assert analysis["post_cutoff_market"]["count"] == 1

    def test_no_dated_keys(self):
        result = _make_result(
            forecasts={"q1": 0.5},
            outcomes={"q1": 1},
            sources={"q1": "metaculus"},
        )
        analysis = knowledge_cutoff_analysis(result)
        assert analysis["pre_cutoff_all"]["count"] == 0
        assert analysis["post_cutoff_all"]["count"] == 0


class TestSuperforecasterGap:
    def test_gap_calculation(self):
        sr = {
            "dataset_brier": 0.15,
            "dataset_index": 61.0,
            "market_brier": 0.08,
            "market_index": 71.0,
            "overall_brier": 0.115,
            "overall_index": 66.1,
            "n_dataset": 100,
            "n_market": 50,
            "n_missing": 0,
            "difficulty_adjusted": False,
        }
        result = _make_result(
            forecasts={"q1": 0.5},
            outcomes={"q1": 1},
            sources={"q1": "acled"},
            scoring_result=sr,
        )
        gap = superforecaster_gap(result)
        assert gap["gap_overall"] == pytest.approx(66.1 - 68.2)
        assert gap["gap_dataset"] == pytest.approx(61.0 - 63.9)
        assert gap["gap_market"] == pytest.approx(71.0 - 73.1)


class TestCompareRoundSizes:
    def test_comparison(self):
        rounds = [
            {"round_date": "2025-08-17", "total": 1000, "n_dataset": 800,
             "n_market": 200, "missing_rate": 0.5, "dataset_index": 48.0,
             "market_index": 55.0, "overall_index": 51.5, "size_category": "1000q"},
            {"round_date": "2026-03-01", "total": 500, "n_dataset": 350,
             "n_market": 150, "missing_rate": 0.3, "dataset_index": 55.0,
             "market_index": 65.0, "overall_index": 60.0, "size_category": "500q"},
        ]
        comparison = compare_round_sizes(rounds)
        assert comparison["large_rounds"]["count"] == 1
        assert comparison["small_rounds"]["count"] == 1
        assert comparison["large_rounds"]["avg_overall_index"] == pytest.approx(51.5)
        assert comparison["small_rounds"]["avg_overall_index"] == pytest.approx(60.0)

    def test_empty(self):
        comparison = compare_round_sizes([])
        assert comparison["large_rounds"]["count"] == 0
        assert comparison["small_rounds"]["count"] == 0


class TestRunInvestigation:
    def test_empty_results(self):
        report = run_investigation([])
        assert "error" in report["summary"]

    def test_full_investigation(self):
        result = _make_result(
            forecasts={"dq1_2026-08-09": 0.6, "m1": 0.7},
            outcomes={"dq1_2026-03-16": 1, "m1": 0},
            sources={
                "dq1_2026-08-09": "acled",
                "dq1_2026-03-16": "acled",
                "m1": "metaculus",
            },
            round_name="2026-03-01-llm",
        )
        report = run_investigation([result])
        assert "id_mismatch" in report["analyses"]
        assert "source_stratification" in report["analyses"]
        assert "per_round" in report["analyses"]
        assert "knowledge_cutoff" in report["analyses"]
        assert "superforecaster_gap" in report["analyses"]
        assert "round_size_comparison" in report["analyses"]
        assert "findings" in report["summary"]
        assert "recommendations" in report["summary"]

    def test_mismatch_detected_in_summary(self):
        result = _make_result(
            forecasts={"dq1_2026-08-09": 0.6},
            outcomes={"dq1_2026-03-16": 1},
            sources={"dq1_2026-08-09": "acled", "dq1_2026-03-16": "acled"},
        )
        report = run_investigation([result])
        findings = report["summary"]["findings"]
        assert any("MISMATCH" in f for f in findings)


class TestFormatReport:
    def test_format_produces_output(self):
        result = _make_result(
            forecasts={"q1": 0.7, "m1": 0.6},
            outcomes={"q1": 1, "m1": 0},
            sources={"q1": "acled", "m1": "metaculus"},
            round_name="2026-03-01-llm",
        )
        report = run_investigation([result])
        text = format_report(report)
        assert "PARITY INVESTIGATION REPORT" in text
        assert "MULTI-HORIZON" in text
        assert "SOURCE-LEVEL" in text
        assert "PER-ROUND" in text
        assert "KNOWLEDGE CUTOFF" in text
        assert "SUPERFORECASTER" in text
        assert "FINDINGS" in text

    def test_format_empty_report(self):
        report = run_investigation([])
        text = format_report(report)
        assert "PARITY INVESTIGATION REPORT" in text


class TestLoadResults:
    def test_load_from_empty_dir(self, tmp_path: Path):
        assert load_all_results(tmp_path) == []

    def test_load_nonexistent_dir(self, tmp_path: Path):
        assert load_all_results(tmp_path / "nope") == []

    def test_load_valid_files(self, tmp_path: Path):
        result = _make_result(
            forecasts={"q1": 0.5},
            outcomes={"q1": 1},
            sources={"q1": "acled"},
        )
        (tmp_path / "test.json").write_text(json.dumps(result))
        loaded = load_all_results(tmp_path)
        assert len(loaded) == 1
        assert loaded[0]["model_slug"] == "test_model"

    def test_skip_invalid_json(self, tmp_path: Path):
        (tmp_path / "bad.json").write_text("{invalid json")
        assert load_all_results(tmp_path) == []
