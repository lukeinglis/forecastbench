"""Tests for dashboard data model — aggregate run grouping."""

from __future__ import annotations

from typing import Any

def _make_result(
    slug: str,
    round_name: str,
    forecasts: dict[str, float] | None = None,
    outcomes: dict[str, int] | None = None,
    sources: dict[str, str] | None = None,
    overall_brier: float = 0.2,
) -> dict[str, Any]:
    if forecasts is None:
        forecasts = {"q1": 0.7, "q2": 0.3}
    if outcomes is None:
        outcomes = {"q1": 1, "q2": 0}
    if sources is None:
        sources = {"q1": "metaculus", "q2": "fred"}
    return {
        "model_slug": slug,
        "forecasts": forecasts,
        "outcomes": outcomes,
        "sources": sources,
        "scoring_result": {
            "overall_brier": overall_brier,
            "overall_index": 55.0,
            "dataset_brier": 0.2,
            "dataset_index": 55.0,
            "market_brier": 0.2,
            "market_index": 55.0,
            "n_dataset": 1,
            "n_market": 1,
            "n_missing": 0,
        },
        "metadata": {
            "round": round_name,
            "question_sets_used": [round_name.replace("-llm", "")],
        },
        "_filename": f"test_{slug}_{round_name}.json",
    }


def test_group_results_single_slug() -> None:
    from dashboard import _group_results_into_runs

    results = [
        _make_result("model_a", "2026-01-01-llm", {"q1": 0.7}, {"q1": 1}, {"q1": "metaculus"}),
        _make_result("model_a", "2026-01-15-llm", {"q2": 0.3}, {"q2": 0}, {"q2": "fred"}),
    ]
    runs = _group_results_into_runs(results)

    assert len(runs) == 1
    assert runs[0].slug == "model_a"
    assert runs[0].n_rounds == 2
    assert "q1" in runs[0].combined_forecasts
    assert "q2" in runs[0].combined_forecasts
    assert runs[0].combined_forecasts["q1"] == 0.7
    assert runs[0].combined_forecasts["q2"] == 0.3


def test_group_results_multiple_slugs() -> None:
    from dashboard import _group_results_into_runs

    results = [
        _make_result("model_a", "2026-01-01-llm"),
        _make_result("model_b", "2026-01-01-llm"),
        _make_result("model_a", "2026-01-15-llm"),
    ]
    runs = _group_results_into_runs(results)

    assert len(runs) == 2
    slugs = {r.slug for r in runs}
    assert slugs == {"model_a", "model_b"}

    run_a = next(r for r in runs if r.slug == "model_a")
    assert run_a.n_rounds == 2

    run_b = next(r for r in runs if r.slug == "model_b")
    assert run_b.n_rounds == 1


def test_group_results_combines_forecasts() -> None:
    from dashboard import _group_results_into_runs

    results = [
        _make_result(
            "model_a", "round1",
            forecasts={"q1": 0.7, "q2": 0.3},
            outcomes={"q1": 1, "q2": 0},
            sources={"q1": "metaculus", "q2": "fred"},
        ),
        _make_result(
            "model_a", "round2",
            forecasts={"q3": 0.8, "q4": 0.1},
            outcomes={"q3": 1, "q4": 0},
            sources={"q3": "metaculus", "q4": "fred"},
        ),
    ]
    runs = _group_results_into_runs(results)
    assert len(runs) == 1
    run = runs[0]
    assert set(run.combined_forecasts.keys()) == {"q1", "q2", "q3", "q4"}
    assert set(run.combined_outcomes.keys()) == {"q1", "q2", "q3", "q4"}


def test_aggregate_scoring_computed() -> None:
    from dashboard import _group_results_into_runs

    results = [
        _make_result(
            "model_a", "round1",
            forecasts={"q1": 0.9},
            outcomes={"q1": 1},
            sources={"q1": "fred"},
        ),
    ]
    runs = _group_results_into_runs(results)
    sr = runs[0].scoring_result
    assert sr["n_dataset"] == 1
    assert sr["n_market"] == 0
    assert sr["dataset_brier"] < 0.05


def test_empty_results() -> None:
    from dashboard import _group_results_into_runs

    runs = _group_results_into_runs([])
    assert runs == []


def test_round_name_from_result() -> None:
    from dashboard import _round_name_from_result

    r1 = {"metadata": {"round": "2026-01-01-llm"}}
    assert _round_name_from_result(r1) == "2026-01-01-llm"

    r2 = {"metadata": {"question_sets_used": ["2026-01-01", "2026-01-15"]}}
    assert "2026-01-01" in _round_name_from_result(r2)
    assert "2026-01-15" in _round_name_from_result(r2)

    r3 = {"metadata": {}, "_filename": "test.json"}
    assert _round_name_from_result(r3) == "test.json"


def test_aggregate_label() -> None:
    from dashboard import _group_results_into_runs

    results = [_make_result("my_model", "round1")]
    runs = _group_results_into_runs(results)
    assert runs[0].label == "my_model"


def test_per_round_results_preserved() -> None:
    from dashboard import _group_results_into_runs

    r1 = _make_result("m", "r1")
    r2 = _make_result("m", "r2")
    runs = _group_results_into_runs([r1, r2])
    assert len(runs[0].per_round_results) == 2


def test_compute_aggregate_scoring_market_and_dataset() -> None:
    from dashboard import _compute_aggregate_scoring

    forecasts = {"q1": 0.9, "q2": 0.1}
    outcomes = {"q1": 1, "q2": 0}
    sources = {"q1": "metaculus", "q2": "fred"}
    sr = _compute_aggregate_scoring(forecasts, outcomes, sources)

    assert sr["n_market"] == 1
    assert sr["n_dataset"] == 1
    assert sr["overall_brier"] < 0.05


def test_compute_aggregate_scoring_no_shared() -> None:
    from dashboard import _compute_aggregate_scoring

    sr = _compute_aggregate_scoring({"q1": 0.5}, {"q2": 1}, {})
    assert sr["n_dataset"] == 0
    assert sr["n_market"] == 0
