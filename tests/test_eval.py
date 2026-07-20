"""Tests for eval.py."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from unittest.mock import patch

from fetch_data import QuestionSet, Question, ResolvedQuestion
from eval import split_held_out, save_result, load_previous_results, run_eval, _run_sync
from score import ScoringResult


class TestSplitHeldOut:
    def test_disjoint_and_union(self, mock_question_sets: list[QuestionSet]) -> None:
        iteration, held_out = split_held_out(mock_question_sets, n_held_out=2)
        iter_ids = {qs.forecast_due_date for qs in iteration}
        held_ids = {qs.forecast_due_date for qs in held_out}
        assert iter_ids & held_ids == set()
        assert iter_ids | held_ids == {qs.forecast_due_date for qs in mock_question_sets}

    def test_held_out_count(self, mock_question_sets: list[QuestionSet]) -> None:
        _, held_out = split_held_out(mock_question_sets, n_held_out=2)
        assert len(held_out) == 2

    def test_held_out_are_most_recent(self, mock_question_sets: list[QuestionSet]) -> None:
        _, held_out = split_held_out(mock_question_sets, n_held_out=2)
        held_dates = sorted(qs.forecast_due_date for qs in held_out)
        assert held_dates == ["2024-04-01", "2024-05-01"]

    def test_zero_held_out(self, mock_question_sets: list[QuestionSet]) -> None:
        iteration, held_out = split_held_out(mock_question_sets, n_held_out=0)
        assert len(held_out) == 0
        assert len(iteration) == len(mock_question_sets)

    def test_all_held_out(self, mock_question_sets: list[QuestionSet]) -> None:
        iteration, held_out = split_held_out(mock_question_sets, n_held_out=5)
        assert len(iteration) == 0
        assert len(held_out) == 5


class TestDummyForecasterIntegration:
    def test_dummy_produces_near_50_percent(self) -> None:
        resolved = [
            ResolvedQuestion(
                id=f"q{i}",
                source="acled",
                question=f"Q{i}",
                outcome=outcome,
                forecast_due_date="2024-01-01",
            )
            for i, outcome in enumerate([1, 0, 1, 0, 1, 0, 1, 0, 1, 0])
        ]

        from dummy_forecaster import forecast
        from score import score_forecasts

        forecasts = {q.id: forecast(Question(id=q.id, source=q.source, question=q.question)) for q in resolved}
        result = score_forecasts(forecasts, resolved)
        assert abs(result.dataset_brier - 0.25) < 1e-10
        assert abs(result.dataset_index - 50.0) < 1.0


class TestResultPersistence:
    def test_save_and_load_results(self, tmp_path: Path) -> None:
        result = ScoringResult(
            dataset_brier=0.25,
            dataset_index=50.0,
            market_brier=0.30,
            market_index=45.2,
            overall_brier=0.275,
            overall_index=47.6,
            n_dataset=5,
            n_market=3,
            n_missing=1,
            difficulty_adjusted=False,
        )
        forecasts = {"q1": 0.7, "q2": 0.3}
        model_slug = "test_model"
        question_sets_used = ["2024-01-01", "2024-02-01"]
        n_held_out = 2

        # Monkey-patch RESULTS_DIR for this test
        import eval as eval_mod
        original_dir = eval_mod.RESULTS_DIR
        eval_mod.RESULTS_DIR = tmp_path
        try:
            outcomes = {"q1": 1, "q2": 0}
            path = save_result(result, forecasts, outcomes, model_slug, question_sets_used, n_held_out)
            assert path.exists()
            assert path.suffix == ".json"

            loaded = load_previous_results(tmp_path)
            assert len(loaded) == 1
            data = loaded[0]
            assert data["model_slug"] == "test_model"
            assert data["scoring_result"]["dataset_brier"] == 0.25
            assert data["scoring_result"]["n_dataset"] == 5
            assert data["scoring_result"]["n_missing"] == 1
            assert data["scoring_result"]["difficulty_adjusted"] is False
            assert data["forecasts"] == {"q1": 0.7, "q2": 0.3}
            assert data["outcomes"] == {"q1": 1, "q2": 0}
            assert data["metadata"]["n_questions"] == 8
            assert data["metadata"]["n_held_out"] == 2
            assert data["metadata"]["question_sets_used"] == ["2024-01-01", "2024-02-01"]
        finally:
            eval_mod.RESULTS_DIR = original_dir

    def test_load_previous_results_empty_dir(self, tmp_path: Path) -> None:
        results = load_previous_results(tmp_path)
        assert results == []

    def test_load_previous_results_no_dir(self, tmp_path: Path) -> None:
        nonexistent = tmp_path / "does_not_exist"
        results = load_previous_results(nonexistent)
        assert results == []


def _make_resolved_questions() -> list[ResolvedQuestion]:
    """Create simple resolved questions for wiring tests."""
    return [
        ResolvedQuestion(
            id="q0", source="acled", question="Q0", outcome=1, forecast_due_date="2024-01-01",
        ),
        ResolvedQuestion(
            id="q1", source="acled", question="Q1", outcome=0, forecast_due_date="2024-01-01",
        ),
        ResolvedQuestion(
            id="q2", source="metaculus", question="Q2", outcome=1, forecast_due_date="2024-01-01",
        ),
    ]


def _make_question_sets(resolved: list[ResolvedQuestion]) -> list[QuestionSet]:
    """Wrap resolved questions into question sets for run_eval."""
    questions = [
        Question(id=rq.id, source=rq.source, question=rq.question)
        for rq in resolved
    ]
    return [
        QuestionSet(
            forecast_due_date="2024-01-01",
            question_set="set_0",
            questions=questions,
        ),
        QuestionSet(
            forecast_due_date="2024-02-01",
            question_set="set_1",
            questions=[],
        ),
        QuestionSet(
            forecast_due_date="2024-03-01",
            question_set="set_2",
            questions=[],
        ),
    ]


def _write_fake_result(results_dir: Path, slug: str, forecasts: dict[str, float]) -> None:
    """Write a fake result JSON file for peer pool loading."""
    payload = {
        "timestamp": "20240101T000000Z",
        "model_slug": slug,
        "scoring_result": {
            "dataset_brier": 0.25, "dataset_index": 50.0,
            "market_brier": 0.25, "market_index": 50.0,
            "overall_brier": 0.25, "overall_index": 50.0,
            "n_dataset": 2, "n_market": 1, "n_missing": 0,
            "difficulty_adjusted": False,
        },
        "forecasts": forecasts,
        "metadata": {"n_questions": 3, "n_held_out": 2, "question_sets_used": ["2024-01-01"]},
    }
    (results_dir / f"20240101T000000Z_{slug}.json").write_text(json.dumps(payload))


def _dummy_forecaster(question: Question, resolution_date: str | None = None, **kwargs: object) -> float:
    return 0.5


class TestDifficultyAdjustmentWiring:
    def test_raw_mode_skips_adjustment(self, tmp_path: Path, monkeypatch: object) -> None:
        """With --raw, difficulty adjustment is skipped even with peer pool."""
        import eval as eval_mod

        resolved = _make_resolved_questions()
        question_sets = _make_question_sets(resolved)

        # Write 2 fake peer results
        results_dir = tmp_path / "results"
        results_dir.mkdir()
        _write_fake_result(results_dir, "peer_a", {"q0": 0.9, "q1": 0.1, "q2": 0.8})
        _write_fake_result(results_dir, "peer_b", {"q0": 0.7, "q1": 0.3, "q2": 0.6})

        monkeypatch.setattr(eval_mod, "RESULTS_DIR", results_dir)  # type: ignore[attr-defined]
        monkeypatch.setattr(eval_mod, "load_data", lambda: (question_sets, resolved))  # type: ignore[attr-defined]
        monkeypatch.setattr(eval_mod, "CACHE_DIR", tmp_path / "cache")  # type: ignore[attr-defined]

        eval_result = asyncio.run(run_eval(_dummy_forecaster, n_held_out=2, raw=True))
        assert eval_result.scoring.difficulty_adjusted is False

    def test_adjustment_with_peer_pool(self, tmp_path: Path, monkeypatch: object) -> None:
        """With 2+ results, difficulty adjustment activates."""
        import eval as eval_mod

        resolved = _make_resolved_questions()
        question_sets = _make_question_sets(resolved)

        results_dir = tmp_path / "results"
        results_dir.mkdir()
        _write_fake_result(results_dir, "peer_a", {"q0": 0.9, "q1": 0.1, "q2": 0.8})
        _write_fake_result(results_dir, "peer_b", {"q0": 0.7, "q1": 0.3, "q2": 0.6})

        monkeypatch.setattr(eval_mod, "RESULTS_DIR", results_dir)  # type: ignore[attr-defined]
        monkeypatch.setattr(eval_mod, "load_data", lambda: (question_sets, resolved))  # type: ignore[attr-defined]
        monkeypatch.setattr(eval_mod, "CACHE_DIR", tmp_path / "cache")  # type: ignore[attr-defined]

        eval_result = asyncio.run(run_eval(_dummy_forecaster, n_held_out=2, raw=False))
        assert eval_result.scoring.difficulty_adjusted is True

    def test_no_adjustment_without_peer_pool(self, tmp_path: Path, monkeypatch: object) -> None:
        """With <2 results, difficulty adjustment is skipped."""
        import eval as eval_mod

        resolved = _make_resolved_questions()
        question_sets = _make_question_sets(resolved)

        results_dir = tmp_path / "results"
        results_dir.mkdir()
        # Only 1 peer result -- not enough
        _write_fake_result(results_dir, "peer_a", {"q0": 0.9, "q1": 0.1, "q2": 0.8})

        monkeypatch.setattr(eval_mod, "RESULTS_DIR", results_dir)  # type: ignore[attr-defined]
        monkeypatch.setattr(eval_mod, "load_data", lambda: (question_sets, resolved))  # type: ignore[attr-defined]
        monkeypatch.setattr(eval_mod, "CACHE_DIR", tmp_path / "cache")  # type: ignore[attr-defined]

        eval_result = asyncio.run(run_eval(_dummy_forecaster, n_held_out=2, raw=False))
        assert eval_result.scoring.difficulty_adjusted is False


def _raising_forecaster(question: Question, resolution_date: str | None = None, **kwargs: object) -> float:
    raise RuntimeError("LLM API timeout")


class TestForecastErrorFallback:
    def test_multi_horizon_error_skips_question(self) -> None:
        """A forecaster that raises on multi-horizon question should skip it."""
        q = Question(
            id="mh1", source="acled", question="MH?",
            resolution_dates=["2024-01-01"],
        )
        with patch("eval._read_cache", return_value=None), \
             patch("eval._write_cache"):
            forecasts = _run_sync(_raising_forecaster, [q], "test_slug")
        assert "mh1_2024-01-01" not in forecasts

    def test_multi_horizon_error_logs_warning(self) -> None:
        """A skipped question should log a warning with question_id."""
        import eval as eval_mod

        q = Question(
            id="mh2", source="acled", question="MH?",
            resolution_dates=["2024-06-01"],
        )
        log_events: list[dict[str, object]] = []
        original_logger = eval_mod.logger

        class CapturingLogger:
            def __getattr__(self, name: str):  # type: ignore[no-untyped-def]
                def log_method(event: str, **kwargs: object) -> None:
                    log_events.append({"event": event, **kwargs})
                return log_method

        eval_mod.logger = CapturingLogger()  # type: ignore[assignment]
        try:
            with patch("eval._read_cache", return_value=None), \
                 patch("eval._write_cache"):
                _run_sync(_raising_forecaster, [q], "test_slug")
        finally:
            eval_mod.logger = original_logger  # type: ignore[assignment]

        skip_events = [e for e in log_events if "skip" in e["event"]]
        assert len(skip_events) == 1
        assert skip_events[0]["question_id"] == "mh2"


class TestEvalResultCompositeIds:
    def test_resolved_has_composite_ids_for_multi_horizon(self, tmp_path: Path, monkeypatch: object) -> None:
        """run_eval should return expanded_resolved with composite IDs for dataset questions."""
        import eval as eval_mod

        resolved = [
            ResolvedQuestion(
                id="dq1", source="fred", question="Dataset Q",
                outcome=1, forecast_due_date="2024-01-01",
                resolution_dates=["2024-07-01", "2024-08-01"],
            ),
            ResolvedQuestion(
                id="mq1", source="metaculus", question="Market Q",
                outcome=0, forecast_due_date="2024-01-01",
            ),
        ]
        question_sets = [
            QuestionSet(
                forecast_due_date="2024-01-01",
                question_set="set_0",
                questions=[
                    Question(id=rq.id, source=rq.source, question=rq.question,
                             resolution_dates=rq.resolution_dates)
                    for rq in resolved
                ],
            ),
            QuestionSet(forecast_due_date="2024-02-01", question_set="set_1", questions=[]),
            QuestionSet(forecast_due_date="2024-03-01", question_set="set_2", questions=[]),
        ]

        results_dir = tmp_path / "results"
        results_dir.mkdir()

        monkeypatch.setattr(eval_mod, "RESULTS_DIR", results_dir)
        monkeypatch.setattr(eval_mod, "load_data", lambda: (question_sets, resolved))
        monkeypatch.setattr(eval_mod, "CACHE_DIR", tmp_path / "cache")

        eval_result = asyncio.run(run_eval(_dummy_forecaster, n_held_out=2, raw=True))

        resolved_ids = {rq.id for rq in eval_result.resolved}
        assert "dq1_2024-07-01" in resolved_ids
        assert "dq1_2024-08-01" in resolved_ids
        assert "mq1" in resolved_ids
        assert "dq1" not in resolved_ids


class TestDifficultyAdjustmentLogging:
    def test_skip_message_includes_reason(self, tmp_path: Path, monkeypatch: object, caplog: object) -> None:
        """difficulty_adjustment_skipped log should include reason field."""
        import eval as eval_mod

        resolved = _make_resolved_questions()
        question_sets = _make_question_sets(resolved)

        results_dir = tmp_path / "results"
        results_dir.mkdir()
        _write_fake_result(results_dir, "peer_a", {"q0": 0.9, "q1": 0.1, "q2": 0.8})

        monkeypatch.setattr(eval_mod, "RESULTS_DIR", results_dir)  # type: ignore[attr-defined]
        monkeypatch.setattr(eval_mod, "load_data", lambda: (question_sets, resolved))  # type: ignore[attr-defined]
        monkeypatch.setattr(eval_mod, "CACHE_DIR", tmp_path / "cache")  # type: ignore[attr-defined]

        log_events: list[dict[str, object]] = []
        original_logger = eval_mod.logger

        class CapturingLogger:
            def __getattr__(self, name: str):  # type: ignore[no-untyped-def]
                def log_method(event: str, **kwargs: object) -> None:
                    log_events.append({"event": event, **kwargs})
                return log_method

        monkeypatch.setattr(eval_mod, "logger", CapturingLogger())  # type: ignore[attr-defined]
        try:
            asyncio.run(run_eval(_dummy_forecaster, n_held_out=2, raw=False))
        finally:
            monkeypatch.setattr(eval_mod, "logger", original_logger)  # type: ignore[attr-defined]

        skip_events = [e for e in log_events if e["event"] == "difficulty_adjustment_skipped"]
        assert len(skip_events) == 1
        assert skip_events[0]["reason"] == "need_at_least_2_prior_results"
        assert skip_events[0]["note"] == "scores_not_difficulty_adjusted_this_run"
