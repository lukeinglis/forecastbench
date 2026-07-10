"""Tests for eval.py."""

from __future__ import annotations

from pathlib import Path

from fetch_data import QuestionSet, Question, ResolvedQuestion
from eval import split_held_out, save_result, load_previous_results
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
            path = save_result(result, forecasts, model_slug, question_sets_used, n_held_out)
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
