"""Tests for eval.py."""

from __future__ import annotations


from fetch_data import QuestionSet, Question, ResolvedQuestion
from eval import split_held_out


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
