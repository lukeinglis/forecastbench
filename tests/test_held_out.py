"""Leakage test: held-out data must never appear in the iteration scoring path."""

from __future__ import annotations

from fetch_data import QuestionSet, Question, ResolvedQuestion, join_resolved_questions
from eval import split_held_out


class TestHeldOutLeakage:
    def test_no_overlap_between_sets(self, mock_question_sets: list[QuestionSet]) -> None:
        """The iteration set must contain zero questions from the held-out sets."""
        iteration, held_out = split_held_out(mock_question_sets, n_held_out=2)

        iter_question_ids = {q.id for qs in iteration for q in qs.questions}
        held_question_ids = {q.id for qs in held_out for q in qs.questions}

        overlap = iter_question_ids & held_question_ids
        assert overlap == set(), f"Leaked question IDs: {overlap}"

    def test_held_out_excluded_from_scoring_path(self, mock_question_sets: list[QuestionSet]) -> None:
        """Even if resolutions exist for held-out questions, they must not be scored."""
        iteration, held_out = split_held_out(mock_question_sets, n_held_out=2)

        all_question_ids = {q.id for qs in mock_question_sets for q in qs.questions}
        held_question_ids = {q.id for qs in held_out for q in qs.questions}

        class FakeRes:
            def __init__(self, qid: str) -> None:
                self.id = qid
                self.outcome = 1
                self.resolution_date = "2024-06-01"

        all_resolutions = {qid: FakeRes(qid) for qid in all_question_ids}  # type: ignore[dict-item]

        resolved = join_resolved_questions(iteration, all_resolutions)  # type: ignore[arg-type]
        resolved_ids = {q.id for q in resolved}

        leaked = resolved_ids & held_question_ids
        assert leaked == set(), f"Held-out questions leaked into scoring: {leaked}"

    def test_full_coverage(self, mock_question_sets: list[QuestionSet]) -> None:
        """Iteration + held-out must cover all question sets exactly."""
        iteration, held_out = split_held_out(mock_question_sets, n_held_out=2)
        all_dates = {qs.forecast_due_date for qs in mock_question_sets}
        covered = {qs.forecast_due_date for qs in iteration} | {qs.forecast_due_date for qs in held_out}
        assert covered == all_dates

    def test_split_is_deterministic(self, mock_question_sets: list[QuestionSet]) -> None:
        """Running split_held_out twice produces the same partition."""
        iter1, held1 = split_held_out(mock_question_sets, n_held_out=2)
        iter2, held2 = split_held_out(mock_question_sets, n_held_out=2)
        assert [qs.forecast_due_date for qs in iter1] == [qs.forecast_due_date for qs in iter2]
        assert [qs.forecast_due_date for qs in held1] == [qs.forecast_due_date for qs in held2]
