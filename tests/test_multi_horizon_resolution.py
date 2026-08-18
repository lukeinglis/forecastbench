"""Tests for multi-horizon resolution matching (issue #94).

Verifies that dataset questions with multiple resolution dates get correct
per-horizon entries via join_resolved_questions (parity v0.2.0), and that
market questions remain unaffected.

Note: join_resolved_questions from forecastbench-parity v0.2.0 takes
dict[str, list[Resolution]] and performs per-horizon expansion internally.
"""

from __future__ import annotations

from fetch_data import (
    Question,
    QuestionSet,
    Resolution,
    join_resolved_questions,
)


class TestJoinResolvedQuestionsBasic:
    """join_resolved_questions with parity v0.2.0 API (dict[str, list[Resolution]])."""

    def _make_qs(self, question_id: str = "q1", source: str = "fred") -> QuestionSet:
        return QuestionSet(
            forecast_due_date="2024-06-01",
            question_set="round_1",
            questions=[
                Question(
                    id=question_id, source=source, question="Will X?",
                    resolution_dates=["2024-07-01", "2024-08-30", "2024-11-28"],
                )
            ],
        )

    def test_single_resolution_per_question(self) -> None:
        qs = self._make_qs()
        resolutions: dict[str, list[Resolution]] = {
            "q1": [Resolution(id="q1", outcome=0, resolution_date="2024-07-01")],
        }
        result = join_resolved_questions([qs], resolutions)
        assert len(result) == 1
        assert result[0].outcome == 0

    def test_multiple_resolutions_per_question(self) -> None:
        qs = self._make_qs()
        resolutions: dict[str, list[Resolution]] = {
            "q1": [
                Resolution(id="q1", outcome=0, resolution_date="2024-07-01"),
                Resolution(id="q1", outcome=1, resolution_date="2024-08-30"),
            ],
        }
        result = join_resolved_questions([qs], resolutions)
        assert len(result) == 2
        outcomes = {r.resolution_date: r.outcome for r in result}
        assert outcomes["2024-07-01"] == 0
        assert outcomes["2024-08-30"] == 1

    def test_market_question_single_resolution(self) -> None:
        qs = QuestionSet(
            forecast_due_date="2024-06-01",
            question_set="round_1",
            questions=[
                Question(id="m1", source="metaculus", question="Market Q?")
            ],
        )
        resolutions: dict[str, list[Resolution]] = {
            "m1": [Resolution(id="m1", outcome=1, resolution_date="2024-06-15")],
        }
        result = join_resolved_questions([qs], resolutions)
        assert len(result) == 1
        assert result[0].outcome == 1
        assert result[0].source == "metaculus"

    def test_no_matching_resolution(self) -> None:
        qs = self._make_qs(question_id="q999")
        resolutions: dict[str, list[Resolution]] = {
            "q1": [Resolution(id="q1", outcome=1)],
        }
        result = join_resolved_questions([qs], resolutions)
        assert len(result) == 0

    def test_na_resolution_dates_no_filtering(self) -> None:
        qs = QuestionSet(
            forecast_due_date="2024-06-01",
            question_set="round_1",
            questions=[
                Question(id="q1", source="fred", question="Will X?",
                         resolution_dates="N/A")
            ],
        )
        resolutions: dict[str, list[Resolution]] = {
            "q1": [Resolution(id="q1", outcome=1, resolution_date="2024-07-01")],
        }
        result = join_resolved_questions([qs], resolutions)
        assert len(result) == 1


class TestJoinResolvedPreservesBaseId:
    """After v0.2.0, join_resolved_questions preserves base IDs (no composite)."""

    def test_per_date_entries_keep_base_ids(self) -> None:
        qs = QuestionSet(
            forecast_due_date="2024-06-01",
            question_set="round_1",
            questions=[
                Question(id="q1", source="fred", question="Q",
                         resolution_dates=["2024-07-01", "2024-08-30"]),
            ],
        )
        resolutions: dict[str, list[Resolution]] = {
            "q1": [
                Resolution(id="q1", outcome=0, resolution_date="2024-07-01"),
                Resolution(id="q1", outcome=1, resolution_date="2024-08-30"),
            ],
        }
        result = join_resolved_questions([qs], resolutions)
        assert len(result) == 2
        assert all(rq.id == "q1" for rq in result)

    def test_per_date_outcomes_preserved(self) -> None:
        qs = QuestionSet(
            forecast_due_date="2024-06-01",
            question_set="round_1",
            questions=[
                Question(id="q1", source="fred", question="Q",
                         resolution_dates=["2024-07-01", "2024-08-30"]),
            ],
        )
        resolutions: dict[str, list[Resolution]] = {
            "q1": [
                Resolution(id="q1", outcome=0, resolution_date="2024-07-01"),
                Resolution(id="q1", outcome=1, resolution_date="2024-08-30"),
            ],
        }
        result = join_resolved_questions([qs], resolutions)
        by_date = {rq.resolution_date: rq.outcome for rq in result}
        assert by_date["2024-07-01"] == 0
        assert by_date["2024-08-30"] == 1

    def test_market_questions_pass_through(self) -> None:
        qs = QuestionSet(
            forecast_due_date="2024-06-01",
            question_set="round_1",
            questions=[
                Question(id="m1", source="metaculus", question="M1?"),
                Question(id="m2", source="manifold", question="M2?"),
                Question(id="m3", source="infer", question="M3?"),
            ],
        )
        resolutions: dict[str, list[Resolution]] = {
            "m1": [Resolution(id="m1", outcome=1)],
            "m2": [Resolution(id="m2", outcome=0)],
            "m3": [Resolution(id="m3", outcome=1)],
        }
        result = join_resolved_questions([qs], resolutions)
        ids = {rq.id for rq in result}
        assert "m1" in ids
        assert "m2" in ids
        assert "m3" in ids
        assert len(result) == 3

    def test_resolution_date_filtering(self) -> None:
        """Resolutions with dates not in resolution_dates are excluded."""
        qs = QuestionSet(
            forecast_due_date="2024-06-01",
            question_set="round_1",
            questions=[
                Question(id="q1", source="fred", question="Q",
                         resolution_dates=["2024-07-01"]),
            ],
        )
        resolutions: dict[str, list[Resolution]] = {
            "q1": [
                Resolution(id="q1", outcome=0, resolution_date="2024-07-01"),
                Resolution(id="q1", outcome=1, resolution_date="2024-12-01"),
            ],
        }
        result = join_resolved_questions([qs], resolutions)
        assert len(result) == 1
        assert result[0].resolution_date == "2024-07-01"
