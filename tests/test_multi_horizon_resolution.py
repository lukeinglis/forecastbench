"""Tests for multi-horizon resolution matching (issue #94).

Verifies that dataset questions with multiple resolution dates get correct
per-date outcomes via _expand_resolved_for_horizons, and that market questions
remain unaffected.

Note: join_resolved_questions from forecastbench-parity takes dict[str, Resolution]
(single resolution per ID). Multi-horizon handling is done at the eval layer
via _expand_resolved_for_horizons.
"""

from __future__ import annotations

from fetch_data import (
    Question,
    QuestionSet,
    Resolution,
    ResolvedQuestion,
    join_resolved_questions,
)
from eval import _expand_resolved_for_horizons


class TestJoinResolvedQuestionsBasic:
    """join_resolved_questions with parity API (dict[str, Resolution])."""

    def _make_qs(self, question_id: str = "q1", source: str = "fred") -> QuestionSet:
        return QuestionSet(
            forecast_due_date="2024-06-01",
            question_set="round_1",
            questions=[
                Question(
                    id=question_id, source=source, question="Will X?",
                    resolution_dates=["2024-07-01", "2024-08-01", "2024-09-01"],
                )
            ],
        )

    def test_single_resolution_per_question(self) -> None:
        qs = self._make_qs()
        resolutions = {
            "q1": Resolution(id="q1", outcome=0, resolution_date="2024-07-01"),
        }
        result = join_resolved_questions([qs], resolutions)
        assert len(result) == 1
        assert result[0].outcome == 0

    def test_market_question_single_resolution(self) -> None:
        qs = QuestionSet(
            forecast_due_date="2024-06-01",
            question_set="round_1",
            questions=[
                Question(id="m1", source="metaculus", question="Market Q?")
            ],
        )
        resolutions = {
            "m1": Resolution(id="m1", outcome=1, resolution_date="2024-06-15"),
        }
        result = join_resolved_questions([qs], resolutions)
        assert len(result) == 1
        assert result[0].outcome == 1
        assert result[0].source == "metaculus"

    def test_no_matching_resolution(self) -> None:
        qs = self._make_qs(question_id="q999")
        resolutions = {
            "q1": Resolution(id="q1", outcome=1),
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
        resolutions = {
            "q1": Resolution(id="q1", outcome=1, resolution_date="2024-07-01"),
        }
        result = join_resolved_questions([qs], resolutions)
        assert len(result) == 1


class TestExpandResolvedForHorizonsWithPerDateJoin:
    """After the join fix, expansion should create composite IDs with correct outcomes."""

    def test_per_date_entries_get_composite_ids(self) -> None:
        resolved = [
            ResolvedQuestion(
                id="q1", source="fred", question="Q",
                outcome=0, resolution_date="2024-07-01",
                resolution_dates=["2024-07-01", "2024-08-01"],
                forecast_due_date="2024-06-01",
            ),
            ResolvedQuestion(
                id="q1", source="fred", question="Q",
                outcome=1, resolution_date="2024-08-01",
                resolution_dates=["2024-07-01", "2024-08-01"],
                forecast_due_date="2024-06-01",
            ),
        ]
        expanded = _expand_resolved_for_horizons(resolved)
        ids = {rq.id for rq in expanded}
        assert "q1_2024-07-01" in ids
        assert "q1_2024-08-01" in ids
        assert "q1" not in ids

    def test_per_date_outcomes_preserved_after_expansion(self) -> None:
        resolved = [
            ResolvedQuestion(
                id="q1", source="fred", question="Q",
                outcome=0, resolution_date="2024-07-01",
                resolution_dates=["2024-07-01", "2024-08-01"],
                forecast_due_date="2024-06-01",
            ),
            ResolvedQuestion(
                id="q1", source="fred", question="Q",
                outcome=1, resolution_date="2024-08-01",
                resolution_dates=["2024-07-01", "2024-08-01"],
                forecast_due_date="2024-06-01",
            ),
        ]
        expanded = _expand_resolved_for_horizons(resolved)
        by_id = {rq.id: rq.outcome for rq in expanded}
        assert by_id["q1_2024-07-01"] == 0
        assert by_id["q1_2024-08-01"] == 1

    def test_market_questions_pass_through(self) -> None:
        resolved = [
            ResolvedQuestion(
                id="m1", source="metaculus", question="Market Q",
                outcome=1, forecast_due_date="2024-06-01",
            ),
        ]
        expanded = _expand_resolved_for_horizons(resolved)
        assert len(expanded) == 1
        assert expanded[0].id == "m1"

    def test_no_double_expansion(self) -> None:
        """Per-date entries should NOT re-expand from resolution_dates list."""
        resolved = [
            ResolvedQuestion(
                id="q1", source="fred", question="Q",
                outcome=0, resolution_date="2024-07-01",
                resolution_dates=["2024-07-01", "2024-08-01"],
                forecast_due_date="2024-06-01",
            ),
            ResolvedQuestion(
                id="q1", source="fred", question="Q",
                outcome=1, resolution_date="2024-08-01",
                resolution_dates=["2024-07-01", "2024-08-01"],
                forecast_due_date="2024-06-01",
            ),
        ]
        expanded = _expand_resolved_for_horizons(resolved)
        assert len(expanded) == 2

    def test_deduplication(self) -> None:
        """Duplicate entries for the same composite ID are deduplicated."""
        resolved = [
            ResolvedQuestion(
                id="q1", source="fred", question="Q",
                outcome=0, resolution_date="2024-07-01",
                resolution_dates=["2024-07-01"],
                forecast_due_date="2024-06-01",
            ),
            ResolvedQuestion(
                id="q1", source="fred", question="Q",
                outcome=0, resolution_date="2024-07-01",
                resolution_dates=["2024-07-01"],
                forecast_due_date="2024-06-01",
            ),
        ]
        expanded = _expand_resolved_for_horizons(resolved)
        assert len(expanded) == 1


class TestIntegrationMultiHorizonPipeline:
    """End-to-end: join + expand → verify composite outcomes."""

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
        resolutions = {
            "m1": Resolution(id="m1", outcome=1),
            "m2": Resolution(id="m2", outcome=0),
            "m3": Resolution(id="m3", outcome=1),
        }

        joined = join_resolved_questions([qs], resolutions)
        expanded = _expand_resolved_for_horizons(joined)

        ids = {rq.id for rq in expanded}
        assert "m1" in ids
        assert "m2" in ids
        assert "m3" in ids
        assert len(expanded) == 3
