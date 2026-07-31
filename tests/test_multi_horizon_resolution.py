"""Tests for multi-horizon resolution matching (issue #94).

Verifies that dataset questions with multiple resolution dates get correct
per-date outcomes, and that market questions remain unaffected.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from fetch_data import (
    Question,
    QuestionSet,
    Resolution,
    ResolvedQuestion,
    fetch_all_resolutions,
    join_resolved_questions,
)
from eval import _expand_resolved_for_horizons


class TestFetchAllResolutionsPreservesMultiple:
    """fetch_all_resolutions must keep all resolution entries per question ID."""

    @patch("fetch_data.list_resolution_files")
    @patch("fetch_data.fetch_resolution")
    def test_multiple_resolutions_per_id_preserved(
        self, mock_fetch_res: MagicMock, mock_list: MagicMock
    ) -> None:
        mock_list.return_value = ["res1.json"]
        mock_fetch_res.return_value = [
            Resolution(id="q1", outcome=0, resolution_date="2024-01-01"),
            Resolution(id="q1", outcome=1, resolution_date="2024-06-01"),
            Resolution(id="q1", outcome=0, resolution_date="2024-12-01"),
        ]
        result = fetch_all_resolutions()
        assert "q1" in result
        assert len(result["q1"]) == 3
        outcomes = [r.outcome for r in result["q1"]]
        assert outcomes == [0, 1, 0]

    @patch("fetch_data.list_resolution_files")
    @patch("fetch_data.fetch_resolution")
    def test_single_resolution_still_works(
        self, mock_fetch_res: MagicMock, mock_list: MagicMock
    ) -> None:
        mock_list.return_value = ["res1.json"]
        mock_fetch_res.return_value = [
            Resolution(id="m1", outcome=1, resolution_date="2024-03-01"),
        ]
        result = fetch_all_resolutions()
        assert "m1" in result
        assert len(result["m1"]) == 1
        assert result["m1"][0].outcome == 1

    @patch("fetch_data.list_resolution_files")
    @patch("fetch_data.fetch_resolution")
    def test_resolutions_across_files_accumulated(
        self, mock_fetch_res: MagicMock, mock_list: MagicMock
    ) -> None:
        mock_list.return_value = ["res1.json", "res2.json"]
        mock_fetch_res.side_effect = [
            [Resolution(id="q1", outcome=0, resolution_date="2024-01-01")],
            [Resolution(id="q1", outcome=1, resolution_date="2024-06-01")],
        ]
        result = fetch_all_resolutions()
        assert len(result["q1"]) == 2


class TestJoinResolvedQuestionsMultiHorizon:
    """join_resolved_questions must produce one ResolvedQuestion per resolution entry."""

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

    def test_produces_n_entries_for_n_resolutions(self) -> None:
        qs = self._make_qs()
        resolutions: dict[str, list[Resolution]] = {
            "q1": [
                Resolution(id="q1", outcome=0, resolution_date="2024-07-01"),
                Resolution(id="q1", outcome=1, resolution_date="2024-08-01"),
                Resolution(id="q1", outcome=0, resolution_date="2024-09-01"),
            ],
        }
        result = join_resolved_questions([qs], resolutions)
        assert len(result) == 3

    def test_each_entry_has_correct_per_date_outcome(self) -> None:
        qs = self._make_qs()
        resolutions: dict[str, list[Resolution]] = {
            "q1": [
                Resolution(id="q1", outcome=0, resolution_date="2024-07-01"),
                Resolution(id="q1", outcome=1, resolution_date="2024-08-01"),
                Resolution(id="q1", outcome=0, resolution_date="2024-09-01"),
            ],
        }
        result = join_resolved_questions([qs], resolutions)
        by_date = {rq.resolution_date: rq.outcome for rq in result}
        assert by_date["2024-07-01"] == 0
        assert by_date["2024-08-01"] == 1
        assert by_date["2024-09-01"] == 0

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

    def test_resolved_false_excluded(self) -> None:
        qs = self._make_qs()
        resolutions: dict[str, list[Resolution]] = {
            "q1": [
                Resolution(id="q1", outcome=1, resolution_date="2024-07-01", resolved=True),
                Resolution(id="q1", outcome=0, resolution_date="2024-08-01", resolved=False),
            ],
        }
        result = join_resolved_questions([qs], resolutions)
        assert len(result) == 1
        assert result[0].resolution_date == "2024-07-01"

    def test_outcome_none_excluded(self) -> None:
        qs = self._make_qs()
        resolutions: dict[str, list[Resolution]] = {
            "q1": [
                Resolution(id="q1", outcome=1, resolution_date="2024-07-01"),
                Resolution(id="q1", outcome=None, resolution_date="2024-08-01"),
            ],
        }
        result = join_resolved_questions([qs], resolutions)
        assert len(result) == 1

    def test_no_matching_resolution(self) -> None:
        qs = self._make_qs(question_id="q999")
        resolutions: dict[str, list[Resolution]] = {
            "q1": [Resolution(id="q1", outcome=1)],
        }
        result = join_resolved_questions([qs], resolutions)
        assert len(result) == 0


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
    """End-to-end: mock resolution data → join → expand → verify composite outcomes."""

    def test_full_pipeline_different_outcomes_per_date(self) -> None:
        qs = QuestionSet(
            forecast_due_date="2024-06-01",
            question_set="round_1",
            questions=[
                Question(
                    id="dq1", source="fred", question="Dataset Q?",
                    resolution_dates=["2024-07-01", "2024-08-01", "2024-09-01"],
                ),
                Question(
                    id="mq1", source="polymarket", question="Market Q?",
                ),
            ],
        )
        resolutions: dict[str, list[Resolution]] = {
            "dq1": [
                Resolution(id="dq1", outcome=0, resolution_date="2024-07-01"),
                Resolution(id="dq1", outcome=1, resolution_date="2024-08-01"),
                Resolution(id="dq1", outcome=1, resolution_date="2024-09-01"),
            ],
            "mq1": [
                Resolution(id="mq1", outcome=0, resolution_date="2024-06-15"),
            ],
        }

        joined = join_resolved_questions([qs], resolutions)
        assert len(joined) == 4

        expanded = _expand_resolved_for_horizons(joined)

        by_id = {rq.id: rq.outcome for rq in expanded}
        assert by_id["dq1_2024-07-01"] == 0
        assert by_id["dq1_2024-08-01"] == 1
        assert by_id["dq1_2024-09-01"] == 1
        assert by_id["mq1"] == 0
        assert len(expanded) == 4

    def test_mixed_market_and_dataset_sources(self) -> None:
        qs = QuestionSet(
            forecast_due_date="2024-06-01",
            question_set="round_1",
            questions=[
                Question(id="d1", source="acled", question="D1?",
                         resolution_dates=["2024-07-01", "2024-12-01"]),
                Question(id="m1", source="metaculus", question="M1?"),
                Question(id="m2", source="manifold", question="M2?"),
                Question(id="m3", source="infer", question="M3?"),
            ],
        )
        resolutions: dict[str, list[Resolution]] = {
            "d1": [
                Resolution(id="d1", outcome=1, resolution_date="2024-07-01"),
                Resolution(id="d1", outcome=0, resolution_date="2024-12-01"),
            ],
            "m1": [Resolution(id="m1", outcome=1)],
            "m2": [Resolution(id="m2", outcome=0)],
            "m3": [Resolution(id="m3", outcome=1)],
        }

        joined = join_resolved_questions([qs], resolutions)
        expanded = _expand_resolved_for_horizons(joined)

        ids = {rq.id for rq in expanded}
        assert "d1_2024-07-01" in ids
        assert "d1_2024-12-01" in ids
        assert "m1" in ids
        assert "m2" in ids
        assert "m3" in ids
        assert len(expanded) == 5
