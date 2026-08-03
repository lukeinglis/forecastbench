"""Competition compliance tests for ForecastBench.

These tests verify invariants required by the ForecastBench competition rules.
All tests use synthetic data — NO network access, NO API calls.

Complements test_score.py (scoring math) and test_official_parity.py (prompt/parsing
parity) without duplicating them.
"""

from __future__ import annotations

import math

import pytest

from eval import _expand_resolved_for_horizons
from fetch_data import (
    Question,
    QuestionSet,
    Resolution,
    ResolvedQuestion,
    join_resolved_questions,
)
from score import brier_index, brier_score, score_forecasts


class TestBrierIndexAggregationOrder:
    """Brier Index must be (1 - sqrt(MEAN)) * 100, applied AFTER averaging.

    The wrong formula — MEAN((1 - sqrt(bs_i)) * 100) — gives a different answer.
    """

    def test_index_of_mean_differs_from_mean_of_index(self) -> None:
        scores = [0.01, 0.49]
        mean_bs = sum(scores) / len(scores)  # 0.25
        correct = (1.0 - math.sqrt(mean_bs)) * 100.0  # 50.0

        wrong = sum((1.0 - math.sqrt(s)) * 100.0 for s in scores) / len(scores)
        # (1 - sqrt(0.01))*100 = 90.0, (1 - sqrt(0.49))*100 = 30.0 → mean = 60.0

        assert correct != pytest.approx(wrong), "Test is invalid: formulas should differ"
        assert brier_index(mean_bs) == pytest.approx(correct)
        assert brier_index(mean_bs) == pytest.approx(50.0)

    def test_score_forecasts_uses_correct_aggregation(self) -> None:
        resolved = [
            ResolvedQuestion(id="q1", source="fred", question="Q1", outcome=1, forecast_due_date="2024-01-01"),
            ResolvedQuestion(id="q2", source="fred", question="Q2", outcome=0, forecast_due_date="2024-01-01"),
        ]
        forecasts = {"q1": 0.9, "q2": 0.3}
        result = score_forecasts(forecasts, resolved, difficulty_adjusted=False)

        bs1 = brier_score(0.9, 1)  # 0.01
        bs2 = brier_score(0.3, 0)  # 0.09
        expected_mean = (bs1 + bs2) / 2.0  # 0.05
        expected_index = (1.0 - math.sqrt(expected_mean)) * 100.0

        assert result.overall_brier == pytest.approx(expected_mean)
        assert result.overall_index == pytest.approx(expected_index)


class TestResolutionModelPreservation:
    """join_resolved_questions must produce one ResolvedQuestion per resolution entry."""

    def test_multiple_resolution_dates_produce_multiple_resolved(self) -> None:
        question = Question(
            id="ts1",
            source="fred",
            question="Will value exceed threshold?",
            resolution_dates=["2024-03-01", "2024-06-01", "2024-09-01"],
            forecast_due_date="2024-01-01",
        )
        qs = QuestionSet(
            forecast_due_date="2024-01-01",
            question_set="test_set",
            questions=[question],
        )

        resolutions: dict[str, Resolution] = {
            "ts1": Resolution(id="ts1", outcome=1, resolution_date="2024-03-01"),
        }
        resolved = join_resolved_questions([qs], resolutions)

        assert len(resolved) == 1
        assert resolved[0].id == "ts1"
        assert resolved[0].resolution_dates == ["2024-03-01", "2024-06-01", "2024-09-01"]

    def test_resolution_date_carried_to_resolved(self) -> None:
        question = Question(
            id="ts2",
            source="dbnomics",
            question="Test question",
            resolution_dates=["2024-03-01"],
            forecast_due_date="2024-01-01",
        )
        qs = QuestionSet(
            forecast_due_date="2024-01-01",
            questions=[question],
        )
        resolutions: dict[str, Resolution] = {
            "ts2": Resolution(id="ts2", outcome=0, resolution_date="2024-03-01"),
        }
        resolved = join_resolved_questions([qs], resolutions)

        assert len(resolved) == 1
        assert resolved[0].resolution_date == "2024-03-01"
        assert resolved[0].outcome == 0


class TestMissingForecastDefault:
    """Missing forecasts must default to 0.5 per ForecastBench rules."""

    def test_empty_forecasts_default_to_half(self) -> None:
        resolved = [
            ResolvedQuestion(id=f"q{i}", source="acled", question=f"Q{i}",
                             outcome=o, forecast_due_date="2024-01-01")
            for i, o in enumerate([1, 0, 1, 0])
        ]
        result = score_forecasts({}, resolved, difficulty_adjusted=False)

        assert result.n_missing == 4
        assert result.overall_brier == pytest.approx(0.25)

    def test_partial_forecasts_fill_missing_with_half(self) -> None:
        resolved = [
            ResolvedQuestion(id="q0", source="acled", question="Q0",
                             outcome=1, forecast_due_date="2024-01-01"),
            ResolvedQuestion(id="q1", source="acled", question="Q1",
                             outcome=0, forecast_due_date="2024-01-01"),
        ]
        result = score_forecasts({"q0": 0.9}, resolved, difficulty_adjusted=False)

        assert result.n_missing == 1
        bs_provided = brier_score(0.9, 1)  # 0.01
        bs_default = brier_score(0.5, 0)   # 0.25
        expected = (bs_provided + bs_default) / 2.0
        assert result.overall_brier == pytest.approx(expected)


class TestOutcomeValidation:
    """Only binary {0, 1} outcomes are valid."""

    def test_outcome_zero_valid(self) -> None:
        assert brier_score(0.3, 0) == pytest.approx(0.09)

    def test_outcome_one_valid(self) -> None:
        assert brier_score(0.7, 1) == pytest.approx(0.09)

    def test_outcome_two_rejected(self) -> None:
        with pytest.raises(ValueError, match="0 or 1"):
            brier_score(0.5, 2)

    def test_outcome_negative_rejected(self) -> None:
        with pytest.raises(ValueError, match="0 or 1"):
            brier_score(0.5, -1)

    def test_forecast_out_of_range_rejected(self) -> None:
        with pytest.raises(ValueError):
            brier_score(1.1, 1)
        with pytest.raises(ValueError):
            brier_score(-0.1, 0)


class TestCompositeIdCorrectness:
    """_expand_resolved_for_horizons must produce {id}_{date} composite IDs."""

    def test_dataset_question_expands_to_composite_ids(self) -> None:
        rq = ResolvedQuestion(
            id="fred_q1",
            source="fred",
            question="Will value exceed threshold?",
            resolution_dates=["2026-01-01", "2026-02-01", "2026-03-01"],
            outcome=1,
            forecast_due_date="2025-12-01",
        )
        expanded = _expand_resolved_for_horizons([rq])

        assert len(expanded) == 3
        ids = {e.id for e in expanded}
        assert ids == {"fred_q1_2026-01-01", "fred_q1_2026-02-01", "fred_q1_2026-03-01"}

    def test_composite_id_has_correct_resolution_date(self) -> None:
        rq = ResolvedQuestion(
            id="ts1",
            source="dbnomics",
            question="Test",
            resolution_dates=["2026-04-01", "2026-07-01"],
            outcome=0,
            forecast_due_date="2026-01-01",
        )
        expanded = _expand_resolved_for_horizons([rq])

        by_id = {e.id: e for e in expanded}
        assert by_id["ts1_2026-04-01"].resolution_date == "2026-04-01"
        assert by_id["ts1_2026-07-01"].resolution_date == "2026-07-01"

    def test_market_question_not_expanded(self) -> None:
        rq = ResolvedQuestion(
            id="mkt1",
            source="metaculus",
            question="Market question",
            outcome=1,
            forecast_due_date="2024-01-01",
        )
        expanded = _expand_resolved_for_horizons([rq])

        assert len(expanded) == 1
        assert expanded[0].id == "mkt1"

    def test_no_resolution_dates_not_expanded(self) -> None:
        rq = ResolvedQuestion(
            id="plain1",
            source="fred",
            question="Simple question",
            outcome=1,
            forecast_due_date="2024-01-01",
        )
        expanded = _expand_resolved_for_horizons([rq])

        assert len(expanded) == 1
        assert expanded[0].id == "plain1"

    def test_empty_resolution_dates_not_expanded(self) -> None:
        rq = ResolvedQuestion(
            id="empty1",
            source="fred",
            question="Empty dates",
            resolution_dates=[],
            outcome=0,
            forecast_due_date="2024-01-01",
        )
        expanded = _expand_resolved_for_horizons([rq])

        assert len(expanded) == 1
        assert expanded[0].id == "empty1"


class TestScoringMathInvariants:
    """Core scoring math that must hold for competition validity."""

    def test_dummy_forecaster_brier_025(self) -> None:
        assert brier_score(0.5, 0) == pytest.approx(0.25)
        assert brier_score(0.5, 1) == pytest.approx(0.25)

    def test_dummy_forecaster_index_50(self) -> None:
        assert brier_index(0.25) == pytest.approx(50.0)

    def test_perfect_forecaster_index_100(self) -> None:
        assert brier_index(0.0) == pytest.approx(100.0)

    def test_worst_forecaster_index_0(self) -> None:
        assert brier_index(1.0) == pytest.approx(0.0)

    def test_brier_score_symmetry(self) -> None:
        assert brier_score(0.7, 1) == pytest.approx(brier_score(0.3, 0))

    def test_overall_brier_is_weighted_mean(self) -> None:
        resolved = [
            ResolvedQuestion(id="d1", source="fred", question="D1",
                             outcome=1, forecast_due_date="2024-01-01"),
            ResolvedQuestion(id="d2", source="fred", question="D2",
                             outcome=0, forecast_due_date="2024-01-01"),
            ResolvedQuestion(id="m1", source="metaculus", question="M1",
                             outcome=1, forecast_due_date="2024-01-01"),
        ]
        forecasts = {"d1": 0.8, "d2": 0.2, "m1": 0.7}
        result = score_forecasts(forecasts, resolved, difficulty_adjusted=False)

        expected_overall = (
            result.dataset_brier * result.n_dataset
            + result.market_brier * result.n_market
        ) / (result.n_dataset + result.n_market)
        assert result.overall_brier == pytest.approx(expected_overall)
