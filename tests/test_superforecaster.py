"""Tests for superforecaster median computation and comparison."""

from __future__ import annotations

import pytest

from fetch_data import ResolvedQuestion, superforecaster_medians
from analyze import compare_to_superforecasters


class TestSuperforecasterMedians:
    def test_single_forecaster_per_question(self) -> None:
        forecasts = [
            {"id": "q1", "forecast": 0.7},
            {"id": "q2", "forecast": 0.3},
        ]
        result = superforecaster_medians(forecasts)
        assert result == {"q1": 0.7, "q2": 0.3}

    def test_multiple_forecasters_same_question(self) -> None:
        forecasts = [
            {"id": "q1", "forecast": 0.2},
            {"id": "q1", "forecast": 0.8},
            {"id": "q1", "forecast": 0.5},
        ]
        result = superforecaster_medians(forecasts)
        assert result["q1"] == 0.5

    def test_even_number_of_forecasters(self) -> None:
        forecasts = [
            {"id": "q1", "forecast": 0.2},
            {"id": "q1", "forecast": 0.6},
        ]
        result = superforecaster_medians(forecasts)
        assert result["q1"] == pytest.approx(0.4)

    def test_empty_input(self) -> None:
        result = superforecaster_medians([])
        assert result == {}

    def test_none_forecast_skipped(self) -> None:
        forecasts = [
            {"id": "q1", "forecast": None},
            {"id": "q1", "forecast": 0.6},
        ]
        result = superforecaster_medians(forecasts)
        assert result["q1"] == 0.6

    def test_missing_forecast_key_skipped(self) -> None:
        forecasts = [
            {"id": "q1"},
            {"id": "q1", "forecast": 0.4},
        ]
        result = superforecaster_medians(forecasts)
        assert result["q1"] == 0.4

    def test_all_none_forecasts_excluded(self) -> None:
        forecasts = [
            {"id": "q1", "forecast": None},
            {"id": "q1", "forecast": None},
        ]
        result = superforecaster_medians(forecasts)
        assert "q1" not in result


class TestCompareToSuperforecasters:
    @pytest.fixture
    def resolved(self) -> list[ResolvedQuestion]:
        return [
            ResolvedQuestion(id="q1", source="acled", question="Q1", outcome=1, forecast_due_date="2024-07-21"),
            ResolvedQuestion(id="q2", source="acled", question="Q2", outcome=0, forecast_due_date="2024-07-21"),
            ResolvedQuestion(id="q3", source="acled", question="Q3", outcome=1, forecast_due_date="2024-07-21"),
        ]

    def test_correct_win_counts(self, resolved: list[ResolvedQuestion]) -> None:
        our_forecasts = {"q1": 0.9, "q2": 0.1, "q3": 0.5}
        sf_medians = {"q1": 0.5, "q2": 0.5, "q3": 0.9}
        result = compare_to_superforecasters(our_forecasts, resolved, sf_medians)
        assert result["n_shared"] == 3
        assert result["n_we_won"] == 2  # q1: 0.01 < 0.25, q2: 0.01 < 0.25
        assert result["n_they_won"] == 1  # q3: 0.25 > 0.01

    def test_no_shared_questions(self, resolved: list[ResolvedQuestion]) -> None:
        our_forecasts = {"qX": 0.5}
        sf_medians = {"qY": 0.5}
        result = compare_to_superforecasters(our_forecasts, resolved, sf_medians)
        assert result["n_shared"] == 0

    def test_mean_brier_calculation(self, resolved: list[ResolvedQuestion]) -> None:
        our_forecasts = {"q1": 1.0, "q2": 0.0}
        sf_medians = {"q1": 0.5, "q2": 0.5}
        result = compare_to_superforecasters(our_forecasts, resolved, sf_medians)
        assert result["n_shared"] == 2
        assert result["our_mean_brier"] == pytest.approx(0.0)  # perfect
        assert result["sf_mean_brier"] == pytest.approx(0.25)  # always 0.5

    def test_tie_counted_as_neither(self, resolved: list[ResolvedQuestion]) -> None:
        our_forecasts = {"q1": 0.5}
        sf_medians = {"q1": 0.5}
        result = compare_to_superforecasters(our_forecasts, resolved, sf_medians)
        assert result["n_we_won"] == 0
        assert result["n_they_won"] == 0
