"""Test fixtures for ForecastBench."""

from __future__ import annotations

import pytest

from fetch_data import QuestionSet, Question, ResolvedQuestion


@pytest.fixture(autouse=True)
def _disable_ensemble(monkeypatch: pytest.MonkeyPatch) -> None:
    """Default to single-call mode so existing tests aren't affected by ensemble."""
    monkeypatch.setattr("baseline_agent.ENSEMBLE_N", 1)


@pytest.fixture
def five_question_fixture() -> tuple[list[float], list[int], list[float], float, float]:
    """Hand-computed 5-question fixture.

    Returns (forecasts, outcomes, expected_brier_scores, expected_mean, expected_index).
    """
    forecasts = [0.9, 0.1, 0.5, 0.8, 0.3]
    outcomes = [1, 0, 1, 0, 1]
    expected_bs = [0.01, 0.01, 0.25, 0.64, 0.49]
    expected_mean = 0.28
    expected_index = (1.0 - 0.28**0.5) * 100.0
    return forecasts, outcomes, expected_bs, expected_mean, expected_index


@pytest.fixture
def five_resolved_questions() -> list[ResolvedQuestion]:
    """5 resolved questions for scoring tests, all dataset source."""
    return [
        ResolvedQuestion(
            id=f"q{i}",
            source="acled",
            question=f"Test question {i}",
            outcome=outcome,
            forecast_due_date="2024-01-01",
        )
        for i, outcome in enumerate([1, 0, 1, 0, 1])
    ]


@pytest.fixture
def mixed_resolved_questions() -> list[ResolvedQuestion]:
    """Resolved questions with both dataset and market sources."""
    dataset_qs = [
        ResolvedQuestion(id="d1", source="acled", question="Dataset Q1", outcome=1, forecast_due_date="2024-01-01"),
        ResolvedQuestion(id="d2", source="acled", question="Dataset Q2", outcome=0, forecast_due_date="2024-01-01"),
    ]
    market_qs = [
        ResolvedQuestion(id="m1", source="metaculus", question="Market Q1", outcome=1, forecast_due_date="2024-01-01"),
        ResolvedQuestion(id="m2", source="polymarket", question="Market Q2", outcome=0, forecast_due_date="2024-01-01"),
    ]
    return dataset_qs + market_qs


@pytest.fixture
def mock_question_sets() -> list[QuestionSet]:
    """5 question sets with sequential dates for held-out split testing."""
    dates = ["2024-01-01", "2024-02-01", "2024-03-01", "2024-04-01", "2024-05-01"]
    return [
        QuestionSet(
            forecast_due_date=date,
            question_set=f"set_{i}",
            questions=[
                Question(
                    id=f"qs{i}_q{j}",
                    source="acled",
                    question=f"Question {j} from set {i}",
                )
                for j in range(3)
            ],
        )
        for i, date in enumerate(dates)
    ]
