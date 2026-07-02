"""Tests for score.py."""

from __future__ import annotations

import math

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from fetch_data import ResolvedQuestion
from score import brier_score, mean_brier_score, brier_index, score_forecasts, ScoringResult


class TestBrierScore:
    def test_perfect_positive(self) -> None:
        assert brier_score(1.0, 1) == 0.0

    def test_perfect_negative(self) -> None:
        assert brier_score(0.0, 0) == 0.0

    def test_worst_positive(self) -> None:
        assert brier_score(0.0, 1) == 1.0

    def test_worst_negative(self) -> None:
        assert brier_score(1.0, 0) == 1.0

    def test_half(self) -> None:
        assert brier_score(0.5, 1) == 0.25
        assert brier_score(0.5, 0) == 0.25

    def test_rejects_nan(self) -> None:
        with pytest.raises(ValueError, match="finite"):
            brier_score(float("nan"), 1)

    def test_rejects_inf(self) -> None:
        with pytest.raises(ValueError, match="finite"):
            brier_score(float("inf"), 1)

    def test_rejects_out_of_range(self) -> None:
        with pytest.raises(ValueError, match=r"\[0, 1\]"):
            brier_score(1.5, 1)
        with pytest.raises(ValueError, match=r"\[0, 1\]"):
            brier_score(-0.1, 0)

    def test_rejects_invalid_outcome(self) -> None:
        with pytest.raises(ValueError, match="0 or 1"):
            brier_score(0.5, 2)


class TestHandComputedFixture:
    def test_individual_scores(self, five_question_fixture: tuple) -> None:
        forecasts, outcomes, expected_bs, _, _ = five_question_fixture
        for f, o, expected in zip(forecasts, outcomes, expected_bs):
            actual = brier_score(f, o)
            assert abs(actual - expected) < 1e-10, f"brier_score({f}, {o}) = {actual}, expected {expected}"

    def test_mean_brier_score(self, five_question_fixture: tuple) -> None:
        forecasts, outcomes, _, expected_mean, _ = five_question_fixture
        pairs = list(zip(forecasts, outcomes))
        actual = mean_brier_score(pairs)
        assert abs(actual - expected_mean) < 1e-10, f"mean={actual}, expected={expected_mean}"

    def test_brier_index(self, five_question_fixture: tuple) -> None:
        _, _, _, expected_mean, expected_index = five_question_fixture
        actual = brier_index(expected_mean)
        assert abs(actual - expected_index) < 1e-6, f"index={actual}, expected={expected_index}"


class TestMeanBrierScore:
    def test_empty_raises(self) -> None:
        with pytest.raises(ValueError, match="empty"):
            mean_brier_score([])

    def test_single(self) -> None:
        assert mean_brier_score([(0.5, 1)]) == 0.25


class TestBrierIndex:
    def test_perfect(self) -> None:
        assert brier_index(0.0) == 100.0

    def test_worst(self) -> None:
        assert brier_index(1.0) == 0.0

    def test_half(self) -> None:
        expected = (1.0 - math.sqrt(0.25)) * 100.0
        assert abs(brier_index(0.25) - expected) < 1e-10


class TestScoreForecasts:
    def test_empty_raises(self) -> None:
        with pytest.raises(ValueError, match="No resolved"):
            score_forecasts({}, [])

    def test_dataset_and_market_separate(self, mixed_resolved_questions: list) -> None:
        forecasts = {"d1": 0.9, "d2": 0.1, "m1": 0.9, "m2": 0.1}
        result = score_forecasts(forecasts, mixed_resolved_questions)
        assert result.n_dataset == 2
        assert result.n_market == 2
        assert result.dataset_brier == result.market_brier
        assert result.n_missing == 0

    def test_missing_defaults_to_half(self, five_resolved_questions: list) -> None:
        result = score_forecasts({}, five_resolved_questions)
        assert result.n_missing == 5
        assert abs(result.dataset_brier - 0.25) < 1e-10

    def test_overall_is_average_of_components(self, mixed_resolved_questions: list) -> None:
        forecasts = {"d1": 0.8, "d2": 0.2, "m1": 0.6, "m2": 0.4}
        result = score_forecasts(forecasts, mixed_resolved_questions)
        expected_overall = (result.dataset_brier + result.market_brier) / 2.0
        assert abs(result.overall_brier - expected_overall) < 1e-10


class TestPropertyBased:
    @given(
        forecast=st.floats(min_value=0.0, max_value=1.0),
        outcome=st.sampled_from([0, 1]),
    )
    def test_brier_in_range(self, forecast: float, outcome: int) -> None:
        bs = brier_score(forecast, outcome)
        assert 0.0 <= bs <= 1.0

    @given(
        forecasts=st.lists(
            st.tuples(
                st.floats(min_value=0.0, max_value=1.0),
                st.sampled_from([0, 1]),
            ),
            min_size=1,
            max_size=20,
        ),
    )
    def test_mean_brier_in_range(self, forecasts: list[tuple[float, int]]) -> None:
        mbs = mean_brier_score(forecasts)
        assert 0.0 <= mbs <= 1.0

    @given(mean_bs=st.floats(min_value=0.0, max_value=1.0))
    def test_brier_index_in_range(self, mean_bs: float) -> None:
        bi = brier_index(mean_bs)
        assert 0.0 <= bi <= 100.0

    def test_perfect_forecasts(self) -> None:
        pairs = [(1.0, 1), (0.0, 0), (1.0, 1), (0.0, 0)]
        mbs = mean_brier_score(pairs)
        assert mbs == 0.0
        assert brier_index(mbs) == 100.0

    def test_worst_forecasts(self) -> None:
        pairs = [(0.0, 1), (1.0, 0), (0.0, 1), (1.0, 0)]
        mbs = mean_brier_score(pairs)
        assert mbs == 1.0
        assert brier_index(mbs) == 0.0
