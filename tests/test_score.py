"""Tests for score.py."""

from __future__ import annotations

import math

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from fetch_data import ResolvedQuestion
from score import (
    _estimate_difficulty_effects_ols,
    adjust_for_difficulty,
    brier_index,
    brier_score,
    mean_brier_score,
    score_forecasts,
)


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


def _make_resolved(
    qid: str, source: str, outcome: int, due: str = "2024-01-01",
) -> ResolvedQuestion:
    return ResolvedQuestion(
        id=qid, source=source, question=f"Q {qid}",
        outcome=outcome, forecast_due_date=due,
    )


class TestDifficultyEffectsOLS:
    def test_symmetric_forecasters(self) -> None:
        """Two forecasters, two questions. Effects should be mean-centered."""
        qs = [
            _make_resolved("q1", "acled", 1),
            _make_resolved("q2", "acled", 0),
        ]
        forecasts = {
            "A": {"q1": 0.9, "q2": 0.1},
            "B": {"q1": 0.7, "q2": 0.3},
        }
        outcomes = {q.id: q.outcome for q in qs}
        effects = _estimate_difficulty_effects_ols(
            forecasts, outcomes, ["q1", "q2"],
        )
        assert abs(effects["q1"] + effects["q2"]) < 1e-10

    def test_single_forecaster_effects_zero(self) -> None:
        """With one forecaster, all effects equal mean minus grand_mean = 0."""
        qs = [
            _make_resolved("q1", "acled", 1),
            _make_resolved("q2", "acled", 0),
        ]
        forecasts = {"A": {"q1": 0.8, "q2": 0.2}}
        outcomes = {q.id: q.outcome for q in qs}
        effects = _estimate_difficulty_effects_ols(
            forecasts, outcomes, ["q1", "q2"],
        )
        assert abs(effects["q1"]) < 1e-10
        assert abs(effects["q2"]) < 1e-10


class TestAdjustForDifficulty:
    def test_constant_half_yields_025(self) -> None:
        """A forecaster always predicting 0.5 should get adjusted mean = 0.25."""
        qs = [
            _make_resolved("q1", "acled", 1),
            _make_resolved("q2", "acled", 0),
            _make_resolved("q3", "acled", 1),
        ]
        forecasts = {
            "half": {"q1": 0.5, "q2": 0.5, "q3": 0.5},
            "good": {"q1": 0.9, "q2": 0.1, "q3": 0.8},
            "bad": {"q1": 0.2, "q2": 0.8, "q3": 0.3},
        }
        adjusted = adjust_for_difficulty(forecasts, qs)
        half_scores = adjusted["half"]
        mean_adj = sum(half_scores.values()) / len(half_scores)
        assert abs(mean_adj - 0.25) < 1e-10

    def test_good_forecaster_below_half(self) -> None:
        """A good forecaster should score below 0.25 (below the constant-0.5 baseline)."""
        qs = [
            _make_resolved("q1", "acled", 1),
            _make_resolved("q2", "acled", 0),
            _make_resolved("q3", "acled", 1),
            _make_resolved("q4", "acled", 0),
        ]
        forecasts = {
            "good": {"q1": 0.95, "q2": 0.05, "q3": 0.9, "q4": 0.1},
            "half": {"q1": 0.5, "q2": 0.5, "q3": 0.5, "q4": 0.5},
            "bad": {"q1": 0.1, "q2": 0.9, "q3": 0.2, "q4": 0.8},
        }
        adjusted = adjust_for_difficulty(forecasts, qs)
        good_mean = sum(adjusted["good"].values()) / len(adjusted["good"])
        half_mean = sum(adjusted["half"].values()) / len(adjusted["half"])
        assert good_mean < half_mean

    def test_hand_computed_adjustment(self) -> None:
        """Verify exact difficulty adjustment on a 2-forecaster, 2-question example.

        q1: outcome=1, A predicts 0.9 (bs=0.01), B predicts 0.7 (bs=0.09)
        q2: outcome=0, A predicts 0.1 (bs=0.01), B predicts 0.3 (bs=0.09)

        Question means: q1_mean=0.05, q2_mean=0.05
        Grand mean: 0.05
        Effects: q1=0.0, q2=0.0

        Unscaled: same as raw Brier since effects are 0.
        Constant-0.5: q1 bs=0.25, q2 bs=0.25 → mean_half_unscaled=0.25
        Shift = 0.25 - 0.25 = 0.0
        Adjusted = raw Brier scores (since effects are symmetric and cancel out).
        """
        qs = [
            _make_resolved("q1", "acled", 1),
            _make_resolved("q2", "acled", 0),
        ]
        forecasts = {
            "A": {"q1": 0.9, "q2": 0.1},
            "B": {"q1": 0.7, "q2": 0.3},
        }
        adjusted = adjust_for_difficulty(forecasts, qs)
        assert abs(adjusted["A"]["q1"] - 0.01) < 1e-10
        assert abs(adjusted["A"]["q2"] - 0.01) < 1e-10
        assert abs(adjusted["B"]["q1"] - 0.09) < 1e-10
        assert abs(adjusted["B"]["q2"] - 0.09) < 1e-10

    def test_asymmetric_difficulty(self) -> None:
        """When questions have different difficulty, adjustment should compensate.

        q1 (hard): outcome=1. A predicts 0.6 (bs=0.16), B predicts 0.5 (bs=0.25)
        q2 (easy): outcome=1. A predicts 0.9 (bs=0.01), B predicts 0.95 (bs=0.0025)

        Question means: q1=0.205, q2=0.00625
        Grand mean: 0.105625
        Effects: q1=0.099375, q2=-0.099375

        Unscaled A: q1=0.16-0.099375=0.060625, q2=0.01-(-0.099375)=0.109375
        Unscaled B: q1=0.25-0.099375=0.150625, q2=0.0025-(-0.099375)=0.101875

        Constant-0.5 on q1: bs=0.25, unscaled=0.25-0.099375=0.150625
        Constant-0.5 on q2: bs=0.25, unscaled=0.25-(-0.099375)=0.349375
        mean_half_unscaled = (0.150625+0.349375)/2 = 0.25
        Shift = 0.25 - 0.25 = 0.0
        """
        qs = [
            _make_resolved("q1", "acled", 1),
            _make_resolved("q2", "acled", 1),
        ]
        forecasts = {
            "A": {"q1": 0.6, "q2": 0.9},
            "B": {"q1": 0.5, "q2": 0.95},
        }
        adjusted = adjust_for_difficulty(forecasts, qs)
        assert abs(adjusted["A"]["q1"] - 0.060625) < 1e-10
        assert abs(adjusted["A"]["q2"] - 0.109375) < 1e-10

    def test_empty_returns_empty(self) -> None:
        assert adjust_for_difficulty({}, []) == {}


class TestScoreForecastsDifficultyAdjusted:
    def test_no_peer_pool_falls_back_to_raw(self) -> None:
        """Without all_forecasts, score_forecasts should produce raw scores."""
        qs = [_make_resolved(f"q{i}", "acled", o) for i, o in enumerate([1, 0, 1])]
        forecasts = {"q0": 0.5, "q1": 0.5, "q2": 0.5}
        result = score_forecasts(forecasts, qs, difficulty_adjusted=True)
        assert not result.difficulty_adjusted
        assert abs(result.dataset_brier - 0.25) < 1e-10

    def test_single_forecaster_degenerates_to_raw(self) -> None:
        """With only one forecaster in the pool, adjustment is skipped."""
        qs = [_make_resolved(f"q{i}", "acled", o) for i, o in enumerate([1, 0])]
        forecasts = {"q0": 0.8, "q1": 0.2}
        result = score_forecasts(
            forecasts, qs, difficulty_adjusted=True,
            all_forecasts={"only_one": {"q0": 0.5, "q1": 0.5}},
        )
        assert not result.difficulty_adjusted

    def test_adjusted_with_peer_pool(self) -> None:
        """With a peer pool, difficulty adjustment should be applied."""
        qs = [
            _make_resolved("q1", "acled", 1),
            _make_resolved("q2", "acled", 0),
        ]
        peer_pool = {
            "peer1": {"q1": 0.7, "q2": 0.3},
            "peer2": {"q1": 0.6, "q2": 0.4},
        }
        result = score_forecasts(
            {"q1": 0.9, "q2": 0.1}, qs,
            difficulty_adjusted=True, all_forecasts=peer_pool,
        )
        assert result.difficulty_adjusted
        assert result.n_dataset == 2
        assert result.dataset_brier >= 0.0

    def test_adjusted_constant_half_yields_025_mean(self) -> None:
        """Constant-0.5 forecaster should yield 0.25 adjusted mean Brier."""
        qs = [
            _make_resolved("q1", "acled", 1),
            _make_resolved("q2", "acled", 0),
            _make_resolved("q3", "acled", 1),
        ]
        peer_pool = {
            "peer1": {"q1": 0.9, "q2": 0.1, "q3": 0.8},
            "peer2": {"q1": 0.3, "q2": 0.7, "q3": 0.4},
        }
        result = score_forecasts(
            {"q1": 0.5, "q2": 0.5, "q3": 0.5}, qs,
            difficulty_adjusted=True, all_forecasts=peer_pool,
        )
        assert result.difficulty_adjusted
        assert abs(result.dataset_brier - 0.25) < 1e-10
        assert abs(result.dataset_index - brier_index(0.25)) < 1e-6

    def test_difficulty_adjusted_false_gives_raw(self) -> None:
        """When difficulty_adjusted=False, raw scores used even with peer pool."""
        qs = [_make_resolved("q1", "acled", 1)]
        peer_pool = {"p1": {"q1": 0.9}, "p2": {"q1": 0.1}}
        result = score_forecasts(
            {"q1": 0.8}, qs,
            difficulty_adjusted=False, all_forecasts=peer_pool,
        )
        assert not result.difficulty_adjusted
        assert abs(result.dataset_brier - 0.04) < 1e-10

    def test_market_and_dataset_separation(self) -> None:
        """Market and dataset questions should be scored separately."""
        qs = [
            _make_resolved("d1", "acled", 1),
            _make_resolved("m1", "metaculus", 1),
        ]
        peer_pool = {
            "p1": {"d1": 0.7, "m1": 0.8},
            "p2": {"d1": 0.6, "m1": 0.7},
        }
        result = score_forecasts(
            {"d1": 0.9, "m1": 0.9}, qs,
            difficulty_adjusted=True, all_forecasts=peer_pool,
        )
        assert result.n_dataset == 1
        assert result.n_market == 1
        assert result.difficulty_adjusted

    def test_brier_index_applied_after_averaging(self) -> None:
        """Verify Brier Index is (1-sqrt(mean)) * 100, applied AFTER averaging."""
        qs = [
            _make_resolved("q1", "acled", 1),
            _make_resolved("q2", "acled", 0),
        ]
        peer_pool = {
            "p1": {"q1": 0.7, "q2": 0.3},
            "p2": {"q1": 0.6, "q2": 0.4},
        }
        result = score_forecasts(
            {"q1": 0.9, "q2": 0.1}, qs,
            difficulty_adjusted=True, all_forecasts=peer_pool,
        )
        expected_index = (1.0 - math.sqrt(result.dataset_brier)) * 100.0
        assert abs(result.dataset_index - expected_index) < 1e-6


class TestDifficultyAdjustedPropertyBased:
    @given(
        forecasts=st.lists(
            st.tuples(
                st.floats(min_value=0.0, max_value=1.0),
                st.sampled_from([0, 1]),
            ),
            min_size=2,
            max_size=10,
        ),
    )
    @settings(max_examples=50)
    def test_adjusted_scores_in_range(
        self, forecasts: list[tuple[float, int]],
    ) -> None:
        """Adjusted Brier scores should remain in [0, 1]."""
        qs = [
            _make_resolved(f"q{i}", "acled", o)
            for i, (_, o) in enumerate(forecasts)
        ]
        fcast_map_a = {f"q{i}": f for i, (f, _) in enumerate(forecasts)}
        fcast_map_b = {f"q{i}": 0.5 for i in range(len(forecasts))}
        all_f = {"A": fcast_map_a, "B": fcast_map_b}
        adjusted = adjust_for_difficulty(all_f, qs)
        for fid, scores in adjusted.items():
            for qid, s in scores.items():
                assert 0.0 <= s <= 1.0, f"{fid}/{qid}: {s} out of [0,1]"
