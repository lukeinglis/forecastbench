"""Tests for Brier Skill Score and bootstrap confidence intervals."""

from __future__ import annotations

import pytest

from score import brier_skill_score, bootstrap_ci


class TestBrierSkillScore:
    def test_same_as_reference(self) -> None:
        assert brier_skill_score(0.25, 0.25) == pytest.approx(0.0)

    def test_better_than_reference(self) -> None:
        bss = brier_skill_score(0.10, 0.25)
        assert bss > 0
        assert bss == pytest.approx(0.6)

    def test_worse_than_reference(self) -> None:
        bss = brier_skill_score(0.40, 0.25)
        assert bss < 0
        assert bss == pytest.approx(-0.6)

    def test_perfect_forecaster(self) -> None:
        assert brier_skill_score(0.0, 0.25) == pytest.approx(1.0)

    def test_reference_zero_returns_zero(self) -> None:
        assert brier_skill_score(0.25, 0.0) == 0.0

    def test_default_reference(self) -> None:
        bss = brier_skill_score(0.25)
        assert bss == pytest.approx(0.0)


class TestBootstrapCI:
    def test_returns_valid_interval(self) -> None:
        pairs = [(0.7, 1), (0.3, 0), (0.5, 1), (0.6, 0)]
        lo, hi = bootstrap_ci(pairs)
        assert lo <= hi

    def test_seed_deterministic(self) -> None:
        pairs = [(0.8, 1), (0.2, 0), (0.5, 1)]
        r1 = bootstrap_ci(pairs, seed=42)
        r2 = bootstrap_ci(pairs, seed=42)
        assert r1 == r2

    def test_different_seed_different_result(self) -> None:
        pairs = [(0.8, 1), (0.2, 0), (0.5, 1), (0.6, 0), (0.4, 1)]
        r1 = bootstrap_ci(pairs, seed=42)
        r2 = bootstrap_ci(pairs, seed=99)
        assert r1 != r2

    def test_lower_le_upper(self) -> None:
        pairs = [(0.9, 1), (0.1, 0), (0.5, 1), (0.8, 0), (0.3, 1)]
        lo, hi = bootstrap_ci(pairs)
        assert lo <= hi

    def test_perfect_forecaster(self) -> None:
        pairs = [(1.0, 1), (0.0, 0), (1.0, 1), (0.0, 0)]
        lo, hi = bootstrap_ci(pairs)
        assert lo == pytest.approx(0.0)
        assert hi == pytest.approx(0.0)

    def test_empty_pairs(self) -> None:
        lo, hi = bootstrap_ci([])
        assert lo == 0.0
        assert hi == 0.0

    def test_single_pair(self) -> None:
        pairs = [(0.5, 1)]
        lo, hi = bootstrap_ci(pairs)
        assert lo == pytest.approx(0.25)
        assert hi == pytest.approx(0.25)

    def test_ci_narrows_with_more_replicates(self) -> None:
        pairs = [(0.7, 1), (0.3, 0), (0.5, 1), (0.6, 0), (0.4, 1)] * 10
        lo1, hi1 = bootstrap_ci(pairs, n_replicates=100, seed=42)
        lo2, hi2 = bootstrap_ci(pairs, n_replicates=10000, seed=42)
        width1 = hi1 - lo1
        width2 = hi2 - lo2
        assert width2 <= width1 + 0.01
