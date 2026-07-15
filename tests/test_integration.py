"""Integration tests that fetch real data from the ForecastBench repo.

Skipped by default — run with: uv run pytest -m integration
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from fetch_data import (
    QuestionSet,
    ResolvedQuestion,
    fetch_all_resolutions,
    fetch_question_set,
    join_resolved_questions,
)
from eval import run_eval
from score import score_forecasts

pytestmark = pytest.mark.integration

REAL_ROUND = "2026-07-05-llm"


def _dummy_forecaster(question: object, resolution_date: str | None = None) -> float:
    return 0.5


class TestFetchRealQuestionSet:
    def test_returns_question_set_with_questions(self) -> None:
        qs = fetch_question_set(f"{REAL_ROUND}.json")
        assert isinstance(qs, QuestionSet)
        assert len(qs.questions) > 0

    def test_questions_have_required_fields(self) -> None:
        qs = fetch_question_set(f"{REAL_ROUND}.json")
        for q in qs.questions:
            assert q.id, "question id must be non-empty"
            assert q.source, "question source must be non-empty"
            assert q.question, "question text must be non-empty"


class TestFetchRealResolutions:
    def test_resolutions_not_empty(self) -> None:
        resolutions = fetch_all_resolutions()
        assert len(resolutions) > 0

    def test_some_have_outcomes(self) -> None:
        resolutions = fetch_all_resolutions()
        with_outcome = [r for r in resolutions.values() if r.outcome is not None]
        assert len(with_outcome) > 0


class TestJoinRealData:
    def test_join_produces_resolved_questions(self) -> None:
        qs = fetch_question_set(f"{REAL_ROUND}.json")
        resolutions = fetch_all_resolutions()
        resolved = join_resolved_questions([qs], resolutions)
        assert len(resolved) > 0
        for rq in resolved:
            assert isinstance(rq, ResolvedQuestion)
            assert rq.outcome in (0, 1)


class TestScoreRealData:
    def test_constant_half_gives_brier_025(self) -> None:
        qs = fetch_question_set(f"{REAL_ROUND}.json")
        resolutions = fetch_all_resolutions()
        resolved = join_resolved_questions([qs], resolutions)
        if not resolved:
            pytest.skip("No resolved questions available for this round")
        forecasts = {rq.id: 0.5 for rq in resolved}
        result = score_forecasts(forecasts, resolved, difficulty_adjusted=False)
        assert abs(result.overall_brier - 0.25) < 1e-9


class TestFullEvalPipeline:
    def test_dummy_eval_on_real_round(self, tmp_path: Path) -> None:
        import eval as eval_mod

        original_results = eval_mod.RESULTS_DIR
        original_cache = eval_mod.CACHE_DIR
        eval_mod.RESULTS_DIR = tmp_path / "results"
        eval_mod.CACHE_DIR = tmp_path / "cache"
        try:
            result = asyncio.run(
                run_eval(_dummy_forecaster, round_name=REAL_ROUND)
            )
            assert result.scoring.n_dataset + result.scoring.n_market > 0
            assert result.scoring.overall_brier >= 0.0
            assert len(result.forecasts) > 0
        finally:
            eval_mod.RESULTS_DIR = original_results
            eval_mod.CACHE_DIR = original_cache
