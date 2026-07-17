"""Tests for missing question fields (H1, Issue #9) and multi-horizon forecasts (H2, Issue #8)."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from fetch_data import Question, QuestionSet, ResolvedQuestion, join_resolved_questions, Resolution
from baseline_agent import _build_prompt
from eval import (
    _build_question,
    _has_multi_horizon,
    _expand_resolved_for_horizons,
    _run_sync,
    _write_cache,
)


NEW_FIELDS = {
    "source_intro": "We would like you to predict the outcome of a prediction market question.",
    "freeze_datetime_value_explanation": "The market value.",
    "market_info_open_datetime": "2024-01-01T00:00:00Z",
    "market_info_close_datetime": "2024-12-31T23:59:59Z",
    "market_info_resolution_criteria": "Resolves based on official data.",
}


def _make_question_with_fields(**overrides: object) -> Question:
    defaults: dict[str, object] = {
        "id": "q1",
        "source": "metaculus",
        "question": "Will X happen?",
        **NEW_FIELDS,
    }
    defaults.update(overrides)
    return Question(**defaults)  # type: ignore[arg-type]


class TestNewFieldsOnQuestion:
    def test_all_new_fields_stored(self) -> None:
        q = _make_question_with_fields()
        for field, value in NEW_FIELDS.items():
            assert getattr(q, field) == value

    def test_new_fields_default_to_none(self) -> None:
        q = Question(id="q1", source="acled", question="Test?")
        for field in NEW_FIELDS:
            assert getattr(q, field) is None

    def test_backward_compatibility_no_new_fields(self) -> None:
        q = Question(id="q1", source="acled", question="Test?", background="bg")
        assert q.id == "q1"
        assert q.background == "bg"


class TestNewFieldsOnResolvedQuestion:
    def test_all_new_fields_stored(self) -> None:
        rq = ResolvedQuestion(
            id="q1",
            source="acled",
            question="Test?",
            outcome=1,
            **NEW_FIELDS,
        )
        for field, value in NEW_FIELDS.items():
            assert getattr(rq, field) == value

    def test_new_fields_default_to_none(self) -> None:
        rq = ResolvedQuestion(id="q1", source="acled", question="Test?", outcome=0)
        for field in NEW_FIELDS:
            assert getattr(rq, field) is None


class TestJoinPropagatesNewFields:
    def test_new_fields_propagated(self) -> None:
        q = _make_question_with_fields(source="acled")
        qs = QuestionSet(
            forecast_due_date="2024-01-01",
            questions=[q],
        )
        resolutions = {"q1": Resolution(id="q1", outcome=1)}
        result = join_resolved_questions([qs], resolutions)
        assert len(result) == 1
        rq = result[0]
        for field, value in NEW_FIELDS.items():
            assert getattr(rq, field) == value, f"{field} not propagated"

    def test_none_fields_propagated(self) -> None:
        q = Question(id="q2", source="acled", question="Test?")
        qs = QuestionSet(forecast_due_date="2024-01-01", questions=[q])
        resolutions = {"q2": Resolution(id="q2", outcome=0)}
        result = join_resolved_questions([qs], resolutions)
        rq = result[0]
        for field in NEW_FIELDS:
            assert getattr(rq, field) is None


class TestBuildQuestionPropagatesNewFields:
    def test_from_question(self) -> None:
        q = _make_question_with_fields()
        built = _build_question(q)
        for field, value in NEW_FIELDS.items():
            assert getattr(built, field) == value

    def test_from_resolved_question(self) -> None:
        rq = ResolvedQuestion(
            id="q1",
            source="acled",
            question="Test?",
            outcome=1,
            **NEW_FIELDS,
        )
        built = _build_question(rq)
        for field, value in NEW_FIELDS.items():
            assert getattr(built, field) == value

    def test_none_defaults(self) -> None:
        q = Question(id="q1", source="acled", question="Test?")
        built = _build_question(q)
        for field in NEW_FIELDS:
            assert getattr(built, field) is None


class TestPromptSourceIntro:
    def test_source_intro_not_in_upstream_prompt(self) -> None:
        q = _make_question_with_fields(freeze_datetime=None)
        prompt = _build_prompt(q)
        assert "Source Context:" not in prompt

    def test_source_intro_omitted_when_none(self) -> None:
        q = Question(id="q1", source="acled", question="Test?")
        prompt = _build_prompt(q)
        assert "Source Context:" not in prompt


class TestPromptResolutionDate:
    def test_resolution_date_included_when_provided(self) -> None:
        q = Question(id="q1", source="acled", question="Test?")
        prompt = _build_prompt(q, resolution_date="2025-07-15")
        assert "2025-07-15" in prompt

    def test_resolution_date_omitted_when_none(self) -> None:
        q = Question(id="q1", source="acled", question="Test?")
        prompt = _build_prompt(q)
        assert "Question resolution date:" in prompt

    def test_resolution_date_uses_upstream_label(self) -> None:
        q = Question(id="q1", source="acled", question="Test?")
        prompt = _build_prompt(q, resolution_date="2025-07-15")
        assert "Question resolution date:" in prompt


class TestHasMultiHorizon:
    def test_dataset_with_list(self) -> None:
        q = Question(id="q1", source="acled", question="Test?", resolution_dates=["2024-07-28", "2025-01-17"])
        assert _has_multi_horizon(q) is True

    def test_dataset_with_empty_list(self) -> None:
        q = Question(id="q1", source="acled", question="Test?", resolution_dates=[])
        assert _has_multi_horizon(q) is False

    def test_dataset_with_na(self) -> None:
        q = Question(id="q1", source="acled", question="Test?", resolution_dates="N/A")
        assert _has_multi_horizon(q) is False

    def test_dataset_with_none(self) -> None:
        q = Question(id="q1", source="acled", question="Test?", resolution_dates=None)
        assert _has_multi_horizon(q) is False

    def test_market_question_never_multi(self) -> None:
        q = Question(id="q1", source="metaculus", question="Test?", resolution_dates=["2024-07-28"])
        assert _has_multi_horizon(q) is False

    def test_polymarket_never_multi(self) -> None:
        q = Question(id="q1", source="polymarket", question="Test?", resolution_dates=["2024-07-28"])
        assert _has_multi_horizon(q) is False


class TestExpandResolvedForHorizons:
    def test_dataset_expanded(self) -> None:
        rq = ResolvedQuestion(
            id="q1",
            source="acled",
            question="Test?",
            outcome=1,
            resolution_dates=["2024-07-28", "2025-01-17"],
            forecast_due_date="2024-01-01",
        )
        expanded = _expand_resolved_for_horizons([rq])
        assert len(expanded) == 2
        assert expanded[0].id == "q1_2024-07-28"
        assert expanded[1].id == "q1_2025-01-17"
        assert expanded[0].resolution_date == "2024-07-28"
        assert expanded[1].resolution_date == "2025-01-17"

    def test_market_not_expanded(self) -> None:
        rq = ResolvedQuestion(
            id="m1",
            source="metaculus",
            question="Market Q?",
            outcome=0,
            resolution_dates="N/A",
            forecast_due_date="2024-01-01",
        )
        expanded = _expand_resolved_for_horizons([rq])
        assert len(expanded) == 1
        assert expanded[0].id == "m1"

    def test_dataset_no_resolution_dates_not_expanded(self) -> None:
        rq = ResolvedQuestion(
            id="d1",
            source="acled",
            question="Test?",
            outcome=1,
            resolution_dates=None,
            forecast_due_date="2024-01-01",
        )
        expanded = _expand_resolved_for_horizons([rq])
        assert len(expanded) == 1
        assert expanded[0].id == "d1"

    def test_expanded_preserves_fields(self) -> None:
        rq = ResolvedQuestion(
            id="q1",
            source="acled",
            question="Test?",
            background="bg",
            outcome=1,
            resolution_dates=["2024-07-28"],
            source_intro="Intro text",
            forecast_due_date="2024-01-01",
        )
        expanded = _expand_resolved_for_horizons([rq])
        assert expanded[0].background == "bg"
        assert expanded[0].source_intro == "Intro text"
        assert expanded[0].outcome == 1

    def test_mixed_questions(self) -> None:
        dataset_q = ResolvedQuestion(
            id="d1",
            source="acled",
            question="Dataset?",
            outcome=1,
            resolution_dates=["2024-07-28", "2025-01-17"],
            forecast_due_date="2024-01-01",
        )
        market_q = ResolvedQuestion(
            id="m1",
            source="metaculus",
            question="Market?",
            outcome=0,
            resolution_dates="N/A",
            forecast_due_date="2024-01-01",
        )
        expanded = _expand_resolved_for_horizons([dataset_q, market_q])
        assert len(expanded) == 3
        ids = [e.id for e in expanded]
        assert "d1_2024-07-28" in ids
        assert "d1_2025-01-17" in ids
        assert "m1" in ids


class TestMultiHorizonSyncPath:
    def test_multi_horizon_produces_composite_keys(self, tmp_path: object) -> None:
        from pathlib import Path as P

        tmp = P(str(tmp_path))

        def dummy(q: Question, resolution_date: str | None = None, **kwargs: object) -> float:
            return 0.6

        q = Question(
            id="q1",
            source="acled",
            question="Test?",
            resolution_dates=["2024-07-28", "2025-01-17"],
        )
        with patch("eval.CACHE_DIR", tmp):
            forecasts = _run_sync(dummy, [q], "test")

        assert "q1_2024-07-28" in forecasts
        assert "q1_2025-01-17" in forecasts
        assert "q1" not in forecasts

    def test_market_question_uses_simple_key(self, tmp_path: object) -> None:
        from pathlib import Path as P

        tmp = P(str(tmp_path))

        def dummy(q: Question, resolution_date: str | None = None, **kwargs: object) -> float:
            return 0.5

        q = Question(
            id="m1",
            source="metaculus",
            question="Market?",
            resolution_dates=["2024-07-28"],
        )
        with patch("eval.CACHE_DIR", tmp):
            forecasts = _run_sync(dummy, [q], "test")

        assert "m1" in forecasts
        assert len(forecasts) == 1

    def test_multi_horizon_passes_resolution_date_to_forecaster(self, tmp_path: object) -> None:
        from pathlib import Path as P

        tmp = P(str(tmp_path))
        calls: list[str | None] = []

        def tracking_fn(q: Question, resolution_date: str | None = None, **kwargs: object) -> float:
            calls.append(resolution_date)
            return 0.5

        q = Question(
            id="q1",
            source="acled",
            question="Test?",
            resolution_dates=["2024-07-28", "2025-01-17"],
        )
        with patch("eval.CACHE_DIR", tmp):
            _run_sync(tracking_fn, [q], "test")

        assert calls == ["2024-07-28", "2025-01-17"]

    def test_multi_horizon_uses_cache(self, tmp_path: object) -> None:
        from pathlib import Path as P

        tmp = P(str(tmp_path))
        call_count = 0

        def counting_fn(q: Question, resolution_date: str | None = None, **kwargs: object) -> float:
            nonlocal call_count
            call_count += 1
            return 0.7

        with patch("eval.CACHE_DIR", tmp):
            _write_cache("test", "q1_2024-07-28", 0.99)
            q = Question(
                id="q1",
                source="acled",
                question="Test?",
                resolution_dates=["2024-07-28", "2025-01-17"],
            )
            forecasts = _run_sync(counting_fn, [q], "test")

        assert forecasts["q1_2024-07-28"] == pytest.approx(0.99)
        assert forecasts["q1_2025-01-17"] == pytest.approx(0.7)
        assert call_count == 1


class TestMultiHorizonEndToEnd:
    def test_expanded_forecasts_score_against_expanded_resolved(self) -> None:
        from score import score_forecasts

        rq = ResolvedQuestion(
            id="q1",
            source="acled",
            question="Test?",
            outcome=1,
            resolution_dates=["2024-07-28", "2025-01-17"],
            forecast_due_date="2024-01-01",
        )
        expanded = _expand_resolved_for_horizons([rq])
        forecasts = {
            "q1_2024-07-28": 0.8,
            "q1_2025-01-17": 0.9,
        }
        result = score_forecasts(forecasts, expanded)
        assert result.n_dataset == 2
        assert result.n_missing == 0
