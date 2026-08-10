"""Tests for missing question fields (H1, Issue #9) and multi-horizon forecasts (H2, Issue #8)."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from fetch_data import Question, QuestionSet, ResolvedQuestion, join_resolved_questions, Resolution
from lab_forecaster import _build_prompt
from eval import (
    _build_question,
    _has_multi_horizon,
    _run_async,
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
        resolutions: dict[str, list[Resolution]] = {"q1": [Resolution(id="q1", outcome=1)]}
        result = join_resolved_questions([qs], resolutions)
        assert len(result) == 1
        rq = result[0]
        for field, value in NEW_FIELDS.items():
            assert getattr(rq, field) == value, f"{field} not propagated"

    def test_none_fields_propagated(self) -> None:
        q = Question(id="q2", source="acled", question="Test?")
        qs = QuestionSet(forecast_due_date="2024-01-01", questions=[q])
        resolutions: dict[str, list[Resolution]] = {"q2": [Resolution(id="q2", outcome=0)]}
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
        q = Question(id="q1", source="metaculus", question="Test?")
        prompt = _build_prompt(q, resolution_date="2025-07-15")
        assert "2025-07-15" in prompt

    def test_resolution_date_omitted_when_none(self) -> None:
        q = Question(id="q1", source="metaculus", question="Test?")
        prompt = _build_prompt(q)
        assert "Question resolution date:" in prompt

    def test_resolution_date_uses_upstream_label(self) -> None:
        q = Question(id="q1", source="metaculus", question="Test?")
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


class TestBaseIdForecasting:
    """v0.2.0: forecasting uses base question IDs, scoring handles per-horizon internally."""

    def test_all_questions_use_base_id_key(self, tmp_path: object) -> None:
        from pathlib import Path as P

        tmp = P(str(tmp_path))

        def dummy(q: Question, **kwargs: object) -> float:
            return 0.6

        q = Question(
            id="q1",
            source="acled",
            question="Test?",
            resolution_dates=["2024-07-28", "2025-01-17"],
        )
        with patch("eval.CACHE_DIR", tmp):
            forecasts = _run_sync(dummy, [q], "test")

        assert "q1" in forecasts
        assert len(forecasts) == 1

    def test_market_question_uses_base_id_key(self, tmp_path: object) -> None:
        from pathlib import Path as P

        tmp = P(str(tmp_path))

        def dummy(q: Question, **kwargs: object) -> float:
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

    def test_forecaster_called_once_per_question(self, tmp_path: object) -> None:
        from pathlib import Path as P

        tmp = P(str(tmp_path))
        call_count = 0

        def counting_fn(q: Question, **kwargs: object) -> float:
            nonlocal call_count
            call_count += 1
            return 0.5

        q = Question(
            id="q1",
            source="acled",
            question="Test?",
            resolution_dates=["2024-07-28", "2025-01-17"],
        )
        with patch("eval.CACHE_DIR", tmp):
            _run_sync(counting_fn, [q], "test")

        assert call_count == 1

    def test_cache_uses_base_id(self, tmp_path: object) -> None:
        from pathlib import Path as P

        tmp = P(str(tmp_path))
        call_count = 0

        def counting_fn(q: Question, **kwargs: object) -> float:
            nonlocal call_count
            call_count += 1
            return 0.7

        with patch("eval.CACHE_DIR", tmp):
            _write_cache("test", "q1", 0.99)
            q = Question(
                id="q1",
                source="acled",
                question="Test?",
                resolution_dates=["2024-07-28", "2025-01-17"],
            )
            forecasts = _run_sync(counting_fn, [q], "test")

        assert forecasts["q1"] == pytest.approx(0.99)
        assert call_count == 0


class TestAsyncBaseIdForecasting:
    async def test_async_uses_base_id_key(self, tmp_path: object) -> None:
        from pathlib import Path as P

        tmp = P(str(tmp_path))

        async def dummy(q: Question, **kwargs: object) -> float:
            return 0.5

        q = Question(
            id="q1",
            source="acled",
            question="Test?",
            resolution_dates=["2024-07-28", "2025-01-17"],
        )
        with patch("eval.CACHE_DIR", tmp):
            forecasts = await _run_async(dummy, [q], "test")

        assert "q1" in forecasts
        assert len(forecasts) == 1


class TestMultiHorizonEndToEnd:
    def test_base_id_forecasts_score_against_multi_horizon_resolved(self) -> None:
        from score import score_forecasts

        resolved = [
            ResolvedQuestion(
                id="q1", source="acled", question="Test?",
                outcome=1, resolution_date="2024-07-28",
                resolution_dates=["2024-07-28", "2025-01-17"],
                forecast_due_date="2024-01-01",
            ),
            ResolvedQuestion(
                id="q1", source="acled", question="Test?",
                outcome=0, resolution_date="2025-01-17",
                resolution_dates=["2024-07-28", "2025-01-17"],
                forecast_due_date="2024-01-01",
            ),
        ]
        forecasts = {"q1": 0.8}
        result = score_forecasts(forecasts, resolved)
        assert result.n_dataset == 2
        assert result.n_missing == 0
