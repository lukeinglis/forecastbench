"""Tests for chronological data cutoff enforcement."""

from __future__ import annotations

from datetime import datetime, timezone

from fetch_data import Question
from cutoff import CutoffEnvironment


def _make_question(**kwargs) -> Question:
    defaults = {"id": "q1", "source": "acled", "question": "Will X happen?"}
    defaults.update(kwargs)
    return Question(**defaults)


class TestPrepareContext:
    def test_uses_freeze_datetime(self) -> None:
        env = CutoffEnvironment()
        q = _make_question(freeze_datetime="2024-06-15T12:00:00")
        ctx = env.prepare_context(q, "2024-07-01")
        assert ctx.freeze_datetime.year == 2024
        assert ctx.freeze_datetime.month == 6
        assert ctx.freeze_datetime.day == 15

    def test_falls_back_to_forecast_due_date(self) -> None:
        env = CutoffEnvironment()
        q = _make_question(freeze_datetime=None)
        ctx = env.prepare_context(q, "2024-07-01")
        assert ctx.freeze_datetime.year == 2024
        assert ctx.freeze_datetime.month == 7
        assert ctx.freeze_datetime.day == 1

    def test_context_has_correct_question(self) -> None:
        env = CutoffEnvironment()
        q = _make_question(freeze_datetime="2024-03-10")
        ctx = env.prepare_context(q, "2024-04-01")
        assert ctx.question is q
        assert ctx.forecast_due_date == "2024-04-01"


class TestGetTodayDate:
    def test_formatted_output(self) -> None:
        env = CutoffEnvironment()
        q = _make_question(freeze_datetime="2024-01-15")
        ctx = env.prepare_context(q, "2024-02-01")
        assert env.get_today_date(ctx) == "January 15, 2024"

    def test_different_questions_different_dates(self) -> None:
        env = CutoffEnvironment()
        q1 = _make_question(id="q1", freeze_datetime="2024-03-01")
        q2 = _make_question(id="q2", freeze_datetime="2024-09-20")
        ctx1 = env.prepare_context(q1, "2024-04-01")
        ctx2 = env.prepare_context(q2, "2024-10-01")
        assert env.get_today_date(ctx1) != env.get_today_date(ctx2)
        assert "March" in env.get_today_date(ctx1)
        assert "September" in env.get_today_date(ctx2)


class TestDatetimeFormats:
    def test_iso_with_timezone(self) -> None:
        env = CutoffEnvironment()
        q = _make_question(freeze_datetime="2024-06-15T12:00:00+00:00")
        ctx = env.prepare_context(q, "2024-07-01")
        assert ctx.freeze_datetime == datetime(2024, 6, 15, 12, 0, 0, tzinfo=timezone.utc)

    def test_iso_without_timezone(self) -> None:
        env = CutoffEnvironment()
        q = _make_question(freeze_datetime="2024-06-15T12:00:00")
        ctx = env.prepare_context(q, "2024-07-01")
        assert ctx.freeze_datetime.tzinfo == timezone.utc

    def test_date_only(self) -> None:
        env = CutoffEnvironment()
        q = _make_question(freeze_datetime="2024-06-15")
        ctx = env.prepare_context(q, "2024-07-01")
        assert ctx.freeze_datetime.day == 15

    def test_malformed_uses_fallback(self) -> None:
        env = CutoffEnvironment()
        q = _make_question(freeze_datetime="not-a-date")
        ctx = env.prepare_context(q, "2024-07-01")
        assert ctx.freeze_datetime.month == 7
        assert ctx.freeze_datetime.day == 1


class TestValidateNoFutureAccess:
    def test_always_returns_true(self) -> None:
        env = CutoffEnvironment()
        q = _make_question(freeze_datetime="2024-01-01")
        ctx = env.prepare_context(q, "2024-01-01")
        assert env.validate_no_future_access(ctx) is True
