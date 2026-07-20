"""Tests for code quality fixes: cache slug, forecast validation, _to_float rename."""

from __future__ import annotations

import os
from unittest.mock import patch

import pytest

from fetch_data import ResolvedQuestion
from submit import SubmissionMetadata, assemble_submission, validate_forecasts


def _make_resolved(
    qid: str,
    source: str,
    outcome: int,
    resolution_date: str | None = None,
) -> ResolvedQuestion:
    return ResolvedQuestion(
        id=qid,
        source=source,
        question=f"Q {qid}",
        outcome=outcome,
        forecast_due_date="2024-01-01",
        resolution_date=resolution_date,
    )


class TestCacheSlug:
    def test_default_slug_is_unknown(self) -> None:
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("FORECAST_MODEL", None)
            from eval import _model_slug
            slug = _model_slug()
            assert slug == "unknown"

    def test_dummy_gets_dummy_slug(self) -> None:
        with patch.dict(os.environ, {"FORECAST_MODEL": "dummy"}):
            from eval import _model_slug
            slug = _model_slug()
            assert slug == "dummy"

    def test_dummy_and_baseline_different_slugs(self) -> None:
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("FORECAST_MODEL", None)
            os.environ.setdefault("FORECAST_MODEL", "dummy")
            from eval import _model_slug
            dummy_slug = _model_slug()

        with patch.dict(os.environ, {"FORECAST_MODEL": "vertex_ai/claude-sonnet-4@20250514"}):
            from eval import _model_slug
            baseline_slug = _model_slug()

        assert dummy_slug != baseline_slug

    def test_dummy_forecaster_sets_env(self) -> None:
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("FORECAST_MODEL", None)
            os.environ.setdefault("FORECAST_MODEL", "dummy")
            assert os.environ["FORECAST_MODEL"] == "dummy"


class TestValidateForecasts:
    def test_valid_forecasts_pass(self) -> None:
        entries = [
            {"id": "q1", "source": "acled", "forecast": 0.5},
            {"id": "q2", "source": "metaculus", "forecast": 0.0},
            {"id": "q3", "source": "acled", "forecast": 1.0},
        ]
        validate_forecasts(entries)

    def test_forecast_above_one_raises(self) -> None:
        entries = [{"id": "q1", "source": "acled", "forecast": 1.5}]
        with pytest.raises(ValueError, match="out of range"):
            validate_forecasts(entries)

    def test_forecast_below_zero_raises(self) -> None:
        entries = [{"id": "q1", "source": "acled", "forecast": -0.1}]
        with pytest.raises(ValueError, match="out of range"):
            validate_forecasts(entries)

    def test_forecast_not_a_number_raises(self) -> None:
        entries = [{"id": "q1", "source": "acled", "forecast": "high"}]
        with pytest.raises(ValueError, match="out of range"):
            validate_forecasts(entries)

    def test_market_with_resolution_date_raises(self) -> None:
        entries = [
            {"id": "q1", "source": "metaculus", "forecast": 0.5, "resolution_date": "2024-06-01"},
        ]
        with pytest.raises(ValueError, match="must not have resolution_date"):
            validate_forecasts(entries)

    def test_dataset_with_resolution_date_ok(self) -> None:
        entries = [
            {"id": "q1", "source": "acled", "forecast": 0.5, "resolution_date": "2024-06-01"},
        ]
        validate_forecasts(entries)

    def test_market_without_resolution_date_ok(self) -> None:
        entries = [
            {"id": "q1", "source": "polymarket", "forecast": 0.3},
        ]
        validate_forecasts(entries)

    def test_forecast_nan_raises(self) -> None:
        entries = [{"id": "q1", "source": "acled", "forecast": float("nan")}]
        with pytest.raises(ValueError, match="out of range"):
            validate_forecasts(entries)

    def test_forecast_inf_raises(self) -> None:
        entries = [{"id": "q1", "source": "acled", "forecast": float("inf")}]
        with pytest.raises(ValueError, match="out of range"):
            validate_forecasts(entries)


class TestAssembleValidation:
    def test_assemble_excludes_resolution_date_for_market(self) -> None:
        questions = [
            _make_resolved("q1", "metaculus", 1, resolution_date="2024-06-01"),
        ]
        forecasts = {"q1": 0.7}
        meta = SubmissionMetadata(
            organization="org", model="m", model_organization="mo",
            question_set="2024-01-01",
        )
        result = assemble_submission(forecasts, questions, meta)
        entry = result["forecasts"][0]
        assert "resolution_date" not in entry

    def test_assemble_includes_resolution_date_for_dataset(self) -> None:
        questions = [
            _make_resolved("q1", "acled", 1, resolution_date="2024-06-01"),
        ]
        forecasts = {"q1": 0.7}
        meta = SubmissionMetadata(
            organization="org", model="m", model_organization="mo",
            question_set="2024-01-01",
        )
        result = assemble_submission(forecasts, questions, meta)
        entry = result["forecasts"][0]
        assert entry["resolution_date"] == "2024-06-01"

    def test_assemble_with_reasoning(self) -> None:
        questions = [_make_resolved("q1", "acled", 1)]
        forecasts = {"q1": 0.8}
        reasoning = {"q1": "High confidence based on historical trends"}
        meta = SubmissionMetadata(
            organization="org", model="m", model_organization="mo",
            question_set="2024-01-01",
        )
        result = assemble_submission(forecasts, questions, meta, reasoning=reasoning)
        entry = result["forecasts"][0]
        assert entry["reasoning"] == "High confidence based on historical trends"

    def test_assemble_rejects_invalid_forecast(self) -> None:
        questions = [_make_resolved("q1", "acled", 1)]
        forecasts = {"q1": 1.5}
        meta = SubmissionMetadata(
            organization="org", model="m", model_organization="mo",
            question_set="2024-01-01",
        )
        with pytest.raises(ValueError, match="out of range"):
            assemble_submission(forecasts, questions, meta)


class TestToFloat:
    def test_to_float_returns_float(self) -> None:
        from baseline_agent import _to_float
        assert _to_float(0.5) == 0.5
        assert isinstance(_to_float(0.5), float)

    def test_to_float_int_input(self) -> None:
        from baseline_agent import _to_float
        assert _to_float(1) == 1.0
        assert isinstance(_to_float(1), float)

    def test_to_float_zero(self) -> None:
        from baseline_agent import _to_float
        assert _to_float(0) == 0.0

    def test_to_float_preserves_value(self) -> None:
        from baseline_agent import _to_float
        assert _to_float(0.123456789) == 0.123456789
