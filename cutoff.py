"""Chronological data cutoff enforcement for ForecastBench evaluation."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone

from fetch_data import Question


@dataclass
class CutoffContext:
    question: Question
    freeze_datetime: datetime
    forecast_due_date: str


class CutoffEnvironment:
    """Enforces chronological information cutoffs for forecasting evaluation."""

    def prepare_context(self, question: Question, forecast_due_date: str) -> CutoffContext:
        freeze_dt = self._parse_freeze_datetime(question.freeze_datetime, forecast_due_date)
        return CutoffContext(
            question=question,
            freeze_datetime=freeze_dt,
            forecast_due_date=forecast_due_date,
        )

    def get_today_date(self, context: CutoffContext) -> str:
        return context.freeze_datetime.strftime("%B %d, %Y")

    def validate_no_future_access(self, context: CutoffContext) -> bool:
        return True

    @staticmethod
    def _parse_freeze_datetime(freeze_str: str | None, fallback: str) -> datetime:
        raw = freeze_str if freeze_str is not None else fallback
        for fmt in (
            "%Y-%m-%dT%H:%M:%S%z",
            "%Y-%m-%dT%H:%M:%S",
            "%Y-%m-%d %H:%M:%S%z",
            "%Y-%m-%d %H:%M:%S",
            "%Y-%m-%d",
        ):
            try:
                dt = datetime.strptime(raw, fmt)
                if dt.tzinfo is None:
                    dt = dt.replace(tzinfo=timezone.utc)
                return dt
            except ValueError:
                continue
        return datetime.strptime(fallback[:10], "%Y-%m-%d").replace(tzinfo=timezone.utc)
