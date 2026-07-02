"""Dummy forecaster: returns 0.5 for every question."""

from __future__ import annotations

from fetch_data import Question


def forecast(question: Question) -> float:
    """Return 0.5 regardless of the question."""
    return 0.5


if __name__ == "__main__":
    from eval import run_eval
    run_eval(forecast)
