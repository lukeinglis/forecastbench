"""Chronological data cutoff enforcement for ForecastBench."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class CutoffEnvironment:
    freeze_datetime: str


def format_cutoff_instruction(freeze_datetime: str) -> str:
    """Return a prompt fragment instructing the model about its knowledge cutoff."""
    return (
        f"Important: Your knowledge cutoff for this question is {freeze_datetime}. "
        "Do not use any information from after this date. "
        "Base your forecast only on what was known before this date."
    )
