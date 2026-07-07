"""Tests for chronological cutoff enforcement."""

from __future__ import annotations

from cutoff import CutoffEnvironment, format_cutoff_instruction
from fetch_data import Question
from baseline_agent import _build_user_prompt


class TestFormatCutoffInstruction:
    def test_produces_nonempty_string(self) -> None:
        result = format_cutoff_instruction("2024-03-15")
        assert isinstance(result, str)
        assert len(result) > 0

    def test_contains_date(self) -> None:
        result = format_cutoff_instruction("2024-03-15")
        assert "2024-03-15" in result

    def test_different_dates(self) -> None:
        r1 = format_cutoff_instruction("2023-01-01")
        r2 = format_cutoff_instruction("2025-12-31")
        assert "2023-01-01" in r1
        assert "2025-12-31" in r2


class TestCutoffInPrompt:
    def test_cutoff_included_when_provided(self) -> None:
        q = Question(id="q1", source="acled", question="Will X happen?")
        instruction = format_cutoff_instruction("2024-06-01")
        prompt = _build_user_prompt(q, cutoff_instruction=instruction)
        assert "2024-06-01" in prompt

    def test_cutoff_absent_when_not_provided(self) -> None:
        q = Question(id="q1", source="acled", question="Will X happen?")
        prompt = _build_user_prompt(q, cutoff_instruction=None)
        assert "knowledge cutoff" not in prompt.lower()

    def test_cutoff_absent_by_default(self) -> None:
        q = Question(id="q1", source="acled", question="Will X happen?")
        prompt = _build_user_prompt(q)
        assert "knowledge cutoff" not in prompt.lower()


class TestCutoffEnvironment:
    def test_frozen_dataclass(self) -> None:
        env = CutoffEnvironment(freeze_datetime="2024-03-15")
        assert env.freeze_datetime == "2024-03-15"
