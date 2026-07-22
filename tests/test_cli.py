"""CLI smoke tests for eval.py, analyze.py, and submit.py."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

from fetch_data import Question, QuestionSet, Resolution


def _run_script(script: str, *args: str, timeout: int = 30) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, script, *args],
        capture_output=True,
        text=True,
        timeout=timeout,
    )


class TestEvalHelp:
    def test_help_exits_zero(self) -> None:
        result = _run_script("eval.py", "--help")
        assert result.returncode == 0
        assert "ForecastBench evaluation" in result.stdout


class TestEvalListRounds:
    def test_list_rounds_completes(self, capsys: pytest.CaptureFixture[str]) -> None:
        mock_files = ["2026-07-05-llm.json"]
        mock_qs = QuestionSet(
            forecast_due_date="2026-07-05",
            question_set="2026-07-05-llm",
            questions=[
                Question(id="q1", source="acled", question="Test Q1"),
            ],
        )
        with patch("eval.list_question_set_files", return_value=mock_files), \
             patch("eval.fetch_question_set", return_value=mock_qs), \
             patch("sys.argv", ["eval.py", "--list-rounds"]):
            from eval import main
            main()
        captured = capsys.readouterr()
        assert "2026-07-05-llm" in captured.out

    def test_list_rounds_prints_output(self, capsys: pytest.CaptureFixture[str]) -> None:
        mock_files = ["2026-07-05-llm.json"]
        mock_qs = QuestionSet(
            forecast_due_date="2026-07-05",
            question_set="2026-07-05-llm",
            questions=[
                Question(id=f"q{i}", source="acled", question=f"Test Q{i}")
                for i in range(10)
            ],
        )
        with patch("eval.list_question_set_files", return_value=mock_files), \
             patch("eval.fetch_question_set", return_value=mock_qs), \
             patch("sys.argv", ["eval.py", "--list-rounds"]):
            from eval import main
            main()
        captured = capsys.readouterr()
        assert "2026-07-05-llm" in captured.out
        assert "10" in captured.out


class TestEvalDummyAgent:
    def test_dummy_agent_runs(self, tmp_path: Path) -> None:
        import eval as eval_mod

        mock_qs = QuestionSet(
            forecast_due_date="2026-07-05",
            question_set="2026-07-05-llm",
            questions=[
                Question(id="q1", source="acled", question="Test Q1"),
            ],
        )
        mock_resolutions = {
            "q1": Resolution(id="q1", outcome=1, resolution_date="2026-07-19"),
        }

        with patch("eval.fetch_question_set", return_value=mock_qs), \
             patch("eval.fetch_all_resolutions", return_value=mock_resolutions), \
             patch.object(eval_mod, "RESULTS_DIR", tmp_path / "results"), \
             patch.object(eval_mod, "CACHE_DIR", tmp_path / "cache"), \
             patch("sys.argv", ["eval.py", "--agent", "dummy", "--round", "2026-07-05-llm"]):
            eval_mod.main()

        results_files = list((tmp_path / "results").glob("*.json"))
        assert len(results_files) == 1


class TestEvalRoundWithDummy:
    def test_round_flag_with_dummy(self, tmp_path: Path) -> None:
        import eval as eval_mod

        mock_qs = QuestionSet(
            forecast_due_date="2026-07-05",
            question_set="2026-07-05-llm",
            questions=[
                Question(id="q1", source="acled", question="Will event X happen?"),
                Question(id="q2", source="metaculus", question="Will event Y happen?"),
            ],
        )
        mock_resolutions = {
            "q1": Resolution(id="q1", outcome=0, resolution_date="2026-07-19"),
            "q2": Resolution(id="q2", outcome=1, resolution_date="2026-07-19"),
        }

        with patch("eval.fetch_question_set", return_value=mock_qs), \
             patch("eval.fetch_all_resolutions", return_value=mock_resolutions), \
             patch.object(eval_mod, "RESULTS_DIR", tmp_path / "results"), \
             patch.object(eval_mod, "CACHE_DIR", tmp_path / "cache"), \
             patch("sys.argv", ["eval.py", "--round", "2026-07-05", "--agent", "dummy"]):
            eval_mod.main()

        results_files = list((tmp_path / "results").glob("*.json"))
        assert len(results_files) == 1


class TestSubmitFlag:
    def test_submit_flag_accepted(self) -> None:
        result = _run_script("eval.py", "--help")
        assert result.returncode == 0
        assert "--submit" in result.stdout

    def test_submit_mode_defaults_to_false(self) -> None:
        import argparse
        from unittest.mock import patch as _patch

        with _patch("sys.argv", ["eval.py", "--agent", "dummy"]):
            parser = argparse.ArgumentParser()
            parser.add_argument("--agent", choices=["dummy", "baseline"], default="dummy")
            parser.add_argument("--submit", action="store_true", default=False)
            args = parser.parse_args(["--agent", "dummy"])
            assert args.submit is False

    def test_submit_mode_with_dummy_agent(self, tmp_path: Path) -> None:
        import eval as eval_mod

        mock_qs = QuestionSet(
            forecast_due_date="2026-07-05",
            question_set="2026-07-05-llm",
            questions=[
                Question(id="q1", source="acled", question="Test Q1"),
                Question(id="q2", source="acled", question="Test Q2 (unresolved)"),
            ],
        )
        mock_resolutions = {
            "q1": Resolution(id="q1", outcome=1, resolution_date="2026-07-19"),
        }

        with patch("eval.fetch_question_set", return_value=mock_qs), \
             patch("eval.fetch_all_resolutions", return_value=mock_resolutions), \
             patch.object(eval_mod, "RESULTS_DIR", tmp_path / "results"), \
             patch.object(eval_mod, "CACHE_DIR", tmp_path / "cache"), \
             patch("sys.argv", ["eval.py", "--agent", "dummy", "--round", "2026-07-05-llm", "--submit"]):
            eval_mod.main()

        results_files = list((tmp_path / "results").glob("submit_*.json"))
        assert len(results_files) == 1


class TestAnalyzeHelp:
    def test_help_exits_zero(self) -> None:
        result = _run_script("analyze.py", "--help")
        assert result.returncode == 0
        assert "ForecastBench analysis" in result.stdout


class TestSubmitHelp:
    def test_help_exits_zero(self) -> None:
        result = _run_script("submit.py", "--help")
        assert result.returncode == 0
        assert "ForecastBench submission" in result.stdout
