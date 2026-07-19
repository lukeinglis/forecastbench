"""Tests for leaderboard fetching, caching, and comparison display."""

from __future__ import annotations

import io
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from fetch_data import fetch_leaderboard, LEADERBOARD_NAMES


SAMPLE_CSV = """Rank,Team,Model Organization,Model,Dataset,Market,Overall,N,95% CI
1,Team Alpha,Org A,Superforecaster median,72.1,65.3,68.5,500,1.2
2,Team Beta,Org B,o3 (scratchpad),66.0,59.1,62.3,500,1.5
3,Team Gamma,Org C,Claude Sonnet 4.5,64.2,59.5,61.7,500,1.4
4,Team Delta,Org D,GPT-4o,63.1,58.0,60.2,500,1.6
5,Team Epsilon,Org E,Llama 3.1,60.0,55.0,57.2,500,1.8
6,Team Zeta,Org F,Always 0.5,50.5,49.5,50.0,500,2.0
"""


class TestFetchLeaderboard:
    @patch("fetch_data.requests.get")
    def test_fetches_and_parses_csv(self, mock_get: MagicMock, tmp_path: Path) -> None:
        mock_resp = MagicMock()
        mock_resp.text = SAMPLE_CSV
        mock_resp.raise_for_status = MagicMock()
        mock_get.return_value = mock_resp

        with patch("fetch_data.CACHE_DIR", tmp_path):
            rows = fetch_leaderboard("baseline")

        assert len(rows) == 6
        assert rows[0]["Rank"] == "1"
        assert rows[0]["Model"] == "Superforecaster median"
        assert rows[0]["Overall"] == "68.5"

    @patch("fetch_data.requests.get")
    def test_caches_csv_on_disk(self, mock_get: MagicMock, tmp_path: Path) -> None:
        mock_resp = MagicMock()
        mock_resp.text = SAMPLE_CSV
        mock_resp.raise_for_status = MagicMock()
        mock_get.return_value = mock_resp

        with patch("fetch_data.CACHE_DIR", tmp_path):
            fetch_leaderboard("baseline")
            fetch_leaderboard("baseline")

        mock_get.assert_called_once()

    def test_rejects_unknown_leaderboard(self) -> None:
        with pytest.raises(ValueError, match="Unknown leaderboard"):
            fetch_leaderboard("nonexistent")

    def test_supported_leaderboard_names(self) -> None:
        assert "baseline" in LEADERBOARD_NAMES
        assert "tournament" in LEADERBOARD_NAMES
        assert "dataset" in LEADERBOARD_NAMES
        assert "preliminary" in LEADERBOARD_NAMES

    @patch("fetch_data.requests.get")
    def test_handles_empty_csv(self, mock_get: MagicMock, tmp_path: Path) -> None:
        mock_resp = MagicMock()
        mock_resp.text = "Rank,Model,Overall\n"
        mock_resp.raise_for_status = MagicMock()
        mock_get.return_value = mock_resp

        with patch("fetch_data.CACHE_DIR", tmp_path):
            rows = fetch_leaderboard("baseline")

        assert rows == []


class TestLeaderboardComparison:
    @patch("eval.fetch_leaderboard")
    def test_prints_comparison(self, mock_lb: MagicMock, capsys: pytest.CaptureFixture[str]) -> None:
        import csv
        mock_lb.return_value = list(csv.DictReader(io.StringIO(SAMPLE_CSV)))

        from eval import print_leaderboard_comparison
        print_leaderboard_comparison(58.3)

        captured = capsys.readouterr()
        assert "Your result" in captured.out
        assert "Superforecaster median" in captured.out
        assert "58.3%" in captured.out

    @patch("eval.fetch_leaderboard")
    def test_user_ranks_correctly(self, mock_lb: MagicMock, capsys: pytest.CaptureFixture[str]) -> None:
        import csv
        mock_lb.return_value = list(csv.DictReader(io.StringIO(SAMPLE_CSV)))

        from eval import print_leaderboard_comparison
        print_leaderboard_comparison(70.0)

        captured = capsys.readouterr()
        assert "70.0%" in captured.out

    @patch("eval.fetch_leaderboard")
    def test_handles_fetch_failure(self, mock_lb: MagicMock, capsys: pytest.CaptureFixture[str]) -> None:
        mock_lb.side_effect = ConnectionError("Network error")

        from eval import print_leaderboard_comparison
        print_leaderboard_comparison(55.0)

        captured = capsys.readouterr()
        assert "Could not fetch leaderboard" in captured.out


class TestAnalyzeLeaderboard:
    @patch("analyze.fetch_leaderboard")
    def test_compare_to_leaderboard(self, mock_lb: MagicMock, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
        import csv
        import json
        mock_lb.return_value = list(csv.DictReader(io.StringIO(SAMPLE_CSV)))

        result_data = {
            "timestamp": "20260101T000000Z",
            "model_slug": "test_model",
            "scoring_result": {
                "dataset_brier": 0.2,
                "dataset_index": 55.3,
                "market_brier": 0.3,
                "market_index": 45.2,
                "overall_brier": 0.25,
                "overall_index": 50.0,
                "n_dataset": 100,
                "n_market": 100,
                "n_missing": 0,
                "difficulty_adjusted": False,
            },
            "forecasts": {},
            "outcomes": {},
            "metadata": {"n_questions": 200, "n_held_out": 2, "question_sets_used": []},
        }
        result_file = tmp_path / "result.json"
        result_file.write_text(json.dumps(result_data))

        from analyze import compare_to_leaderboard
        compare_to_leaderboard(str(tmp_path))

        captured = capsys.readouterr()
        assert "test_model" in captured.out
        assert "50.0%" in captured.out

    def test_compare_to_leaderboard_no_dir(self, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
        from analyze import compare_to_leaderboard
        compare_to_leaderboard(str(tmp_path / "nonexistent"))

        captured = capsys.readouterr()
        assert "No results directory" in captured.out
