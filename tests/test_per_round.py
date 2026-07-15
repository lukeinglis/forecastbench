"""Tests for per-round evaluation (--round, --list-rounds, --by-round)."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from unittest.mock import patch

import pytest

from fetch_data import Question, QuestionSet, Resolution
from eval import (
    run_eval,
    save_result,
    list_rounds,
    _normalize_round_name,
)
from score import ScoringResult


def _dummy_forecaster(question: Question, resolution_date: str | None = None) -> float:
    return 0.5


def _make_round_question_set(round_name: str = "2026-07-05-llm") -> QuestionSet:
    return QuestionSet(
        forecast_due_date="2026-07-05",
        question_set=round_name,
        questions=[
            Question(id=f"rq{i}", source="acled", question=f"Round Q{i}")
            for i in range(5)
        ],
    )


def _make_resolutions() -> dict[str, Resolution]:
    return {
        f"rq{i}": Resolution(id=f"rq{i}", outcome=i % 2, resolution_date="2026-07-19")
        for i in range(5)
    }


class TestNormalizeRoundName:
    def test_appends_llm_suffix(self) -> None:
        assert _normalize_round_name("2026-07-05") == "2026-07-05-llm"

    def test_keeps_llm_suffix(self) -> None:
        assert _normalize_round_name("2026-07-05-llm") == "2026-07-05-llm"

    def test_keeps_human_suffix(self) -> None:
        assert _normalize_round_name("2024-07-21-human") == "2024-07-21-human"

    def test_strips_json_extension(self) -> None:
        assert _normalize_round_name("2026-07-05-llm.json") == "2026-07-05-llm"


class TestRoundEvalLoadsOnlySpecifiedSet:
    def test_round_loads_single_question_set(self, tmp_path: Path, monkeypatch: object) -> None:
        import eval as eval_mod

        round_qs = _make_round_question_set()
        resolutions = _make_resolutions()

        monkeypatch.setattr(eval_mod, "fetch_question_set", lambda f: round_qs)  # type: ignore[attr-defined]
        monkeypatch.setattr(eval_mod, "fetch_all_resolutions", lambda: resolutions)  # type: ignore[attr-defined]
        monkeypatch.setattr(eval_mod, "RESULTS_DIR", tmp_path / "results")  # type: ignore[attr-defined]
        monkeypatch.setattr(eval_mod, "CACHE_DIR", tmp_path / "cache")  # type: ignore[attr-defined]

        result = asyncio.run(run_eval(_dummy_forecaster, round_name="2026-07-05-llm"))
        assert len(result.resolved) == 5
        assert all(q.id.startswith("rq") for q in result.resolved)


class TestRoundEvalSkipsHeldOut:
    def test_round_scores_all_questions(self, tmp_path: Path, monkeypatch: object) -> None:
        import eval as eval_mod

        round_qs = _make_round_question_set()
        resolutions = _make_resolutions()

        monkeypatch.setattr(eval_mod, "fetch_question_set", lambda f: round_qs)  # type: ignore[attr-defined]
        monkeypatch.setattr(eval_mod, "fetch_all_resolutions", lambda: resolutions)  # type: ignore[attr-defined]
        monkeypatch.setattr(eval_mod, "RESULTS_DIR", tmp_path / "results")  # type: ignore[attr-defined]
        monkeypatch.setattr(eval_mod, "CACHE_DIR", tmp_path / "cache")  # type: ignore[attr-defined]

        result = asyncio.run(run_eval(_dummy_forecaster, n_held_out=2, round_name="2026-07-05-llm"))
        assert result.scoring.n_dataset == 5
        assert result.scoring.n_missing == 0


class TestRoundMetadataInResult:
    def test_round_name_in_saved_metadata(self, tmp_path: Path) -> None:
        result = ScoringResult(
            dataset_brier=0.25, dataset_index=50.0,
            market_brier=0.0, market_index=0.0,
            overall_brier=0.25, overall_index=50.0,
            n_dataset=5, n_market=0, n_missing=0,
        )
        import eval as eval_mod
        original_dir = eval_mod.RESULTS_DIR
        eval_mod.RESULTS_DIR = tmp_path
        try:
            path = save_result(
                result, {"rq0": 0.5}, {"rq0": 1}, "test_model",
                ["2026-07-05"], 0, round_name="2026-07-05-llm",
            )
            data = json.loads(path.read_text())
            assert data["metadata"]["round"] == "2026-07-05-llm"
        finally:
            eval_mod.RESULTS_DIR = original_dir

    def test_no_round_key_without_flag(self, tmp_path: Path) -> None:
        result = ScoringResult(
            dataset_brier=0.25, dataset_index=50.0,
            market_brier=0.0, market_index=0.0,
            overall_brier=0.25, overall_index=50.0,
            n_dataset=5, n_market=0, n_missing=0,
        )
        import eval as eval_mod
        original_dir = eval_mod.RESULTS_DIR
        eval_mod.RESULTS_DIR = tmp_path
        try:
            path = save_result(
                result, {"rq0": 0.5}, {"rq0": 1}, "test_model",
                ["2026-07-05"], 2,
            )
            data = json.loads(path.read_text())
            assert "round" not in data["metadata"]
        finally:
            eval_mod.RESULTS_DIR = original_dir


class TestRoundNameInFilename:
    def test_filename_includes_round_name(self, tmp_path: Path) -> None:
        result = ScoringResult(
            dataset_brier=0.25, dataset_index=50.0,
            market_brier=0.0, market_index=0.0,
            overall_brier=0.25, overall_index=50.0,
            n_dataset=5, n_market=0, n_missing=0,
        )
        import eval as eval_mod
        original_dir = eval_mod.RESULTS_DIR
        eval_mod.RESULTS_DIR = tmp_path
        try:
            path = save_result(
                result, {"rq0": 0.5}, {"rq0": 1}, "test_model",
                ["2026-07-05"], 0, round_name="2026-07-05-llm",
            )
            assert "2026-07-05-llm" in path.name
            assert path.name.endswith(".json")
        finally:
            eval_mod.RESULTS_DIR = original_dir

    def test_filename_without_round(self, tmp_path: Path) -> None:
        result = ScoringResult(
            dataset_brier=0.25, dataset_index=50.0,
            market_brier=0.0, market_index=0.0,
            overall_brier=0.25, overall_index=50.0,
            n_dataset=5, n_market=0, n_missing=0,
        )
        import eval as eval_mod
        original_dir = eval_mod.RESULTS_DIR
        eval_mod.RESULTS_DIR = tmp_path
        try:
            path = save_result(
                result, {"rq0": 0.5}, {"rq0": 1}, "test_model",
                ["2026-07-05"], 2,
            )
            assert "test_model" in path.name
            assert "2026-07-05-llm" not in path.name
        finally:
            eval_mod.RESULTS_DIR = original_dir


class TestListRounds:
    def test_list_rounds_returns_names_and_counts(self) -> None:
        mock_files = ["2026-07-05-llm.json", "2026-06-21-llm.json"]
        mock_qs = {
            "2026-07-05-llm.json": QuestionSet(
                forecast_due_date="2026-07-05", questions=[
                    Question(id=f"q{i}", source="acled", question=f"Q{i}")
                    for i in range(500)
                ],
            ),
            "2026-06-21-llm.json": QuestionSet(
                forecast_due_date="2026-06-21", questions=[
                    Question(id=f"q{i}", source="acled", question=f"Q{i}")
                    for i in range(492)
                ],
            ),
        }

        with patch("eval.list_question_set_files", return_value=mock_files), \
             patch("eval.fetch_question_set", side_effect=lambda f: mock_qs[f]):
            rounds = list_rounds()

        assert len(rounds) == 2
        names = [r[0] for r in rounds]
        assert "2026-07-05-llm" in names
        assert "2026-06-21-llm" in names
        counts = {r[0]: r[1] for r in rounds}
        assert counts["2026-07-05-llm"] == 500
        assert counts["2026-06-21-llm"] == 492


class TestRoundWithZeroResolvedQuestions:
    def test_raises_on_no_resolved_questions(self, tmp_path: Path, monkeypatch: object) -> None:
        import eval as eval_mod

        round_qs = _make_round_question_set()

        monkeypatch.setattr(eval_mod, "fetch_question_set", lambda f: round_qs)  # type: ignore[attr-defined]
        monkeypatch.setattr(eval_mod, "fetch_all_resolutions", lambda: {})  # type: ignore[attr-defined]
        monkeypatch.setattr(eval_mod, "RESULTS_DIR", tmp_path / "results")  # type: ignore[attr-defined]
        monkeypatch.setattr(eval_mod, "CACHE_DIR", tmp_path / "cache")  # type: ignore[attr-defined]

        with pytest.raises(ValueError, match="No resolved questions"):
            asyncio.run(run_eval(_dummy_forecaster, round_name="2026-07-05-llm"))


class TestRoundWithNonExistentName:
    def test_raises_on_bad_round_name(self, tmp_path: Path, monkeypatch: object) -> None:
        import eval as eval_mod
        from requests.exceptions import HTTPError

        def _raise_not_found(f: str) -> QuestionSet:
            raise HTTPError("404 Not Found")

        monkeypatch.setattr(eval_mod, "fetch_question_set", _raise_not_found)  # type: ignore[attr-defined]
        monkeypatch.setattr(eval_mod, "RESULTS_DIR", tmp_path / "results")  # type: ignore[attr-defined]
        monkeypatch.setattr(eval_mod, "CACHE_DIR", tmp_path / "cache")  # type: ignore[attr-defined]

        with pytest.raises(HTTPError):
            asyncio.run(run_eval(_dummy_forecaster, round_name="9999-99-99-llm"))


class TestRoundWithMixedSources:
    def test_mixed_market_and_dataset_questions(self, tmp_path: Path, monkeypatch: object) -> None:
        import eval as eval_mod

        mixed_qs = QuestionSet(
            forecast_due_date="2026-07-05",
            question_set="2026-07-05-llm",
            questions=[
                Question(id="d1", source="acled", question="Dataset Q1"),
                Question(id="d2", source="acled", question="Dataset Q2"),
                Question(id="m1", source="metaculus", question="Market Q1"),
                Question(id="m2", source="polymarket", question="Market Q2"),
            ],
        )
        resolutions = {
            "d1": Resolution(id="d1", outcome=1, resolution_date="2026-07-19"),
            "d2": Resolution(id="d2", outcome=0, resolution_date="2026-07-19"),
            "m1": Resolution(id="m1", outcome=1, resolution_date="2026-07-19"),
            "m2": Resolution(id="m2", outcome=0, resolution_date="2026-07-19"),
        }

        monkeypatch.setattr(eval_mod, "fetch_question_set", lambda f: mixed_qs)  # type: ignore[attr-defined]
        monkeypatch.setattr(eval_mod, "fetch_all_resolutions", lambda: resolutions)  # type: ignore[attr-defined]
        monkeypatch.setattr(eval_mod, "RESULTS_DIR", tmp_path / "results")  # type: ignore[attr-defined]
        monkeypatch.setattr(eval_mod, "CACHE_DIR", tmp_path / "cache")  # type: ignore[attr-defined]

        result = asyncio.run(run_eval(_dummy_forecaster, round_name="2026-07-05-llm"))
        assert result.scoring.n_dataset == 2
        assert result.scoring.n_market == 2
        assert result.scoring.overall_brier == pytest.approx(0.25)


class TestByRoundComparison:
    def test_compare_by_round_groups_results(self, tmp_path: Path, capsys: object) -> None:
        from analyze import compare_by_round

        payload = {
            "timestamp": "20260705T120000Z",
            "model_slug": "baseline",
            "scoring_result": {
                "dataset_brier": 0.189, "dataset_index": 56.5,
                "market_brier": 0.0, "market_index": 0.0,
                "overall_brier": 0.189, "overall_index": 56.5,
                "n_dataset": 487, "n_market": 0, "n_missing": 0,
                "difficulty_adjusted": False,
            },
            "forecasts": {},
            "outcomes": {},
            "metadata": {
                "n_questions": 487,
                "n_held_out": 0,
                "question_sets_used": ["2026-07-05"],
                "round": "2026-07-05-llm",
            },
        }
        (tmp_path / "result1.json").write_text(json.dumps(payload))

        compare_by_round(tmp_path)
        captured = capsys.readouterr()  # type: ignore[union-attr]
        assert "2026-07-05-llm" in captured.out
        assert "baseline" in captured.out

    def test_compare_by_round_empty_without_round_results(self, tmp_path: Path, capsys: object) -> None:
        from analyze import compare_by_round

        payload = {
            "timestamp": "20260705T120000Z",
            "model_slug": "default",
            "scoring_result": {
                "dataset_brier": 0.25, "dataset_index": 50.0,
                "market_brier": 0.0, "market_index": 0.0,
                "overall_brier": 0.25, "overall_index": 50.0,
                "n_dataset": 5, "n_market": 0, "n_missing": 0,
                "difficulty_adjusted": False,
            },
            "forecasts": {},
            "outcomes": {},
            "metadata": {"n_questions": 5, "n_held_out": 2, "question_sets_used": []},
        }
        (tmp_path / "result1.json").write_text(json.dumps(payload))

        compare_by_round(tmp_path)
        captured = capsys.readouterr()  # type: ignore[union-attr]
        assert "No per-round results" in captured.out
