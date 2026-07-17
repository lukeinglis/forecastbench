"""Tests for fetch_data.py: model parsing, resolution mapping, caching, network failures."""

from __future__ import annotations

import json

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import requests

from fetch_data import (
    Question,
    QuestionSet,
    Resolution,
    _cache_path,
    _fetch_json,
    fetch_question_set,
    fetch_resolution,
    join_resolved_questions,
    refresh_cache,
)


# ---------------------------------------------------------------------------
# Resolution.model_validate — resolved_to → outcome mapping
# ---------------------------------------------------------------------------


class TestResolutionModelValidate:
    def test_resolved_to_rounds_up(self) -> None:
        r = Resolution.model_validate({"id": "q1", "resolved_to": 0.7})
        assert r.outcome == 1

    def test_resolved_to_rounds_down(self) -> None:
        r = Resolution.model_validate({"id": "q2", "resolved_to": 0.3})
        assert r.outcome == 0

    def test_resolved_to_exact_one(self) -> None:
        r = Resolution.model_validate({"id": "q3", "resolved_to": 1.0})
        assert r.outcome == 1

    def test_resolved_to_exact_zero(self) -> None:
        r = Resolution.model_validate({"id": "q4", "resolved_to": 0.0})
        assert r.outcome == 0

    def test_direct_outcome_field(self) -> None:
        r = Resolution.model_validate({"id": "q5", "outcome": 1})
        assert r.outcome == 1

    def test_resolved_to_half_does_not_crash(self) -> None:
        r = Resolution.model_validate({"id": "q6", "resolved_to": 0.5})
        assert r.outcome in (0, 1)

    def test_outcome_takes_precedence_over_resolved_to(self) -> None:
        r = Resolution.model_validate(
            {"id": "q7", "outcome": 0, "resolved_to": 0.9}
        )
        assert r.outcome == 0

    def test_resolution_date_preserved(self) -> None:
        r = Resolution.model_validate(
            {"id": "q8", "resolved_to": 1.0, "resolution_date": "2024-06-01"}
        )
        assert r.resolution_date == "2024-06-01"
        assert r.outcome == 1

    def test_resolved_to_none(self) -> None:
        r = Resolution.model_validate({"id": "q9", "resolved_to": None})
        assert r.outcome is None

    def test_no_outcome_and_no_resolved_to(self) -> None:
        r = Resolution.model_validate({"id": "q10"})
        assert r.outcome is None


# ---------------------------------------------------------------------------
# Resolution._coerce_id — list IDs joined with '|'
# ---------------------------------------------------------------------------


class TestResolutionCoerceId:
    def test_list_id_joined(self) -> None:
        r = Resolution.model_validate({"id": ["abc", "def"], "outcome": 1})
        assert r.id == "abc|def"

    def test_string_id_passthrough(self) -> None:
        r = Resolution.model_validate({"id": "simple", "outcome": 0})
        assert r.id == "simple"

    def test_single_element_list(self) -> None:
        r = Resolution.model_validate({"id": ["only"], "outcome": 1})
        assert r.id == "only"


# ---------------------------------------------------------------------------
# Question._coerce_id
# ---------------------------------------------------------------------------


class TestQuestionCoerceId:
    def test_list_id(self) -> None:
        q = Question(id=["abc", "def"], source="acled", question="Test")  # type: ignore[arg-type]
        assert q.id == "abc|def"

    def test_string_id(self) -> None:
        q = Question(id="simple_string", source="acled", question="Test")
        assert q.id == "simple_string"

    def test_single_element_list(self) -> None:
        q = Question(id=["single"], source="acled", question="Test")  # type: ignore[arg-type]
        assert q.id == "single"


# ---------------------------------------------------------------------------
# Question._coerce_combination_of
# ---------------------------------------------------------------------------


class TestQuestionCoerceCombinationOf:
    def test_dict_in_list(self) -> None:
        q = Question(
            id="q1",
            source="acled",
            question="Test",
            combination_of=[{"id": "q1"}, {"id": "q2"}],  # type: ignore[list-item]
        )
        assert q.combination_of == ["q1", "q2"]

    def test_string_list(self) -> None:
        q = Question(
            id="q1",
            source="acled",
            question="Test",
            combination_of=["q1", "q2"],
        )
        assert q.combination_of == ["q1", "q2"]

    def test_none(self) -> None:
        q = Question(id="q1", source="acled", question="Test", combination_of=None)
        assert q.combination_of is None

    def test_na_string(self) -> None:
        q = Question(
            id="q1",
            source="acled",
            question="Test",
            combination_of="N/A",  # type: ignore[arg-type]
        )
        assert q.combination_of is None


# ---------------------------------------------------------------------------
# fetch_resolution — three response shapes
# ---------------------------------------------------------------------------


class TestFetchResolutionShapes:
    @patch("fetch_data._fetch_json")
    def test_list_shape(self, mock_fetch: MagicMock) -> None:
        mock_fetch.return_value = [
            {"id": "r1", "outcome": 1},
            {"id": "r2", "outcome": 0},
        ]
        result = fetch_resolution("test.json")
        assert len(result) == 2
        assert result[0].id == "r1"
        assert result[1].outcome == 0

    @patch("fetch_data._fetch_json")
    def test_dict_with_resolutions_key(self, mock_fetch: MagicMock) -> None:
        mock_fetch.return_value = {
            "resolutions": [
                {"id": "r1", "resolved_to": 0.8},
                {"id": "r2", "resolved_to": 0.2},
            ]
        }
        result = fetch_resolution("test.json")
        assert len(result) == 2
        assert result[0].outcome == 1
        assert result[1].outcome == 0

    @patch("fetch_data._fetch_json")
    def test_single_object(self, mock_fetch: MagicMock) -> None:
        mock_fetch.return_value = {"id": "r1", "outcome": 1}
        result = fetch_resolution("test.json")
        assert len(result) == 1
        assert result[0].id == "r1"


# ---------------------------------------------------------------------------
# _fetch_json caching
# ---------------------------------------------------------------------------


class TestFetchJsonCaching:
    def test_first_call_fetches_and_caches(self, tmp_path: Path) -> None:
        payload = {"key": "value"}

        mock_resp = MagicMock()
        mock_resp.json.return_value = payload
        mock_resp.raise_for_status = MagicMock()

        with (
            patch("fetch_data.CACHE_DIR", tmp_path),
            patch("fetch_data.requests.get", return_value=mock_resp) as mock_get,
        ):
            result = _fetch_json("https://example.com/data.json", "test_cache.json")

        assert result == payload
        mock_get.assert_called_once()
        assert (tmp_path / "test_cache.json").exists()

    def test_second_call_uses_cache(self, tmp_path: Path) -> None:
        payload = {"cached": True}
        cache_file = tmp_path / "cached.json"
        cache_file.write_text(json.dumps(payload))

        with (
            patch("fetch_data.CACHE_DIR", tmp_path),
            patch("fetch_data.requests.get") as mock_get,
        ):
            result = _fetch_json("https://example.com/data.json", "cached.json")

        assert result == payload
        mock_get.assert_not_called()

    def test_corrupted_cache_raises(self, tmp_path: Path) -> None:
        cache_file = tmp_path / "bad.json"
        cache_file.write_text("NOT VALID JSON {{{")

        with patch("fetch_data.CACHE_DIR", tmp_path):
            with pytest.raises(json.JSONDecodeError):
                _fetch_json("https://example.com/data.json", "bad.json")

    def test_slash_in_cache_key_replaced(self) -> None:
        result = _cache_path("some/nested/key.json")
        assert "/" not in result.name


# ---------------------------------------------------------------------------
# Network failure tests
# ---------------------------------------------------------------------------


class TestNetworkFailures:
    @patch("fetch_data._fetch_json")
    def test_connection_error_propagates(self, mock_fetch: MagicMock) -> None:
        mock_fetch.side_effect = requests.exceptions.ConnectionError("Connection refused")
        with pytest.raises(requests.exceptions.ConnectionError):
            fetch_question_set("test.json")

    @patch("fetch_data._fetch_json")
    def test_timeout_propagates(self, mock_fetch: MagicMock) -> None:
        mock_fetch.side_effect = requests.exceptions.Timeout("Read timed out")
        with pytest.raises(requests.exceptions.Timeout):
            fetch_question_set("test.json")

    @patch("fetch_data._fetch_json")
    def test_http_404_propagates(self, mock_fetch: MagicMock) -> None:
        mock_fetch.side_effect = requests.exceptions.HTTPError("404 Not Found")
        with pytest.raises(requests.exceptions.HTTPError):
            fetch_resolution("missing.json")

    @patch("fetch_data._fetch_json")
    def test_http_500_propagates(self, mock_fetch: MagicMock) -> None:
        mock_fetch.side_effect = requests.exceptions.HTTPError("500 Internal Server Error")
        with pytest.raises(requests.exceptions.HTTPError):
            fetch_resolution("broken.json")

    def test_fetch_json_connection_error(self, tmp_path: Path) -> None:
        with (
            patch("fetch_data.CACHE_DIR", tmp_path),
            patch(
                "fetch_data.requests.get",
                side_effect=requests.exceptions.ConnectionError("refused"),
            ),
        ):
            with pytest.raises(requests.exceptions.ConnectionError):
                _fetch_json("https://example.com/fail", "fail.json")

    def test_fetch_json_timeout(self, tmp_path: Path) -> None:
        with (
            patch("fetch_data.CACHE_DIR", tmp_path),
            patch(
                "fetch_data.requests.get",
                side_effect=requests.exceptions.Timeout("timed out"),
            ),
        ):
            with pytest.raises(requests.exceptions.Timeout):
                _fetch_json("https://example.com/slow", "slow.json")


# ---------------------------------------------------------------------------
# fetch_question_set — full parse
# ---------------------------------------------------------------------------


class TestFetchQuestionSet:
    @patch("fetch_data._fetch_json")
    def test_parses_question_set(self, mock_fetch: MagicMock) -> None:
        mock_fetch.return_value = {
            "forecast_due_date": "2024-06-01",
            "question_set": "round_1",
            "questions": [
                {
                    "id": "q1",
                    "source": "metaculus",
                    "question": "Will X happen?",
                    "background": "Context here",
                },
                {
                    "id": ["a", "b"],
                    "source": "acled",
                    "question": "Combo question",
                    "combination_of": [{"id": "a"}, {"id": "b"}],
                },
            ],
        }
        qs = fetch_question_set("round1.json")
        assert qs.forecast_due_date == "2024-06-01"
        assert len(qs.questions) == 2
        assert qs.questions[1].id == "a|b"
        assert qs.questions[1].combination_of == ["a", "b"]


# ---------------------------------------------------------------------------
# compare_results
# ---------------------------------------------------------------------------


class TestCompareResults:
    def _make_result(self, model: str, brier: float) -> dict:
        return {
            "timestamp": "2024-01-01T00:00:00",
            "model_slug": model,
            "scoring_result": {
                "overall_brier": brier,
                "overall_index": (1.0 - brier**0.5) * 100.0,
                "dataset_brier": brier,
                "market_brier": brier,
                "n_dataset": 10,
                "n_market": 5,
                "n_missing": 0,
            },
        }

    def test_compare_results_with_files(self, tmp_path: Path) -> None:
        from analyze import compare_results

        (tmp_path / "a.json").write_text(json.dumps(self._make_result("model_a", 0.20)))
        (tmp_path / "b.json").write_text(json.dumps(self._make_result("model_b", 0.30)))
        compare_results(tmp_path)

    def test_compare_results_empty_dir(self, tmp_path: Path) -> None:
        from analyze import compare_results

        compare_results(tmp_path)

    def test_compare_results_no_dir(self, tmp_path: Path) -> None:
        from analyze import compare_results

        compare_results(tmp_path / "nonexistent")

    def test_compare_results_invalid_json(self, tmp_path: Path) -> None:
        from analyze import compare_results

        (tmp_path / "bad.json").write_text("NOT JSON")
        (tmp_path / "good.json").write_text(json.dumps(self._make_result("ok", 0.25)))
        compare_results(tmp_path)

    def test_compare_results_missing_keys(self, tmp_path: Path) -> None:
        from analyze import compare_results

        (tmp_path / "partial.json").write_text(json.dumps({"model_slug": "x"}))
        compare_results(tmp_path)


# ---------------------------------------------------------------------------
# TTL-based caching
# ---------------------------------------------------------------------------


class TestFetchJsonCacheForever:
    def test_old_cache_still_used(self, tmp_path: Path) -> None:
        payload = {"forever": True}
        cache_file = tmp_path / "forever.json"
        cache_file.write_text(json.dumps(payload))

        with (
            patch("fetch_data.CACHE_DIR", tmp_path),
            patch("fetch_data.requests.get") as mock_get,
        ):
            result = _fetch_json("https://example.com/data.json", "forever.json")

        assert result == payload
        mock_get.assert_not_called()

    def test_refresh_cache_forces_refetch(self, tmp_path: Path) -> None:
        from fetch_data import refresh_cache

        (tmp_path / "question_sets_listing.json").write_text(json.dumps({"old": True}))
        (tmp_path / "resolution_sets_listing.json").write_text(json.dumps([]))
        (tmp_path / "res_round1.json").write_text(json.dumps([]))
        (tmp_path / "lb_baseline.csv").write_text("header\n")
        (tmp_path / "qs_immutable.json").write_text(json.dumps({"keep": True}))

        with patch("fetch_data.CACHE_DIR", tmp_path):
            refresh_cache()

        assert not (tmp_path / "question_sets_listing.json").exists()
        assert not (tmp_path / "resolution_sets_listing.json").exists()
        assert not (tmp_path / "res_round1.json").exists()
        assert not (tmp_path / "lb_baseline.csv").exists()
        assert (tmp_path / "qs_immutable.json").exists()


# ---------------------------------------------------------------------------
# refresh_cache
# ---------------------------------------------------------------------------


class TestRefreshCache:
    def test_deletes_listings_and_resolutions(self, tmp_path: Path) -> None:
        (tmp_path / "question_sets_listing.json").write_text("{}")
        (tmp_path / "resolution_sets_listing.json").write_text("{}")
        (tmp_path / "res_2024-01-01.json").write_text("{}")
        (tmp_path / "res_2024-02-01.json").write_text("{}")
        (tmp_path / "lb_baseline.csv").write_text("")
        # Should NOT be deleted
        (tmp_path / "qs_round1.json").write_text("{}")

        with patch("fetch_data.CACHE_DIR", tmp_path):
            refresh_cache()

        assert not (tmp_path / "question_sets_listing.json").exists()
        assert not (tmp_path / "resolution_sets_listing.json").exists()
        assert not (tmp_path / "res_2024-01-01.json").exists()
        assert not (tmp_path / "res_2024-02-01.json").exists()
        assert not (tmp_path / "lb_baseline.csv").exists()
        assert (tmp_path / "qs_round1.json").exists()

    def test_noop_when_cache_dir_missing(self, tmp_path: Path) -> None:
        with patch("fetch_data.CACHE_DIR", tmp_path / "nonexistent"):
            refresh_cache()

    def test_preserves_question_set_caches(self, tmp_path: Path) -> None:
        (tmp_path / "qs_2024-01-01-llm.json").write_text("{}")
        (tmp_path / "qs_2024-02-01-llm.json").write_text("{}")

        with patch("fetch_data.CACHE_DIR", tmp_path):
            refresh_cache()

        assert (tmp_path / "qs_2024-01-01-llm.json").exists()
        assert (tmp_path / "qs_2024-02-01-llm.json").exists()


# ---------------------------------------------------------------------------
# Resolution.resolved field filtering in join_resolved_questions
# ---------------------------------------------------------------------------


class TestResolvedFieldFiltering:
    def _make_qs(self, question_id: str = "q1") -> QuestionSet:
        return QuestionSet(
            forecast_due_date="2024-06-01",
            question_set="round_1",
            questions=[
                Question(id=question_id, source="metaculus", question="Will X?")
            ],
        )

    def test_resolved_false_excluded(self) -> None:
        qs = self._make_qs()
        resolutions = {
            "q1": Resolution(id="q1", outcome=1, resolved=False),
        }
        result = join_resolved_questions([qs], resolutions)
        assert len(result) == 0

    def test_resolved_true_included(self) -> None:
        qs = self._make_qs()
        resolutions = {
            "q1": Resolution(id="q1", outcome=1, resolved=True),
        }
        result = join_resolved_questions([qs], resolutions)
        assert len(result) == 1
        assert result[0].id == "q1"

    def test_resolved_none_included(self) -> None:
        qs = self._make_qs()
        resolutions = {
            "q1": Resolution(id="q1", outcome=0, resolved=None),
        }
        result = join_resolved_questions([qs], resolutions)
        assert len(result) == 1

    def test_resolved_missing_included(self) -> None:
        qs = self._make_qs()
        resolutions = {
            "q1": Resolution(id="q1", outcome=1),
        }
        result = join_resolved_questions([qs], resolutions)
        assert len(result) == 1

    def test_resolved_field_parsed_from_dict(self) -> None:
        r = Resolution.model_validate({"id": "q1", "resolved_to": 0.8, "resolved": False})
        assert r.resolved is False
        assert r.outcome == 1


# ---------------------------------------------------------------------------
# list_question_set_files excludes latest-llm.json
# ---------------------------------------------------------------------------


class TestListQuestionSetFilesFiltering:
    @patch("fetch_data._fetch_json")
    def test_excludes_latest_llm_json(self, mock_fetch: MagicMock) -> None:
        mock_fetch.return_value = [
            {"name": "2026-06-01-llm.json"},
            {"name": "latest-llm.json"},
            {"name": "2026-07-05-llm.json"},
        ]
        from fetch_data import list_question_set_files

        result = list_question_set_files()
        assert "latest-llm.json" not in result
        assert len(result) == 2
        assert "2026-06-01-llm.json" in result
        assert "2026-07-05-llm.json" in result


# ---------------------------------------------------------------------------
# get_latest_round
# ---------------------------------------------------------------------------


class TestGetLatestRound:
    @patch("fetch_data._fetch_text")
    def test_returns_round_name(self, mock_fetch: MagicMock) -> None:
        mock_fetch.return_value = "2026-07-05-llm.json\n"
        from fetch_data import get_latest_round

        result = get_latest_round()
        assert result == "2026-07-05-llm"

    @patch("fetch_data._fetch_text")
    def test_strips_whitespace(self, mock_fetch: MagicMock) -> None:
        mock_fetch.return_value = "  2026-07-05-llm.json  \n"
        from fetch_data import get_latest_round

        result = get_latest_round()
        assert result == "2026-07-05-llm"
