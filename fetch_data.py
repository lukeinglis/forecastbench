"""Fetch ForecastBench question sets and resolutions from GitHub."""

from __future__ import annotations

import json

from pathlib import Path
from typing import Any

import requests
from pydantic import BaseModel, field_validator

from logging_config import get_logger

logger = get_logger("fetch_data")


REPO_OWNER = "forecastingresearch"
REPO_NAME = "forecastbench-datasets"
RAW_BASE = f"https://raw.githubusercontent.com/{REPO_OWNER}/{REPO_NAME}/main/datasets"
API_BASE = f"https://api.github.com/repos/{REPO_OWNER}/{REPO_NAME}/contents/datasets"
LEADERBOARD_BASE = f"https://raw.githubusercontent.com/{REPO_OWNER}/{REPO_NAME}/main/leaderboards/csv"
CACHE_DIR = Path(".cache")

LEADERBOARD_NAMES = frozenset({"baseline", "tournament", "dataset", "preliminary"})

MARKET_SOURCES = frozenset({"metaculus", "polymarket", "manifold", "infer"})


class Question(BaseModel):
    id: str
    source: str
    question: str
    background: str = ""
    resolution_criteria: str = ""
    freeze_datetime: str | None = None
    freeze_datetime_value: float | None = None
    resolution_dates: Any = None  # "N/A", null, or list of date strings
    url: str | None = None
    combination_of: list[str] | None = None
    source_intro: str | None = None
    freeze_datetime_value_explanation: str | None = None
    market_info_open_datetime: str | None = None
    market_info_close_datetime: str | None = None
    market_info_resolution_criteria: str | None = None

    @field_validator("id", mode="before")
    @classmethod
    def _coerce_id(cls, v: Any) -> str:
        if isinstance(v, list):
            return "|".join(str(x) for x in v)
        return str(v)

    @field_validator("freeze_datetime_value", mode="before")
    @classmethod
    def _coerce_freeze_value(cls, v: Any) -> float | None:
        if v is None:
            return None
        try:
            return float(v)
        except (ValueError, TypeError):
            return None

    @field_validator("combination_of", mode="before")
    @classmethod
    def _coerce_combination_of(cls, v: Any) -> list[str] | None:
        if v is None or (isinstance(v, str) and v.upper() == "N/A"):
            return None
        if isinstance(v, list):
            return [x["id"] if isinstance(x, dict) else str(x) for x in v]
        return list(v) if isinstance(v, (list, tuple)) else None


class QuestionSet(BaseModel):
    forecast_due_date: str
    question_set: str = ""
    questions: list[Question]


class Resolution(BaseModel):
    id: str
    outcome: int | None = None
    resolution_date: str | None = None
    resolved: bool | None = None

    @field_validator("id", mode="before")
    @classmethod
    def _coerce_id(cls, v: Any) -> str:
        if isinstance(v, list):
            return "|".join(str(x) for x in v)
        return str(v)

    @classmethod
    def model_validate(cls, obj: Any, **kwargs: Any) -> Resolution:
        if isinstance(obj, dict) and "outcome" not in obj and "resolved_to" in obj:
            obj = dict(obj)
            val = obj.pop("resolved_to", None)
            obj["outcome"] = round(val) if val is not None else None
        return super().model_validate(obj, **kwargs)


class ResolvedQuestion(BaseModel):
    id: str
    source: str
    question: str
    background: str = ""
    resolution_criteria: str = ""
    freeze_datetime: str | None = None
    freeze_datetime_value: float | None = None
    resolution_dates: Any = None
    url: str | None = None
    combination_of: list[str] | None = None
    source_intro: str | None = None
    freeze_datetime_value_explanation: str | None = None
    market_info_open_datetime: str | None = None
    market_info_close_datetime: str | None = None
    market_info_resolution_criteria: str | None = None
    outcome: int
    resolution_date: str | None = None
    forecast_due_date: str = ""
    question_set: str = ""


def _ensure_cache_dir() -> None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)


def _cache_path(filename: str) -> Path:
    return CACHE_DIR / filename.replace("/", "_")


def _fetch_json(url: str, cache_key: str) -> Any:
    _ensure_cache_dir()
    cached = _cache_path(cache_key)
    if cached.exists():
        logger.debug("cache_hit", cache_key=cache_key)
        return json.loads(cached.read_text())
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()
    data = resp.json()
    cached.write_text(json.dumps(data))
    return data


def list_question_set_files() -> list[str]:
    """List available question set JSON filenames from the GitHub repo."""
    data = _fetch_json(f"{API_BASE}/question_sets", "question_sets_listing.json")
    return [
        item["name"]
        for item in data
        if item["name"].endswith(".json") and item["name"] != "latest-llm.json"
    ]


def get_latest_round() -> str:
    """Get the name of the current/latest round from the ForecastBench repo."""
    url = f"{RAW_BASE}/question_sets/latest-llm.json"
    text = _fetch_text(url, "latest_round.txt")
    return text.strip().replace(".json", "")


def list_resolution_files() -> list[str]:
    """List available resolution JSON filenames from the GitHub repo."""
    data = _fetch_json(f"{API_BASE}/resolution_sets", "resolution_sets_listing.json")
    return [item["name"] for item in data if item["name"].endswith(".json")]


def fetch_question_set(filename: str) -> QuestionSet:
    """Fetch and parse a single question set file."""
    url = f"{RAW_BASE}/question_sets/{filename}"
    data = _fetch_json(url, f"qs_{filename}")
    return QuestionSet.model_validate(data)


def fetch_resolution(filename: str) -> list[Resolution]:
    """Fetch and parse a single resolution file."""
    url = f"{RAW_BASE}/resolution_sets/{filename}"
    data = _fetch_json(url, f"res_{filename}")
    if isinstance(data, list):
        return [Resolution.model_validate(r) for r in data]
    if isinstance(data, dict) and "resolutions" in data:
        return [Resolution.model_validate(r) for r in data["resolutions"]]
    return [Resolution.model_validate(data)]


def fetch_all_question_sets() -> list[QuestionSet]:
    """Fetch all available question sets."""
    filenames = list_question_set_files()
    result = []
    for f in filenames:
        try:
            qs = fetch_question_set(f)
            result.append(qs)
        except Exception as e:
            logger.warning("fetch_question_set_failed", filename=f, error=str(e))
    return result


def fetch_all_resolutions() -> dict[str, Resolution]:
    """Fetch all resolutions, returning a dict keyed by question id."""
    filenames = list_resolution_files()
    resolutions: dict[str, Resolution] = {}
    for f in filenames:
        try:
            res_list = fetch_resolution(f)
            for r in res_list:
                resolutions[r.id] = r
        except Exception as e:
            logger.warning("fetch_resolution_failed", filename=f, error=str(e))
    return resolutions


def join_resolved_questions(
    question_sets: list[QuestionSet],
    resolutions: dict[str, Resolution],
) -> list[ResolvedQuestion]:
    """Join questions with their resolutions, returning only resolved questions."""
    resolved = []
    for qs in question_sets:
        for q in qs.questions:
            if q.id in resolutions and resolutions[q.id].outcome is not None and getattr(resolutions[q.id], "resolved", None) is not False:
                r = resolutions[q.id]
                resolved.append(
                    ResolvedQuestion(
                        id=q.id,
                        source=q.source,
                        question=q.question,
                        background=q.background,
                        resolution_criteria=q.resolution_criteria,
                        freeze_datetime=q.freeze_datetime,
                        freeze_datetime_value=q.freeze_datetime_value,
                        resolution_dates=q.resolution_dates,
                        url=q.url,
                        combination_of=q.combination_of,
                        source_intro=q.source_intro,
                        freeze_datetime_value_explanation=q.freeze_datetime_value_explanation,
                        market_info_open_datetime=q.market_info_open_datetime,
                        market_info_close_datetime=q.market_info_close_datetime,
                        market_info_resolution_criteria=q.market_info_resolution_criteria,
                        outcome=r.outcome,  # type: ignore[arg-type]
                        resolution_date=r.resolution_date,
                        forecast_due_date=qs.forecast_due_date,
                        question_set=qs.question_set,
                    )
                )
    return resolved


def _fetch_text(url: str, cache_key: str) -> str:
    _ensure_cache_dir()
    cached = _cache_path(cache_key)
    if cached.exists():
        logger.debug("cache_hit", cache_key=cache_key)
        return cached.read_text()
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()
    text = resp.text
    cached.write_text(text)
    return text


def fetch_leaderboard(name: str = "baseline") -> list[dict[str, str]]:
    """Fetch a leaderboard CSV and return as list of dicts.

    Supported names: baseline, tournament, dataset, preliminary.
    """
    import csv
    import io

    if name not in LEADERBOARD_NAMES:
        raise ValueError(f"Unknown leaderboard {name!r}, expected one of {sorted(LEADERBOARD_NAMES)}")
    url = f"{LEADERBOARD_BASE}/leaderboard_{name}.csv"
    text = _fetch_text(url, f"lb_{name}.csv")
    reader = csv.DictReader(io.StringIO(text))
    rows: list[dict[str, str]] = []
    for row in reader:
        rows.append(dict(row))
    logger.info("leaderboard_fetched", name=name, n_entries=len(rows))
    return rows


def fetch_superforecaster_forecasts() -> list[dict[str, object]]:
    """Fetch individual superforecaster forecasts from the July 2024 round.

    Returns list of forecast entries, each with: id, source, forecast, reasoning, searches, user_id.
    """
    url = (
        "https://media.githubusercontent.com/media/forecastingresearch/"
        "forecastbench-datasets/main/datasets/forecast_sets/"
        "2024-07-21/2024-07-21.ForecastBench.human_super_individual.json"
    )
    data = _fetch_json(url, "superforecaster_individual.json")
    result: list[dict[str, object]] = data.get("forecasts", [])
    return result


def superforecaster_medians(forecasts: list[dict[str, object]]) -> dict[str, float]:
    """Compute median forecast per question from individual superforecaster entries."""
    from statistics import median

    by_question: dict[str, list[float]] = {}
    for entry in forecasts:
        qid = str(entry["id"])
        prob: Any = entry.get("forecast")
        if prob is not None:
            by_question.setdefault(qid, []).append(float(prob))
    return {qid: median(probs) for qid, probs in by_question.items() if probs}


def refresh_cache() -> None:
    """Delete volatile cache files so next fetch pulls fresh data.

    Removes listings, resolution caches, and leaderboard CSVs.
    Question set caches (qs_*) are kept — their content is immutable.
    """
    if not CACHE_DIR.exists():
        return
    patterns = ["question_sets_listing.json", "resolution_sets_listing.json"]
    for name in patterns:
        path = CACHE_DIR / name
        if path.exists():
            path.unlink()
            logger.info("cache_deleted", file=name)
    for path in CACHE_DIR.glob("res_*"):
        path.unlink()
        logger.info("cache_deleted", file=path.name)
    for path in CACHE_DIR.glob("lb_*"):
        path.unlink()
        logger.info("cache_deleted", file=path.name)


def load_data() -> tuple[list[QuestionSet], list[ResolvedQuestion]]:
    """Main entry point: fetch all data and return question sets + resolved questions."""
    question_sets = fetch_all_question_sets()
    resolutions = fetch_all_resolutions()
    resolved = join_resolved_questions(question_sets, resolutions)
    return question_sets, resolved
