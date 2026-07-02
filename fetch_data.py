"""Fetch ForecastBench question sets and resolutions from GitHub."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import requests
from pydantic import BaseModel, Field


REPO_OWNER = "forecastingresearch"
REPO_NAME = "forecastbench-datasets"
RAW_BASE = f"https://raw.githubusercontent.com/{REPO_OWNER}/{REPO_NAME}/main"
API_BASE = f"https://api.github.com/repos/{REPO_OWNER}/{REPO_NAME}/contents"
CACHE_DIR = Path(".cache")


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


class QuestionSet(BaseModel):
    forecast_due_date: str
    question_set: str = ""
    questions: list[Question]


class Resolution(BaseModel):
    id: str
    outcome: int | None = None
    resolution_date: str | None = None


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
        return json.loads(cached.read_text())
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()
    data = resp.json()
    cached.write_text(json.dumps(data))
    return data


def list_question_set_files() -> list[str]:
    """List available question set JSON filenames from the GitHub repo."""
    data = _fetch_json(f"{API_BASE}/question_sets", "question_sets_listing.json")
    return [item["name"] for item in data if item["name"].endswith(".json")]


def list_resolution_files() -> list[str]:
    """List available resolution JSON filenames from the GitHub repo."""
    data = _fetch_json(f"{API_BASE}/resolutions", "resolutions_listing.json")
    return [item["name"] for item in data if item["name"].endswith(".json")]


def fetch_question_set(filename: str) -> QuestionSet:
    """Fetch and parse a single question set file."""
    url = f"{RAW_BASE}/question_sets/{filename}"
    data = _fetch_json(url, f"qs_{filename}")
    return QuestionSet.model_validate(data)


def fetch_resolution(filename: str) -> list[Resolution]:
    """Fetch and parse a single resolution file."""
    url = f"{RAW_BASE}/resolutions/{filename}"
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
            print(f"Warning: failed to fetch question set {f}: {e}")
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
            print(f"Warning: failed to fetch resolution {f}: {e}")
    return resolutions


def join_resolved_questions(
    question_sets: list[QuestionSet],
    resolutions: dict[str, Resolution],
) -> list[ResolvedQuestion]:
    """Join questions with their resolutions, returning only resolved questions."""
    resolved = []
    for qs in question_sets:
        for q in qs.questions:
            if q.id in resolutions and resolutions[q.id].outcome is not None:
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
                        outcome=r.outcome,  # type: ignore[arg-type]
                        resolution_date=r.resolution_date,
                        forecast_due_date=qs.forecast_due_date,
                        question_set=qs.question_set,
                    )
                )
    return resolved


def load_data() -> tuple[list[QuestionSet], list[ResolvedQuestion]]:
    """Main entry point: fetch all data and return question sets + resolved questions."""
    question_sets = fetch_all_question_sets()
    resolutions = fetch_all_resolutions()
    resolved = join_resolved_questions(question_sets, resolutions)
    return question_sets, resolved
