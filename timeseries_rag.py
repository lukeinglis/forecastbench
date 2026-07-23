"""Retrieval-augmented generation for timeseries forecasting questions.

Fetches historical data from FRED, Yahoo Finance, and DBnomics APIs,
caches raw values, and formats compact summaries for prompt injection.
"""

from __future__ import annotations

import json
import os
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import requests

from logging_config import get_logger

logger = get_logger("timeseries_rag")

FORECAST_RAG = os.getenv("FORECAST_RAG", "true").lower() == "true"
FRED_API_KEY = os.getenv("FRED_API_KEY", "")

RAG_CACHE_DIR = Path(".cache/rag")
LOOKBACK_DAYS = 90


def _cache_path(source: str, series_id: str, cutoff_date: str) -> Path:
    safe_id = series_id.replace("/", "_").replace("\\", "_")
    return RAG_CACHE_DIR / source / f"{safe_id}_{cutoff_date}.json"


def _read_cache(source: str, series_id: str, cutoff_date: str) -> dict[str, float] | None:
    path = _cache_path(source, series_id, cutoff_date)
    if path.exists():
        try:
            data = json.loads(path.read_text())
            result: dict[str, float] | None = data.get("values")
            return result
        except (json.JSONDecodeError, KeyError):
            return None
    return None


def _write_cache(source: str, series_id: str, cutoff_date: str, values: dict[str, float]) -> None:
    path = _cache_path(source, series_id, cutoff_date)
    path.parent.mkdir(parents=True, exist_ok=True)
    from datetime import datetime, timezone

    path.write_text(json.dumps({
        "values": values,
        "fetched_at": datetime.now(timezone.utc).isoformat(),
    }))


def _fetch_fred(series_id: str, cutoff_date: date) -> dict[str, float] | None:
    if not FRED_API_KEY:
        logger.warning("fred_no_api_key")
        return None

    import fredapi

    try:
        fred = fredapi.Fred(api_key=FRED_API_KEY)
        start = cutoff_date - timedelta(days=LOOKBACK_DAYS)
        series = fred.get_series(
            series_id,
            observation_start=start,
            observation_end=cutoff_date,
        )
        values: dict[str, float] = {}
        for dt, val in series.items():
            if val is not None and not (hasattr(val, "__class__") and val != val):
                values[str(dt.date())] = float(val)
        return values if values else None
    except Exception:
        logger.warning("fred_fetch_failed", series_id=series_id, exc_info=True)
        return None


def _fetch_yfinance(ticker: str, cutoff_date: date) -> dict[str, float] | None:
    import yfinance

    try:
        start = cutoff_date - timedelta(days=LOOKBACK_DAYS)
        tk = yfinance.Ticker(ticker)
        hist = tk.history(start=str(start), end=str(cutoff_date))
        if hist.empty:
            return None
        close = hist["Close"]
        values: dict[str, float] = {}
        for dt, val in close.items():
            if val is not None and not (hasattr(val, "__class__") and val != val):
                values[str(dt.date()) if hasattr(dt, "date") else str(dt)[:10]] = float(val)
        return values if values else None
    except Exception:
        logger.warning("yfinance_fetch_failed", ticker=ticker, exc_info=True)
        return None


def parse_dbnomics_id(question_id: str) -> tuple[str, str, str] | None:
    """Parse a dbnomics question ID into (provider, dataset, series).

    Format: provider_dataset_series where series may contain underscores/dots.
    Split on first two underscores only.
    """
    parts = question_id.split("_", 2)
    if len(parts) < 3:
        return None
    return parts[0], parts[1], parts[2]


def _fetch_dbnomics(series_id: str, cutoff_date: date) -> dict[str, float] | None:
    parsed = parse_dbnomics_id(series_id)
    if parsed is None:
        logger.warning("dbnomics_parse_failed", series_id=series_id)
        return None

    provider, dataset, series = parsed
    url = f"https://api.db.nomics.world/v22/series/{provider}/{dataset}/{series}?observations=1"

    try:
        resp = requests.get(url, timeout=30)
        resp.raise_for_status()
        data = resp.json()

        series_data = data.get("series", {})
        if isinstance(series_data, dict):
            docs = series_data.get("docs", [])
            if not docs:
                return None
            doc = docs[0]
            period = doc.get("period", [])
            value = doc.get("value", [])
        else:
            return None

        start = cutoff_date - timedelta(days=LOOKBACK_DAYS)
        values: dict[str, float] = {}
        for p, v in zip(period, value):
            if v is None or v == "NA":
                continue
            p_str = str(p)[:10]
            try:
                p_date = date.fromisoformat(p_str)
            except ValueError:
                continue
            if start <= p_date <= cutoff_date:
                values[p_str] = float(v)

        return values if values else None
    except Exception:
        logger.warning("dbnomics_fetch_failed", series_id=series_id, exc_info=True)
        return None


_FETCHERS: dict[str, Any] = {
    "fred": _fetch_fred,
    "yfinance": _fetch_yfinance,
    "dbnomics": _fetch_dbnomics,
}


def format_historical_context(values: dict[str, float], cutoff_date: str) -> str:
    """Format raw date->value dict into a compact summary for prompt injection."""
    if not values:
        return ""

    sorted_dates = sorted(values.keys())
    last_date = sorted_dates[-1]
    current_value = values[last_date]

    all_vals = list(values.values())
    min_val = min(all_vals)
    max_val = max(all_vals)

    change_pct = ""
    if len(sorted_dates) >= 2:
        cutoff = date.fromisoformat(cutoff_date) if isinstance(cutoff_date, str) else cutoff_date
        threshold = cutoff - timedelta(days=30)
        earlier_dates = [d for d in sorted_dates if d <= str(threshold)]
        if earlier_dates:
            ref_date = earlier_dates[-1]
            ref_val = values[ref_date]
            if ref_val != 0:
                pct = ((current_value - ref_val) / abs(ref_val)) * 100
                change_pct = f"\n30-day change: {pct:+.1f}%"

    last_10 = sorted_dates[-10:]
    datapoints = ", ".join(f"{d}: {values[d]}" for d in last_10)

    return (
        f"Recent data (last {LOOKBACK_DAYS} days before {cutoff_date}):\n"
        f"Current value: {current_value}\n"
        f"Range: {min_val} – {max_val}"
        f"{change_pct}\n"
        f"Last {len(last_10)} values: {datapoints}"
    )


def fetch_historical_context(question: Any) -> str:
    """Fetch and format historical context for a timeseries question.

    Returns formatted string on success, empty string on any failure.
    """
    if not FORECAST_RAG:
        return ""

    source = getattr(question, "source", "").lower()
    if source not in _FETCHERS:
        return ""

    question_id = getattr(question, "id", "")
    if not question_id:
        return ""

    cutoff_str = getattr(question, "forecast_due_date", None) or getattr(question, "freeze_datetime", None)
    if not cutoff_str:
        return ""

    cutoff_str = str(cutoff_str)[:10]

    cached = _read_cache(source, question_id, cutoff_str)
    if cached is not None:
        logger.debug("rag_cache_hit", source=source, question_id=question_id)
        return format_historical_context(cached, cutoff_str)

    try:
        cutoff = date.fromisoformat(cutoff_str)
    except ValueError:
        return ""

    fetcher = _FETCHERS[source]
    try:
        values = fetcher(question_id, cutoff)
    except Exception:
        logger.warning("rag_fetch_error", source=source, question_id=question_id, exc_info=True)
        return ""

    if values is None:
        logger.info("rag_fetch_empty", source=source, question_id=question_id)
        return ""

    _write_cache(source, question_id, cutoff_str, values)
    return format_historical_context(values, cutoff_str)
