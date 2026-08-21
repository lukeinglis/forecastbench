"""Evaluation entrypoint for ForecastBench backtest harness."""

from __future__ import annotations

import asyncio
import datetime
import hashlib
import inspect
import json
import os
import re
from datetime import timezone
from pathlib import Path
from typing import Any, NamedTuple, Protocol, Union

os.environ.setdefault("LITELLM_LOG", "ERROR")
import litellm  # noqa: E402

litellm.suppress_debug_info = True

from fetch_data import MARKET_SOURCES, Question, QuestionSet, Resolution, ResolvedQuestion, load_data, join_resolved_questions, fetch_question_set, fetch_all_resolutions, list_question_set_files, fetch_leaderboard, refresh_cache  # noqa: E402
from logging_config import configure_logging, generate_run_id, get_logger  # noqa: E402
from score import ScoringResult, brier_skill_score, score_forecasts  # noqa: E402

logger = get_logger("eval")

CACHE_DIR = Path(os.getenv("FORECAST_CACHE_DIR", ".cache/forecasts"))
RESULTS_DIR = Path("results")


class SyncForecaster(Protocol):
    def __call__(
        self, question: Question,
        resolution_date: str | None = ...,
        source: str | None = ...,
        resolution_dates: Any = ...,
        prompt_variant: str = ...,
    ) -> float: ...


class AsyncForecaster(Protocol):
    async def __call__(
        self, question: Question,
        resolution_date: str | None = ...,
        source: str | None = ...,
        resolution_dates: Any = ...,
        prompt_variant: str = ...,
    ) -> float: ...


class AsyncMultiForecaster(Protocol):
    async def __call__(
        self, question: Question,
        resolution_dates: list[str],
        source: str | None = ...,
        prompt_variant: str = ...,
    ) -> list[float]: ...


class SyncMultiForecaster(Protocol):
    def __call__(
        self, question: Question,
        resolution_dates: list[str],
        source: str | None = ...,
        prompt_variant: str = ...,
    ) -> list[float]: ...


Forecaster = Union[SyncForecaster, AsyncForecaster]


def _is_multi_horizon(q: Question) -> bool:
    if q.source.lower() in MARKET_SOURCES:
        return False
    rd = q.resolution_dates
    result = isinstance(rd, list) and len(rd) > 1
    if result:
        logger.debug("multi_horizon_detected", question_id=q.id, n_horizons=len(rd))
    return result


class EvalResult(NamedTuple):
    scoring: ScoringResult
    forecasts: dict[str, float]
    resolved: list[ResolvedQuestion]
    model_slug: str



def is_async_forecaster(forecaster: Forecaster) -> bool:
    result = inspect.iscoroutinefunction(forecaster)
    logger.debug("forecaster_type", async_mode=result)
    return result


_MARKET_ANCHOR_WEIGHT = 0.94
_EXTREMITY_FLOOR = 0.04
_EXTREMITY_CEIL = 0.96
_DATASET_SHRINKAGE = 0.06


def _apply_calibration(
    forecasts: dict[str, float],
    questions: list[Question],
) -> dict[str, float]:
    logger.debug("apply_calibration_start", n_forecasts=len(forecasts), n_questions=len(questions))
    q_by_id: dict[str, Question] = {q.id: q for q in questions}
    calibrated: dict[str, float] = {}
    for key, prob in forecasts.items():
        base_id = key.rsplit("_", 1)[0] if "_" in key else key
        q = q_by_id.get(base_id) or q_by_id.get(key)

        if q is not None:
            is_market = q.source.lower() in MARKET_SOURCES
            if is_market:
                fv = getattr(q, "freeze_datetime_value", None)
                if fv is not None and 0.0 <= fv <= 1.0:
                    prob = _MARKET_ANCHOR_WEIGHT * fv + (1.0 - _MARKET_ANCHOR_WEIGHT) * prob
            else:
                prob = (1.0 - _DATASET_SHRINKAGE) * prob + _DATASET_SHRINKAGE * 0.5

        prob = max(_EXTREMITY_FLOOR, min(_EXTREMITY_CEIL, prob))

        calibrated[key] = prob
    return calibrated


_PROVIDER_PREFIXES = (
    "vertex_ai/", "openai/", "anthropic/", "google/",
    "litellm/", "azure/", "bedrock/",
)


def _forecaster_fingerprint(prompt_variant: str = "default") -> str:
    """Hash of everything that can change a forecast.

    Hashes the whole of lab_forecaster.py rather than just the prompt builder.
    Coarse on purpose: an agent editing that file MUST invalidate the cache.
    Over-invalidating costs a re-run. Under-invalidating silently scores stale
    forecasts against new code, which is unrecoverable in an experiment loop.
    """
    parts = [
        os.getenv("FORECAST_MODEL", "vertex_ai/claude-sonnet-4@20250514"),
        os.getenv("FORECAST_TEMPERATURE", ""),
        os.getenv("FORECAST_MAX_TOKENS", ""),
        os.getenv("FORECAST_CACHE_BUST", ""),
        prompt_variant,
    ]
    src = Path(__file__).resolve().parent / "lab_forecaster.py"
    if src.exists():
        parts.append(hashlib.sha256(src.read_bytes()).hexdigest())
    return hashlib.sha256("|".join(parts).encode()).hexdigest()[:12]


def _model_slug(
    agent_name: str | None = None,
    run_label: str | None = None,
    prompt_variant: str = "default",
) -> str:
    logger.debug("model_slug_build", agent=agent_name, label=run_label, variant=prompt_variant)
    raw = os.getenv("FORECAST_MODEL", "vertex_ai/claude-sonnet-4@20250514")
    for prefix in _PROVIDER_PREFIXES:
        if raw.startswith(prefix):
            raw = raw[len(prefix):]
            break
    if raw.startswith("claude-"):
        raw = raw[len("claude-"):]
    raw = raw.replace("@", "-")
    slug = re.sub(r"[^\w\-.]", "_", raw)
    if agent_name and agent_name not in ("lab", "dummy"):
        slug = f"{slug}.{agent_name}"
    if run_label:
        safe_label = re.sub(r"[^\w\-.]", "_", run_label)
        slug = f"{slug}.{safe_label}"
    slug = f"{slug}.{_forecaster_fingerprint(prompt_variant)}"
    return slug


def _cache_path_for(model_slug: str, question_id: str) -> Path:
    safe_qid = re.sub(r"[^\w\-.]", "_", question_id)
    return CACHE_DIR / model_slug / f"{safe_qid}.json"


def _read_cache(model_slug: str, question_id: str) -> float | None:
    path = _cache_path_for(model_slug, question_id)
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text())
        prob = float(data["probability"])
        logger.debug("cache_hit", question_id=question_id, probability=prob)
        return prob
    except (json.JSONDecodeError, KeyError, ValueError, TypeError):
        logger.warning("cache_read_error", question_id=question_id, path=str(path))
        return None


def _write_cache(model_slug: str, question_id: str, probability: float) -> None:
    path = _cache_path_for(model_slug, question_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({
        "probability": probability,
        "model": model_slug,
        "question_id": question_id,
    }))
    logger.debug("cache_write", question_id=question_id, probability=probability)


def save_result(
    result: ScoringResult,
    forecasts: dict[str, float],
    outcomes: dict[str, int],
    model_slug: str,
    question_sets_used: list[str],
    n_held_out: int,
    round_name: str | None = None,
    sources: dict[str, str] | None = None,
    prefix: str = "",
    costs: dict[str, float] | None = None,
) -> Path:
    """Save run result to results/{prefix}{timestamp}_{model_slug}[_{round}].json."""
    timestamp = datetime.datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    metadata: dict[str, object] = {
        "n_questions": result.n_dataset + result.n_market,
        "n_held_out": n_held_out,
        "question_sets_used": question_sets_used,
    }
    if round_name is not None:
        metadata["round"] = round_name
    payload: dict[str, Any] = {
        "timestamp": timestamp,
        "model_slug": model_slug,
        "scoring_result": {
            "dataset_brier": result.dataset_brier,
            "dataset_index": result.dataset_index,
            "market_brier": result.market_brier,
            "market_index": result.market_index,
            "overall_brier": result.overall_brier,
            "overall_index": result.overall_index,
            "n_dataset": result.n_dataset,
            "n_market": result.n_market,
            "n_missing": result.n_missing,
            "difficulty_adjusted": result.difficulty_adjusted,
        },
        "forecasts": forecasts,
        "outcomes": outcomes,
        "sources": sources,
        "metadata": metadata,
    }
    if costs:
        payload["costs"] = costs
        source_costs: dict[str, list[float]] = {}
        for qid, cost in costs.items():
            src = (sources or {}).get(qid, "unknown")
            source_costs.setdefault(src, []).append(cost)
        payload["cost_summary"] = {
            "total_usd": sum(costs.values()),
            "mean_usd": sum(costs.values()) / len(costs) if costs else 0.0,
            "by_source": {
                src: {"total_usd": sum(vals), "mean_usd": sum(vals) / len(vals), "count": len(vals)}
                for src, vals in sorted(source_costs.items())
            },
        }
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    if round_name is not None:
        safe_round = re.sub(r"[^\w\-.]", "_", round_name)
        path = RESULTS_DIR / f"{prefix}{timestamp}_{model_slug}_{safe_round}.json"
    else:
        path = RESULTS_DIR / f"{prefix}{timestamp}_{model_slug}.json"
    path.write_text(json.dumps(payload, indent=2))
    return path


def load_previous_results(results_dir: Path | None = None) -> list[dict[str, object]]:
    """Load all previously saved results for building peer pools."""
    if results_dir is None:
        results_dir = RESULTS_DIR
    if not results_dir.exists():
        logger.debug("load_previous_results_no_dir", path=str(results_dir))
        return []
    results: list[dict[str, object]] = []
    for p in sorted(results_dir.glob("*.json")):
        try:
            data = json.loads(p.read_text())
            results.append(data)
        except (json.JSONDecodeError, KeyError):
            logger.warning("load_previous_result_error", path=str(p))
            continue
    logger.info("load_previous_results", n_loaded=len(results))
    return results


def split_held_out(
    question_sets: list[QuestionSet],
    n_held_out: int = 2,
) -> tuple[list[QuestionSet], list[QuestionSet]]:
    """Split question sets into iteration and held-out sets by forecast_due_date."""
    if n_held_out < 0:
        raise ValueError(f"n_held_out must be non-negative, got {n_held_out}")
    if n_held_out >= len(question_sets):
        logger.info("split_held_out_all", n_total=len(question_sets), n_held_out=n_held_out)
        return [], list(question_sets)

    sorted_qs = sorted(question_sets, key=lambda qs: qs.forecast_due_date)
    split_point = len(sorted_qs) - n_held_out
    iteration_set = sorted_qs[:split_point]
    held_out_set = sorted_qs[split_point:]
    logger.info("split_held_out", n_iteration=len(iteration_set), n_held_out=len(held_out_set))
    return iteration_set, held_out_set


def _build_question(q: Question | ResolvedQuestion, forecast_due_date: str | None = None) -> Question:
    fdd = forecast_due_date or getattr(q, "forecast_due_date", None)
    return Question(
        id=q.id,
        source=q.source,
        question=q.question,
        background=getattr(q, "background", ""),
        resolution_criteria=getattr(q, "resolution_criteria", ""),
        freeze_datetime=getattr(q, "freeze_datetime", None),
        freeze_datetime_value=getattr(q, "freeze_datetime_value", None),
        resolution_dates=getattr(q, "resolution_dates", None),
        url=getattr(q, "url", None),
        combination_of=getattr(q, "combination_of", None),
        source_intro=getattr(q, "source_intro", None),
        freeze_datetime_value_explanation=getattr(q, "freeze_datetime_value_explanation", None),
        market_info_open_datetime=getattr(q, "market_info_open_datetime", None),
        market_info_close_datetime=getattr(q, "market_info_close_datetime", None),
        market_info_resolution_criteria=getattr(q, "market_info_resolution_criteria", None),
        forecast_due_date=fdd,
    )


async def run_eval(
    forecaster: Forecaster,
    n_held_out: int = 2,
    raw: bool = False,
    round_name: str | None = None,
    prompt_variant: str = "default",
    submit_mode: bool = False,
    agent_name: str | None = None,
    n_rounds: int | None = None,
    run_label: str | None = None,
    multi_forecaster: AsyncMultiForecaster | SyncMultiForecaster | None = None,
    question_filter: set[str] | None = None,
) -> EvalResult:
    """Run the full evaluation pipeline."""
    if round_name is not None:
        logger.info("round_eval_start", round=round_name)
        filename = round_name if round_name.endswith(".json") else round_name + ".json"
        question_set = fetch_question_set(filename)
        resolutions = fetch_all_resolutions()
        iteration_resolved = join_resolved_questions([question_set], resolutions)
        iteration_set = [question_set]
        logger.info("round_eval_loaded", round=round_name, n_questions=len(iteration_resolved))
    else:
        question_sets, resolved = load_data()
        if n_rounds is not None:
            sorted_qs = sorted(question_sets, key=lambda qs: qs.forecast_due_date, reverse=True)
            question_sets = sorted_qs[:n_rounds]
            logger.info("rounds_filter", n_rounds=n_rounds, n_selected=len(question_sets),
                        dates=[qs.forecast_due_date for qs in question_sets])
        iteration_set, _held_out = split_held_out(question_sets, n_held_out)
        resolutions_by_id: dict[str, list[Resolution]] = {}
        for q in resolved:
            r = Resolution(id=q.id, outcome=q.outcome, resolution_date=q.resolution_date)
            resolutions_by_id.setdefault(q.id, []).append(r)
        iteration_resolved = join_resolved_questions(
            iteration_set, resolutions_by_id,
        )

    if question_filter is not None:
        before = len(iteration_resolved)
        iteration_resolved = [q for q in iteration_resolved if q.id in question_filter]
        logger.info("question_filter_applied", n_before=before,
                    n_after=len(iteration_resolved), n_pinned=len(question_filter))

    if submit_mode:
        all_questions: list[Question] = []
        for qs in iteration_set:
            for q in qs.questions:
                all_questions.append(_build_question(q, forecast_due_date=qs.forecast_due_date))
        questions = all_questions
        logger.info("submit_mode_enabled", n_all=len(questions), n_resolved=len(iteration_resolved))
    else:
        seen_ids: set[str] = set()
        questions = []
        for q in iteration_resolved:
            if q.id not in seen_ids:
                seen_ids.add(q.id)
                questions.append(_build_question(q))
        logger.info("forecasting_questions", n_base=len(questions), n_resolved=len(iteration_resolved))
    model_slug = _model_slug(agent_name, run_label=run_label, prompt_variant=prompt_variant)

    if is_async_forecaster(forecaster):
        forecasts = await _run_async(
            forecaster, questions, model_slug,  # type: ignore[arg-type]
            prompt_variant=prompt_variant,
            multi_forecaster=multi_forecaster,  # type: ignore[arg-type]
        )
    else:
        forecasts = _run_sync(
            forecaster, questions, model_slug,  # type: ignore[arg-type]
            prompt_variant=prompt_variant,
            multi_forecaster=multi_forecaster,  # type: ignore[arg-type]
        )

    forecasts = _apply_calibration(forecasts, questions)

    has_composite = any(
        "_" in k and k != q_id
        for k in forecasts
        for q_id in [k.rsplit("_", 1)[0]] if k.rsplit("_", 1)[0] != k
    )
    if has_composite:
        scoring_resolved: list[ResolvedQuestion] = []
        for rq in iteration_resolved:
            composite = f"{rq.id}_{rq.resolution_date}" if rq.resolution_date and rq.resolution_date != "N/A" else None
            if composite and composite in forecasts:
                scoring_resolved.append(rq.model_copy(update={"id": composite, "resolution_date": None}))
            else:
                scoring_resolved.append(rq)
    else:
        scoring_resolved = iteration_resolved

    all_forecasts: dict[str, dict[str, float]] | None = None
    if not raw:
        previous = load_previous_results()
        if len(previous) >= 2:
            all_forecasts = {}
            for prev in previous:
                slug = prev.get("model_slug")
                fcs = prev.get("forecasts")
                if slug is None or fcs is None:
                    continue
                all_forecasts[str(slug)] = fcs  # type: ignore[assignment]
            logger.info("difficulty_adjustment_enabled", n_peers=len(all_forecasts))
        else:
            logger.info("difficulty_adjustment_skipped",
                        n_results=len(previous),
                        reason="need_at_least_2_prior_results",
                        note="scores_not_difficulty_adjusted_this_run")

    result = score_forecasts(
        forecasts, scoring_resolved,
        difficulty_adjusted=not raw,
        all_forecasts=all_forecasts,
    )
    _print_results(result)

    outcomes = {q.id: q.outcome for q in iteration_resolved}
    sources = {q.id: q.source.lower() for q in iteration_resolved}
    question_sets_used = [qs.forecast_due_date for qs in iteration_set]

    costs: dict[str, float] | None = None
    try:
        from lab_forecaster import get_tracked_costs
        tracked = get_tracked_costs()
        if tracked:
            costs = tracked
    except (ImportError, AttributeError):
        pass

    if submit_mode:
        result_path = save_result(
            result, forecasts, outcomes, model_slug,
            question_sets_used, n_held_out, round_name=round_name,
            sources=sources,
            prefix="submit_",
            costs=costs,
        )
    else:
        result_path = save_result(
            result, forecasts, outcomes, model_slug,
            question_sets_used, n_held_out, round_name=round_name,
            sources=sources,
            costs=costs,
        )
    logger.info("results_saved", path=str(result_path))

    return EvalResult(scoring=result, forecasts=forecasts, resolved=iteration_resolved, model_slug=model_slug)


def _run_sync(
    forecaster: SyncForecaster,
    questions: list[Question],
    model_slug: str,
    prompt_variant: str = "default",
    multi_forecaster: SyncMultiForecaster | None = None,
) -> dict[str, float]:
    forecasts: dict[str, float] = {}
    for q in questions:
        if _is_multi_horizon(q) and multi_forecaster is not None:
            rd = q.resolution_dates
            composite_keys = [f"{q.id}_{d}" for d in rd]
            all_cached = True
            for ck in composite_keys:
                cached = _read_cache(model_slug, ck)
                if cached is not None:
                    forecasts[ck] = cached
                else:
                    all_cached = False
            if all_cached:
                continue
            try:
                probs = multi_forecaster(
                    q, resolution_dates=rd, source=q.source,
                    prompt_variant=prompt_variant,
                )
            except ValueError:
                logger.warning("parse_failure_skip", question_id=q.id)
                continue
            except Exception:
                logger.warning("forecast_error_skip", question_id=q.id, exc_info=True)
                continue
            if probs is None:
                logger.warning("multi_horizon_none_skip", question_id=q.id)
                continue
            if len(probs) != len(rd):
                logger.warning("multi_horizon_length_mismatch", question_id=q.id,
                               n_probs=len(probs), n_horizons=len(rd))
                continue
            for date, prob in zip(rd, probs):
                ck = f"{q.id}_{date}"
                forecasts[ck] = prob
                _write_cache(model_slug, ck, prob)
        else:
            cached = _read_cache(model_slug, q.id)
            if cached is not None:
                forecasts[q.id] = cached
                continue
            try:
                prob = forecaster(
                    q, source=q.source, resolution_dates=q.resolution_dates,
                    prompt_variant=prompt_variant,
                )
            except ValueError:
                logger.warning("parse_failure_skip", question_id=q.id)
                continue
            except Exception:
                logger.warning("forecast_error_skip", question_id=q.id, exc_info=True)
                continue
            forecasts[q.id] = prob
            _write_cache(model_slug, q.id, prob)
    return forecasts


async def _run_async(
    forecaster: AsyncForecaster,
    questions: list[Question],
    model_slug: str,
    prompt_variant: str = "default",
    multi_forecaster: AsyncMultiForecaster | None = None,
) -> dict[str, float]:
    from tqdm.asyncio import tqdm_asyncio

    concurrency = max(1, int(os.getenv("FORECAST_CONCURRENCY", "10")))
    semaphore = asyncio.Semaphore(concurrency)

    async def _forecast_one(
        q: Question,
    ) -> list[tuple[str, float]] | None:
        if _is_multi_horizon(q) and multi_forecaster is not None:
            rd = q.resolution_dates
            composite_keys = [f"{q.id}_{d}" for d in rd]
            cached_results: list[tuple[str, float]] = []
            all_cached = True
            for ck in composite_keys:
                cached = _read_cache(model_slug, ck)
                if cached is not None:
                    cached_results.append((ck, cached))
                else:
                    all_cached = False
            if all_cached:
                return cached_results
            async with semaphore:
                try:
                    probs = await multi_forecaster(
                        q, resolution_dates=rd, source=q.source,
                        prompt_variant=prompt_variant,
                    )
                except ValueError:
                    logger.warning("parse_failure_skip", question_id=q.id)
                    return None
                except Exception:
                    logger.warning("forecast_error_skip", question_id=q.id, exc_info=True)
                    return None
            # multi_forecaster returns list[float] | None: it RETURNS None on a
            # parse failure rather than raising, so the try above does not catch
            # it. Without this guard the zip below raises TypeError outside the
            # try, which propagates out of gather() and kills the whole run.
            if probs is None:
                logger.warning("multi_horizon_none_skip", question_id=q.id)
                return None
            if len(probs) != len(rd):
                logger.warning("multi_horizon_length_mismatch", question_id=q.id,
                               n_probs=len(probs), n_horizons=len(rd))
                return None
            results: list[tuple[str, float]] = []
            for date, prob in zip(rd, probs):
                ck = f"{q.id}_{date}"
                _write_cache(model_slug, ck, prob)
                results.append((ck, prob))
            return results
        else:
            cached = _read_cache(model_slug, q.id)
            if cached is not None:
                return [(q.id, cached)]
            async with semaphore:
                try:
                    prob = await forecaster(
                        q,
                        source=q.source,
                        resolution_dates=q.resolution_dates,
                        prompt_variant=prompt_variant,
                    )
                except ValueError:
                    logger.warning("parse_failure_skip", question_id=q.id)
                    return None
                except Exception:
                    logger.warning("forecast_error_skip", question_id=q.id, exc_info=True)
                    return None
            _write_cache(model_slug, q.id, prob)
            return [(q.id, prob)]

    tasks = [_forecast_one(q) for q in questions]
    raw_results = await tqdm_asyncio.gather(*tasks, desc="Forecasting")
    forecasts: dict[str, float] = {}
    for r in raw_results:
        if r is not None:
            for qid, prob in r:
                forecasts[qid] = prob
    return forecasts


def _normalize_round_name(name: str) -> str:
    name = name.removesuffix(".json")
    if not name.endswith(("-llm", "-human")):
        name = name + "-llm"
    logger.debug("normalize_round_name", result=name)
    return name


def list_rounds() -> list[tuple[str, int]]:
    filenames = list_question_set_files()
    rounds: list[tuple[str, int]] = []
    for fname in sorted(filenames, reverse=True):
        try:
            qs = fetch_question_set(fname)
            round_name = fname.removesuffix(".json")
            rounds.append((round_name, len(qs.questions)))
        except Exception:
            logger.warning("list_rounds_fetch_failed", filename=fname)
    return rounds


def print_leaderboard_comparison(
    user_index: float,
    leaderboard_name: str = "baseline",
) -> None:
    try:
        rows = fetch_leaderboard(leaderboard_name)
    except Exception:
        logger.warning("leaderboard_fetch_failed", name=leaderboard_name, exc_info=True)
        print("  (Could not fetch leaderboard data)")
        return

    entries: list[tuple[int, str, float]] = []
    for row in rows:
        try:
            rank = int(row.get("Rank", "0"))
            model = row.get("Model", "Unknown")
            overall_str = row.get("Overall", "").strip().rstrip("%")
            overall = float(overall_str)
            entries.append((rank, model, overall))
        except (ValueError, TypeError):
            continue

    if not entries:
        print("  (No parseable leaderboard entries)")
        return

    entries.sort(key=lambda e: e[2], reverse=True)

    user_rank = 1
    for _, _, score in entries:
        if score >= user_index:
            user_rank += 1
        else:
            break

    top_5 = entries[:5]
    user_pos = user_rank - 1
    context_start = max(0, user_pos - 2)
    context_end = min(len(entries), user_pos + 3)
    context = entries[context_start:context_end]
    bottom = [e for e in entries if e[2] <= 50.5]
    bottom_entry = bottom[-1] if bottom else entries[-1]

    shown_ranks: set[int] = set()
    display_entries: list[tuple[int | None, str, float, bool]] = []

    for rank, model, score in top_5:
        display_entries.append((rank, model, score, False))
        shown_ranks.add(rank)

    needs_sep_before_context = True
    for rank, model, score in context:
        if rank not in shown_ranks:
            if needs_sep_before_context and display_entries:
                display_entries.append((None, "", 0.0, False))
                needs_sep_before_context = False
            display_entries.append((rank, model, score, False))
            shown_ranks.add(rank)

    user_entry_idx: int = len(display_entries)
    for idx, (_r, _m, score, _u) in enumerate(display_entries):
        if score < user_index:
            user_entry_idx = idx
            break
    display_entries.insert(user_entry_idx, (None, ">>> Your result <<<", user_index, True))

    if bottom_entry[0] not in shown_ranks:
        display_entries.append((None, "", 0.0, False))
        display_entries.append((bottom_entry[0], bottom_entry[1], bottom_entry[2], False))

    print(f"\nLeaderboard comparison ({leaderboard_name}):")
    print(f"  {'Rank':<6s} {'Model':<35s} {'Overall':>8s}")
    for e_rank, e_model, e_score, e_is_user in display_entries:
        if e_model == "" and e_rank is None:
            print(f"  {'...':<6s}")
            continue
        rank_str = "---" if e_is_user or e_rank is None else str(e_rank)
        score_str = f"{e_score:.1f}%"
        print(f"  {rank_str:<6s} {e_model:<35s} {score_str:>8s}")


def _print_results(result: ScoringResult) -> None:
    bss = brier_skill_score(result.overall_brier)
    logger.info(
        "eval_results",
        dataset_brier=round(result.dataset_brier, 4),
        dataset_index=round(result.dataset_index, 1),
        n_dataset=result.n_dataset,
        market_brier=round(result.market_brier, 4),
        market_index=round(result.market_index, 1),
        n_market=result.n_market,
        overall_brier=round(result.overall_brier, 4),
        overall_index=round(result.overall_index, 1),
        brier_skill_score=round(bss, 4),
        n_missing=result.n_missing,
    )


def main() -> None:
    import argparse

    configure_logging()
    run_id = generate_run_id()
    logger.info("eval_start", run_id=run_id)

    parser = argparse.ArgumentParser(description="ForecastBench evaluation")
    parser.add_argument(
        "--agent",
        choices=["dummy", "lab"],
        default="dummy",
        help="Forecaster agent to use (default: dummy)",
    )
    parser.add_argument("--raw", action="store_true", help="Disable difficulty adjustment")
    parser.add_argument("--round", metavar="ROUND", help="Evaluate a single round")
    parser.add_argument(
        "--prompt",
        choices=["default", "zero-shot", "zero-shot-fv", "zero-shot-no-fv", "dataset"],
        default="default",
        help="Prompt variant",
    )
    parser.add_argument("--leaderboard", nargs="?", const="baseline", default=None,
                        choices=["baseline", "tournament", "dataset", "preliminary"],
                        help="Compare against leaderboard")
    parser.add_argument("--refresh", action="store_true", help="Clear cached data")
    parser.add_argument("--ci", action="store_true", help="Show bootstrap confidence intervals")
    parser.add_argument("--list-rounds", action="store_true", help="List available rounds and exit")
    parser.add_argument("--submit", action="store_true", default=False,
                        help="Forecast all questions for submission coverage")
    parser.add_argument(
        "--track",
        choices=["baseline", "tournament"],
        default="baseline",
        help="Competition track: baseline (no tools/ensemble/RAG/calibration) or tournament (all features allowed)",
    )
    parser.add_argument(
        "--rounds",
        type=int,
        metavar="N",
        default=None,
        help="Limit evaluation to the N most recent question sets by forecast_due_date",
    )
    parser.add_argument(
        "--run-label",
        type=str,
        default=None,
        help="Label for this run (e.g., thinking, rag, ensemble)",
    )
    args = parser.parse_args()

    if args.track == "baseline":
        ensemble_n = int(os.getenv("FORECAST_ENSEMBLE_N", "1"))
        if ensemble_n > 1:
            print(f"Error: --track baseline forbids FORECAST_ENSEMBLE_N={ensemble_n} (tournament-only feature)")
            raise SystemExit(1)
        if os.getenv("FORECAST_RAG", "").lower() == "true":
            print("Error: --track baseline forbids FORECAST_RAG=true (tournament-only feature)")
            raise SystemExit(1)

    if args.refresh:
        logger.info("cache_refresh_requested")
        refresh_cache()

    if args.list_rounds:
        rounds = list_rounds()
        if not rounds:
            print("No rounds available.")
        else:
            print("Available rounds:")
            for name, count in rounds:
                print(f"  {name:<25s} {count:>4d} questions")
        return

    round_name: str | None = None
    if args.round:
        round_name = _normalize_round_name(args.round)

    multi_forecaster_fn: AsyncMultiForecaster | SyncMultiForecaster | None = None
    if args.agent == "lab":
        from lab_forecaster import aforecast, amulti_forecast
        forecaster: Forecaster = aforecast
        multi_forecaster_fn = amulti_forecast
    else:
        from dummy_forecaster import forecast
        forecaster = forecast

    eval_result = asyncio.run(run_eval(
        forecaster, raw=args.raw, round_name=round_name,
        prompt_variant=args.prompt,
        submit_mode=args.submit,
        agent_name=args.agent,
        n_rounds=args.rounds,
        run_label=args.run_label,
        multi_forecaster=multi_forecaster_fn,
    ))

    if args.ci:
        from score import bootstrap_ci
        pairs = [
            (eval_result.forecasts.get(q.id, 0.5), q.outcome)
            for q in eval_result.resolved
        ]
        lo, hi = bootstrap_ci(pairs)
        logger.info("bootstrap_ci", lower=round(lo, 4), upper=round(hi, 4), ci="95%")

    if args.leaderboard is not None:
        print(f"\nYour result:  Overall Index = {eval_result.scoring.overall_index:.1f}%")
        print_leaderboard_comparison(eval_result.scoring.overall_index, leaderboard_name=args.leaderboard)

    if args.agent != "dummy":
        _run_analysis(eval_result.forecasts, eval_result.resolved, eval_result.model_slug)


def _run_analysis(
    forecasts: dict[str, float],
    resolved: list[ResolvedQuestion],
    model_slug: str,
) -> None:
    from analyze import (
        analyze_by_source,
        analyze_calibration,
        analyze_biases,
        analyze_decomposition,
        print_analysis,
        save_analysis,
    )

    analysis = {
        "by_source": analyze_by_source(forecasts, resolved),
        "calibration": analyze_calibration(forecasts, resolved),
        "biases": analyze_biases(forecasts, resolved),
        "decomposition": analyze_decomposition(forecasts, resolved),
    }

    print_analysis(analysis)

    analysis_path = Path(f".cache/analysis/{model_slug}/analysis.json")
    save_analysis(analysis, analysis_path)
    logger.info("analysis_saved", path=str(analysis_path))


if __name__ == "__main__":
    main()
