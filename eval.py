"""Evaluation entrypoint for ForecastBench backtest harness."""

from __future__ import annotations

import asyncio
import inspect
import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, NamedTuple, Protocol, Union

os.environ.setdefault("LITELLM_LOG", "ERROR")
import litellm  # noqa: E402

litellm.suppress_debug_info = True

from fetch_data import MARKET_SOURCES, Question, QuestionSet, Resolution, ResolvedQuestion, load_data, join_resolved_questions, fetch_question_set, fetch_all_resolutions, list_question_set_files, fetch_leaderboard, refresh_cache  # noqa: E402
from logging_config import configure_logging, generate_run_id, get_logger  # noqa: E402
from score import ScoringResult, brier_skill_score, score_forecasts  # noqa: E402

logger = get_logger("eval")

CACHE_DIR = Path(".cache/forecasts")
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


Forecaster = Union[SyncForecaster, AsyncForecaster]


class EvalResult(NamedTuple):
    scoring: ScoringResult
    forecasts: dict[str, float]
    resolved: list[ResolvedQuestion]
    model_slug: str


def _has_multi_horizon(question: Question) -> bool:
    if question.source.lower() in MARKET_SOURCES:
        return False
    rd = question.resolution_dates
    return isinstance(rd, list) and any(d for d in rd)


def _expand_resolved_for_horizons(
    resolved: list[ResolvedQuestion],
) -> list[ResolvedQuestion]:
    expanded: list[ResolvedQuestion] = []
    for rq in resolved:
        if rq.source.lower() in MARKET_SOURCES:
            expanded.append(rq)
            continue
        rd = rq.resolution_dates
        if not isinstance(rd, list) or len(rd) == 0:
            expanded.append(rq)
            continue
        for date_str in rd:
            composite_id = f"{rq.id}_{date_str}"
            expanded.append(
                ResolvedQuestion(
                    id=composite_id,
                    source=rq.source,
                    question=rq.question,
                    background=rq.background,
                    resolution_criteria=rq.resolution_criteria,
                    freeze_datetime=rq.freeze_datetime,
                    freeze_datetime_value=rq.freeze_datetime_value,
                    resolution_dates=rq.resolution_dates,
                    url=rq.url,
                    combination_of=rq.combination_of,
                    source_intro=rq.source_intro,
                    freeze_datetime_value_explanation=rq.freeze_datetime_value_explanation,
                    market_info_open_datetime=rq.market_info_open_datetime,
                    market_info_close_datetime=rq.market_info_close_datetime,
                    market_info_resolution_criteria=rq.market_info_resolution_criteria,
                    outcome=rq.outcome,
                    resolution_date=date_str,
                    forecast_due_date=rq.forecast_due_date,
                    question_set=rq.question_set,
                )
            )
    return expanded


def is_async_forecaster(forecaster: Forecaster) -> bool:
    return inspect.iscoroutinefunction(forecaster)


def _model_slug() -> str:
    raw = os.getenv("FORECAST_MODEL", "default")
    return re.sub(r"[^\w\-.]", "_", raw)


def _cache_path_for(model_slug: str, question_id: str) -> Path:
    safe_qid = re.sub(r"[^\w\-.]", "_", question_id)
    return CACHE_DIR / model_slug / f"{safe_qid}.json"


def _read_cache(model_slug: str, question_id: str) -> float | None:
    path = _cache_path_for(model_slug, question_id)
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text())
        return float(data["probability"])
    except (json.JSONDecodeError, KeyError, ValueError, TypeError):
        return None


def _write_cache(model_slug: str, question_id: str, probability: float) -> None:
    path = _cache_path_for(model_slug, question_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({
        "probability": probability,
        "model": model_slug,
        "question_id": question_id,
    }))


def save_result(
    result: ScoringResult,
    forecasts: dict[str, float],
    outcomes: dict[str, int],
    model_slug: str,
    question_sets_used: list[str],
    n_held_out: int,
    round_name: str | None = None,
) -> Path:
    """Save run result to results/{timestamp}_{model_slug}[_{round}].json."""
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    metadata: dict[str, object] = {
        "n_questions": result.n_dataset + result.n_market,
        "n_held_out": n_held_out,
        "question_sets_used": question_sets_used,
    }
    if round_name is not None:
        metadata["round"] = round_name
    payload = {
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
        "metadata": metadata,
    }
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    if round_name is not None:
        safe_round = re.sub(r"[^\w\-.]", "_", round_name)
        path = RESULTS_DIR / f"{timestamp}_{model_slug}_{safe_round}.json"
    else:
        path = RESULTS_DIR / f"{timestamp}_{model_slug}.json"
    path.write_text(json.dumps(payload, indent=2))
    return path


def load_previous_results(results_dir: Path | None = None) -> list[dict[str, object]]:
    """Load all previously saved results for building peer pools."""
    if results_dir is None:
        results_dir = RESULTS_DIR
    if not results_dir.exists():
        return []
    results: list[dict[str, object]] = []
    for p in sorted(results_dir.glob("*.json")):
        try:
            data = json.loads(p.read_text())
            results.append(data)
        except (json.JSONDecodeError, KeyError):
            continue
    return results


def split_held_out(
    question_sets: list[QuestionSet],
    n_held_out: int = 2,
) -> tuple[list[QuestionSet], list[QuestionSet]]:
    """Split question sets into iteration and held-out sets by forecast_due_date.

    The most recent n_held_out sets (by date descending) go to held-out.
    """
    if n_held_out < 0:
        raise ValueError(f"n_held_out must be non-negative, got {n_held_out}")
    if n_held_out >= len(question_sets):
        return [], list(question_sets)

    sorted_qs = sorted(question_sets, key=lambda qs: qs.forecast_due_date)
    split_point = len(sorted_qs) - n_held_out
    iteration_set = sorted_qs[:split_point]
    held_out_set = sorted_qs[split_point:]
    return iteration_set, held_out_set


def _build_question(q: Question | ResolvedQuestion, forecast_due_date: str | None = None) -> Question:
    """Build a Question from a ResolvedQuestion or Question-like object."""
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
    prompt_variant: str = "zero-shot",
    multi_horizon: bool = False,
) -> EvalResult:
    """Run the full evaluation pipeline.

    When round_name is set, only that single question set is evaluated
    (no held-out split). Otherwise, all question sets are loaded and the
    most recent n_held_out sets are held out.
    """
    if round_name is not None:
        logger.info("round_eval_start", round=round_name)
        filename = round_name if round_name.endswith(".json") else round_name + ".json"
        question_set = fetch_question_set(filename)
        resolutions = fetch_all_resolutions()
        iteration_resolved = join_resolved_questions(
            [question_set], resolutions,
        )
        iteration_set = [question_set]
        logger.info("round_eval_loaded", round=round_name, n_questions=len(iteration_resolved))
    else:
        question_sets, resolved = load_data()
        iteration_set, _held_out = split_held_out(question_sets, n_held_out)
        resolutions_by_id = {q.id: q for q in resolved}
        iteration_resolved = join_resolved_questions(
            iteration_set,
            {q_id: Resolution(id=q_id, outcome=r.outcome, resolution_date=r.resolution_date)
             for q_id, r in resolutions_by_id.items()},
        )

    questions = [_build_question(q) for q in iteration_resolved]
    model_slug = _model_slug()

    if is_async_forecaster(forecaster):
        forecasts = await _run_async(forecaster, questions, model_slug, prompt_variant=prompt_variant, multi_horizon=multi_horizon)  # type: ignore[arg-type]
    else:
        forecasts = _run_sync(forecaster, questions, model_slug, prompt_variant=prompt_variant)  # type: ignore[arg-type]

    expanded_resolved = _expand_resolved_for_horizons(iteration_resolved)

    all_forecasts: dict[str, dict[str, float]] | None = None
    if not raw:
        previous = load_previous_results()
        if len(previous) >= 2:
            all_forecasts = {}
            for prev in previous:
                slug = prev["model_slug"]
                all_forecasts[str(slug)] = prev["forecasts"]  # type: ignore[assignment]
            logger.info("difficulty_adjustment_enabled", n_peers=len(all_forecasts))
        else:
            logger.info("difficulty_adjustment_skipped",
                        n_results=len(previous),
                        reason="need_at_least_2_prior_results",
                        note="scores_not_difficulty_adjusted_this_run")

    result = score_forecasts(
        forecasts, expanded_resolved,
        difficulty_adjusted=not raw,
        all_forecasts=all_forecasts,
    )
    _print_results(result)

    outcomes = {q.id: q.outcome for q in expanded_resolved}
    question_sets_used = [qs.forecast_due_date for qs in iteration_set]
    result_path = save_result(
        result, forecasts, outcomes, model_slug,
        question_sets_used, n_held_out, round_name=round_name,
    )
    logger.info("results_saved", path=str(result_path))

    return EvalResult(scoring=result, forecasts=forecasts, resolved=iteration_resolved, model_slug=model_slug)


def _run_sync(
    forecaster: SyncForecaster,
    questions: list[Question],
    model_slug: str,
    prompt_variant: str = "zero-shot",
) -> dict[str, float]:
    forecasts: dict[str, float] = {}
    for q in questions:
        if _has_multi_horizon(q):
            for date_str in q.resolution_dates:
                composite_key = f"{q.id}_{date_str}"
                cached = _read_cache(model_slug, composite_key)
                if cached is not None:
                    forecasts[composite_key] = cached
                    continue
                try:
                    prob = forecaster(
                        q, resolution_date=date_str,
                        source=q.source, resolution_dates=q.resolution_dates,
                        prompt_variant=prompt_variant,
                    )
                except Exception:
                    logger.warning("forecast_error_fallback", question_id=q.id, resolution_date=date_str, exc_info=True)
                    forecasts[composite_key] = 0.5
                    continue
                forecasts[composite_key] = prob
                _write_cache(model_slug, composite_key, prob)
        else:
            cached = _read_cache(model_slug, q.id)
            if cached is not None:
                forecasts[q.id] = cached
                continue
            prob = forecaster(
                q, source=q.source, resolution_dates=q.resolution_dates,
                prompt_variant=prompt_variant,
            )
            forecasts[q.id] = prob
            _write_cache(model_slug, q.id, prob)
    return forecasts


async def _run_async(
    forecaster: AsyncForecaster,
    questions: list[Question],
    model_slug: str,
    prompt_variant: str = "zero-shot",
    multi_horizon: bool = False,
) -> dict[str, float]:
    from tqdm.asyncio import tqdm_asyncio

    concurrency = max(1, int(os.getenv("FORECAST_CONCURRENCY", "10")))
    semaphore = asyncio.Semaphore(concurrency)

    async def _forecast_one(
        q: Question,
        cache_key: str,
        resolution_date: str | None = None,
    ) -> tuple[str, float]:
        cached = _read_cache(model_slug, cache_key)
        if cached is not None:
            return cache_key, cached
        async with semaphore:
            try:
                prob = await forecaster(
                    q,
                    resolution_date=resolution_date,
                    source=q.source,
                    resolution_dates=q.resolution_dates,
                    prompt_variant=prompt_variant,
                )
            except Exception:
                logger.warning("forecast_error_fallback", question_id=q.id, resolution_date=resolution_date, exc_info=True)
                return cache_key, 0.5
        _write_cache(model_slug, cache_key, prob)
        return cache_key, prob

    async def _forecast_multi_horizon(
        q: Question,
    ) -> list[tuple[str, float]]:
        dates = [d for d in q.resolution_dates if d and str(d).upper() != "N/A"]
        composite_keys = [f"{q.id}_{d}" for d in dates]
        cached_values = {k: _read_cache(model_slug, k) for k in composite_keys}
        if all(v is not None for v in cached_values.values()):
            return [(k, v) for k, v in cached_values.items()]  # type: ignore[misc]

        async with semaphore:
            try:
                from baseline_agent import aforecast_multi_horizon
                probs = await aforecast_multi_horizon(
                    q,
                    resolution_dates=dates,
                    source=q.source,
                    prompt_variant=prompt_variant,
                )
            except Exception:
                logger.warning("multi_horizon_error_fallback", question_id=q.id, exc_info=True)
                probs = None

        results: list[tuple[str, float]] = []
        if probs is None:
            for key in composite_keys:
                results.append((key, 0.5))
        else:
            for key, prob in zip(composite_keys, probs):
                _write_cache(model_slug, key, prob)
                results.append((key, prob))
        return results

    tasks: list[Any] = []
    multi_tasks: list[Any] = []

    for q in questions:
        if _has_multi_horizon(q):
            if multi_horizon:
                multi_tasks.append(_forecast_multi_horizon(q))
            else:
                for date_str in q.resolution_dates:
                    composite_key = f"{q.id}_{date_str}"
                    tasks.append(_forecast_one(q, composite_key, resolution_date=date_str))
        else:
            tasks.append(_forecast_one(q, q.id))

    all_results: list[tuple[str, float]] = []

    if tasks:
        single_results = await tqdm_asyncio.gather(*tasks, desc="Forecasting")
        all_results.extend(single_results)

    if multi_tasks:
        multi_results = await tqdm_asyncio.gather(*multi_tasks, desc="Multi-horizon")
        for batch in multi_results:
            all_results.extend(batch)

    return {qid: prob for qid, prob in all_results}


def _normalize_round_name(name: str) -> str:
    """Ensure round name has the -llm suffix and no .json extension."""
    name = name.removesuffix(".json")
    if not name.endswith(("-llm", "-human")):
        name = name + "-llm"
    return name


def list_rounds() -> list[tuple[str, int]]:
    """List available rounds with question counts."""
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
    """Fetch the leaderboard and show where the user's score ranks."""
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
        choices=["dummy", "baseline"],
        default="dummy",
        help="Forecaster agent to use (default: dummy)",
    )
    parser.add_argument(
        "--raw",
        action="store_true",
        help="Disable difficulty adjustment, use raw Brier scores",
    )
    parser.add_argument(
        "--round",
        metavar="ROUND",
        help="Evaluate a single round (e.g. 2026-07-05-llm or 2026-07-05)",
    )
    parser.add_argument(
        "--prompt",
        choices=["zero-shot", "zero-shot-fv", "dataset"],
        default="zero-shot",
        help="Prompt variant: zero-shot (default), zero-shot-fv (with freeze values), dataset (multi-horizon)",
    )
    parser.add_argument(
        "--leaderboard",
        nargs="?",
        const="baseline",
        default=None,
        choices=["baseline", "tournament", "dataset", "preliminary"],
        help="Compare against leaderboard (default: baseline)",
    )
    parser.add_argument(
        "--refresh",
        action="store_true",
        help="Clear cached data and fetch fresh from ForecastBench repo",
    )
    parser.add_argument(
        "--ci",
        action="store_true",
        help="Show bootstrap confidence intervals for Brier scores",
    )
    parser.add_argument(
        "--multi-horizon",
        action="store_true",
        default=False,
        help="Use single-call multi-horizon forecasting for dataset questions (baseline agent only)",
    )
    parser.add_argument(
        "--list-rounds",
        action="store_true",
        help="List available rounds with question counts and exit",
    )
    args = parser.parse_args()

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

    if args.agent == "baseline":
        from baseline_agent import aforecast
        forecaster: Forecaster = aforecast
    else:
        from dummy_forecaster import forecast
        forecaster = forecast

    eval_result = asyncio.run(run_eval(
        forecaster, raw=args.raw, round_name=round_name,
        prompt_variant=args.prompt,
        multi_horizon=args.multi_horizon and args.agent == "baseline",
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
