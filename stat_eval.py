"""Offline evaluation of the statistical forecaster against resolved ForecastBench questions.

Measures random walk + CDF forecaster accuracy on timeseries sources (fred, dbnomics, yfinance)
without making any LLM API calls. Uses cached historical data when available; falls back to
the naive heuristic when no cached data exists.
"""

from __future__ import annotations

import re
import sys
from collections import defaultdict
from datetime import date

from fetch_data import (
    ResolvedQuestion,
    fetch_all_question_sets,
    fetch_all_resolutions,
    join_resolved_questions,
)
from score import brier_index, brier_score
from statistical_baseline import (
    TIMESERIES_SOURCES,
    extract_threshold,
    compute_random_walk_forecast,
    compute_naive_forecast,
)
from timeseries_rag import _read_cache

_COMPARISON_HIGHER = re.compile(
    r"(?:higher|increased|increase)\b.*(?:resolution_date|compared|market close|its value)", re.I
)
_COMPARISON_LOWER = re.compile(
    r"(?:lower|decreased|decrease)\b.*(?:resolution_date|compared|market close|its value)", re.I
)


def _extract_threshold_extended(
    question_text: str, resolution_criteria: str, freeze_value: float
) -> tuple[float, str] | None:
    """Extended threshold extraction: standard patterns first, then comparison-to-self."""
    result = extract_threshold(question_text, resolution_criteria)
    if result is not None:
        return result
    combined = question_text + " " + resolution_criteria
    if _COMPARISON_HIGHER.search(combined):
        return (freeze_value, "above")
    if _COMPARISON_LOWER.search(combined):
        return (freeze_value, "below")
    return None


def _compute_horizon_days(freeze_datetime: str | None, resolution_date: str | None) -> int | None:
    if not freeze_datetime or not resolution_date:
        return None
    try:
        freeze = date.fromisoformat(str(freeze_datetime)[:10])
        resolve = date.fromisoformat(str(resolution_date)[:10])
        delta = (resolve - freeze).days
        return max(delta, 1)
    except (ValueError, TypeError):
        return None


def _get_resolution_dates(q: ResolvedQuestion) -> list[str]:
    """Extract resolution dates for a question, handling multi-horizon."""
    if q.resolution_date:
        return [q.resolution_date]
    if isinstance(q.resolution_dates, list):
        return [str(d) for d in q.resolution_dates if d and str(d).upper() != "N/A"]
    return []


def _try_cached_historical(q: ResolvedQuestion) -> dict[str, float] | None:
    """Try to load historical data from RAG cache without making API calls."""
    source = q.source.lower()
    question_id = q.id
    cutoff_str = q.forecast_due_date or q.freeze_datetime
    if not cutoff_str or not question_id:
        return None
    cutoff_str = str(cutoff_str)[:10]
    return _read_cache(source, question_id, cutoff_str)


def evaluate_statistical_forecaster() -> None:
    print("Fetching question sets and resolutions...")
    question_sets = fetch_all_question_sets()
    resolutions = fetch_all_resolutions()
    resolved = join_resolved_questions(question_sets, resolutions)
    print(f"Total resolved questions: {len(resolved)}")

    timeseries_qs = [q for q in resolved if q.source.lower() in TIMESERIES_SOURCES]
    print(f"Timeseries questions (fred/dbnomics/yfinance): {len(timeseries_qs)}")

    if not timeseries_qs:
        print("No timeseries questions found. Exiting.")
        sys.exit(0)

    results: list[dict[str, object]] = []
    skipped_no_freeze = 0
    skipped_no_threshold = 0
    skipped_no_horizon = 0
    used_random_walk = 0
    used_naive = 0
    cache_hits = 0

    source_results: dict[str, list[dict[str, object]]] = defaultdict(list)

    for q in timeseries_qs:
        source = q.source.lower()

        if q.freeze_datetime_value is None:
            skipped_no_freeze += 1
            continue

        try:
            freeze_value = float(q.freeze_datetime_value)
        except (TypeError, ValueError):
            skipped_no_freeze += 1
            continue

        threshold_result = _extract_threshold_extended(
            q.question, q.resolution_criteria or "", freeze_value
        )
        if threshold_result is None:
            skipped_no_threshold += 1
            continue
        threshold, direction = threshold_result

        resolution_dates = _get_resolution_dates(q)
        if not resolution_dates:
            skipped_no_horizon += 1
            continue

        for res_date in resolution_dates:
            horizon_days = _compute_horizon_days(q.freeze_datetime, res_date)
            if horizon_days is None:
                skipped_no_horizon += 1
                continue

            historical = _try_cached_historical(q)

            method = "naive"
            stat_prob: float | None = None

            if historical and len(historical) >= 5:
                cache_hits += 1
                rw = compute_random_walk_forecast(
                    historical, freeze_value, threshold, direction, horizon_days
                )
                if rw is not None:
                    stat_prob = rw
                    method = "random_walk"
                    used_random_walk += 1

            if stat_prob is None:
                stat_prob = compute_naive_forecast(freeze_value, threshold, direction)
                used_naive += 1

            entry: dict[str, object] = {
                "id": q.id,
                "source": source,
                "stat_prob": stat_prob,
                "outcome": q.outcome,
                "method": method,
                "horizon_days": horizon_days,
                "brier": brier_score(stat_prob, q.outcome),
            }
            results.append(entry)
            source_results[source].append(entry)

    total_timeseries = len(timeseries_qs)
    total_evaluated = len(results)
    coverage_pct = (total_evaluated / total_timeseries * 100) if total_timeseries > 0 else 0.0

    print("\n" + "=" * 72)
    print("STATISTICAL FORECASTER EVALUATION RESULTS")
    print("=" * 72)

    print("\nCoverage:")
    print(f"  Total timeseries questions:    {total_timeseries}")
    print(f"  Evaluated:                     {total_evaluated} ({coverage_pct:.1f}%)")
    print(f"  Skipped (no freeze value):     {skipped_no_freeze}")
    print(f"  Skipped (no threshold parsed): {skipped_no_threshold}")
    print(f"  Skipped (no horizon):          {skipped_no_horizon}")
    print(f"  Cache hits (historical data):  {cache_hits}")

    print("\nMethod breakdown:")
    print(f"  Random walk + CDF:  {used_random_walk}")
    print(f"  Naive heuristic:    {used_naive}")

    header = (
        f"\n{'Source':<12} {'N':>5} {'Brier':>8} {'Index':>8}"
        f" {'Naive .5 Brier':>15} {'Naive .5 Index':>15} {'Skill':>8}"
    )
    print(header)
    print("-" * 80)

    all_briers: list[float] = []
    all_naive_briers: list[float] = []

    for source in sorted(source_results.keys()):
        entries = source_results[source]
        n = len(entries)
        if n == 0:
            continue

        briers = [float(e["brier"]) for e in entries]
        mean_bs = sum(briers) / len(briers)
        bi = brier_index(mean_bs)

        naive_briers = [brier_score(0.5, int(e["outcome"])) for e in entries]
        naive_mean = sum(naive_briers) / len(naive_briers)
        naive_bi = brier_index(naive_mean)

        skill = 1.0 - (mean_bs / naive_mean) if naive_mean > 0 else 0.0

        all_briers.extend(briers)
        all_naive_briers.extend(naive_briers)

        print(
            f"{source:<12} {n:>5} {mean_bs:>8.4f} {bi:>8.2f}"
            f" {naive_mean:>15.4f} {naive_bi:>15.2f} {skill:>+8.3f}"
        )

    if all_briers:
        overall_mean = sum(all_briers) / len(all_briers)
        overall_bi = brier_index(overall_mean)
        naive_overall = sum(all_naive_briers) / len(all_naive_briers)
        naive_overall_bi = brier_index(naive_overall)
        overall_skill = 1.0 - (overall_mean / naive_overall) if naive_overall > 0 else 0.0

        print("-" * 80)
        print(
            f"{'OVERALL':<12} {len(all_briers):>5} {overall_mean:>8.4f} {overall_bi:>8.2f}"
            f" {naive_overall:>15.4f} {naive_overall_bi:>15.2f} {overall_skill:>+8.3f}"
        )

    rw_entries = [e for e in results if e["method"] == "random_walk"]
    naive_entries = [e for e in results if e["method"] == "naive"]

    if rw_entries:
        print("\nMethod comparison:")
        rw_briers = [float(e["brier"]) for e in rw_entries]
        rw_mean = sum(rw_briers) / len(rw_briers)
        rw_bi = brier_index(rw_mean)
        print(f"  Random walk ({len(rw_entries)} questions): Brier={rw_mean:.4f}  Index={rw_bi:.2f}")

    if naive_entries:
        if not rw_entries:
            print("\nMethod comparison:")
        naive_b = [float(e["brier"]) for e in naive_entries]
        naive_m = sum(naive_b) / len(naive_b)
        naive_bi_val = brier_index(naive_m)
        print(f"  Naive       ({len(naive_entries)} questions): Brier={naive_m:.4f}  Index={naive_bi_val:.2f}")

    if all_briers:
        print("\nCalibration check (binned):")
        cal_bins: dict[str, list[dict[str, object]]] = defaultdict(list)
        for e in results:
            p = float(e["stat_prob"])
            if p < 0.2:
                cal_bins["0.0-0.2"].append(e)
            elif p < 0.4:
                cal_bins["0.2-0.4"].append(e)
            elif p < 0.6:
                cal_bins["0.4-0.6"].append(e)
            elif p < 0.8:
                cal_bins["0.6-0.8"].append(e)
            else:
                cal_bins["0.8-1.0"].append(e)

        print(f"  {'Bin':<10} {'N':>5} {'Mean Prob':>10} {'Base Rate':>10}")
        for bin_name in ["0.0-0.2", "0.2-0.4", "0.4-0.6", "0.6-0.8", "0.8-1.0"]:
            if bin_name in cal_bins:
                b_entries = cal_bins[bin_name]
                mean_p = sum(float(e["stat_prob"]) for e in b_entries) / len(b_entries)
                base_rate = sum(int(e["outcome"]) for e in b_entries) / len(b_entries)
                print(f"  {bin_name:<10} {len(b_entries):>5} {mean_p:>10.3f} {base_rate:>10.3f}")

    print("\n" + "=" * 72)


if __name__ == "__main__":
    evaluate_statistical_forecaster()
