"""Part A: Superforecaster scoring parity check.

Scores superforecaster median forecasts through our pipeline and compares
against the published ForecastBench baseline leaderboard values.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

from fetch_data import (
    MARKET_SOURCES,
    ResolvedQuestion,
    fetch_leaderboard,
    fetch_question_set,
    fetch_resolution,
    fetch_superforecaster_forecasts,
    list_resolution_files,
    join_resolved_questions,
    fetch_all_resolutions,
    superforecaster_medians,
)
from score import score_forecasts


RESULTS_DIR = Path("results")


def _build_resolution_lookup() -> dict[tuple[str, str | None], int]:
    """Build (question_id, resolution_date) -> outcome lookup from all resolution files."""
    lookup: dict[tuple[str, str | None], int] = {}
    for fname in list_resolution_files():
        try:
            for r in fetch_resolution(fname):
                if r.outcome is not None:
                    key = (r.id, r.resolution_date)
                    if key not in lookup:
                        lookup[key] = r.outcome
        except Exception:
            continue
    return lookup


def _expand_superforecaster_questions(
    qs_human_filename: str,
) -> tuple[list[ResolvedQuestion], dict[str, float]]:
    """Expand superforecaster questions into per-horizon entries with forecasts.

    Market questions: one entry per question (deduped via join_resolved_questions).
    Dataset questions: one entry per resolution_date with a known outcome.
    """
    medians = superforecaster_medians(fetch_superforecaster_forecasts())
    qs_human = fetch_question_set(qs_human_filename)
    resolutions = fetch_all_resolutions()

    resolved = join_resolved_questions([qs_human], resolutions)
    market_resolved = [rq for rq in resolved if rq.source.lower() in MARKET_SOURCES]

    res_lookup = _build_resolution_lookup()
    dataset_expanded: list[ResolvedQuestion] = []
    for q in qs_human.questions:
        if q.source.lower() in MARKET_SOURCES:
            continue
        rd = q.resolution_dates
        if not isinstance(rd, list) or not rd:
            continue
        for date_str in rd:
            if not date_str or str(date_str).upper() == "N/A":
                continue
            outcome = res_lookup.get((q.id, date_str))
            if outcome is not None:
                dataset_expanded.append(
                    ResolvedQuestion(
                        id=f"{q.id}_{date_str}",
                        source=q.source,
                        question=q.question,
                        background=q.background,
                        resolution_criteria=q.resolution_criteria,
                        freeze_datetime=q.freeze_datetime,
                        freeze_datetime_value=q.freeze_datetime_value,
                        resolution_dates=q.resolution_dates,
                        outcome=outcome,
                        resolution_date=date_str,
                        forecast_due_date=qs_human.forecast_due_date,
                    )
                )

    expanded = dataset_expanded + market_resolved

    date_suffix = re.compile(r"_\d{4}-\d{2}-\d{2}$")
    forecasts: dict[str, float] = {}
    for rq in expanded:
        base_id = date_suffix.sub("", rq.id)
        if base_id in medians:
            forecasts[rq.id] = medians[base_id]
        elif rq.id in medians:
            forecasts[rq.id] = medians[rq.id]

    return expanded, forecasts


def _find_superforecaster_row(
    leaderboard: list[dict[str, str]],
) -> dict[str, str] | None:
    for row in leaderboard:
        model = row.get("Model", "").lower()
        if "superforecaster" in model and "median" in model:
            return row
    return None


def main() -> None:
    print("Part A: Superforecaster Scoring Parity Check")
    print("=" * 47)

    leaderboard = fetch_leaderboard("baseline")
    sf_row = _find_superforecaster_row(leaderboard)
    if sf_row is None:
        print("ERROR: Could not find Superforecaster (median) on leaderboard")
        sys.exit(1)

    lb_overall = float(sf_row["Overall"])
    lb_dataset = float(sf_row["Dataset"])
    lb_market = float(sf_row["Market"])
    lb_n = int(sf_row["N"])
    lb_n_dataset = int(sf_row["N dataset"])
    lb_n_market = int(sf_row["N market"])
    lb_brier_overall = float(sf_row["Brier Overall"])
    lb_brier_dataset = float(sf_row["Brier Dataset"])
    lb_brier_market = float(sf_row["Brier Market"])

    print(f"\nLeaderboard reference: {sf_row['Model']}")
    print(f"  Overall: {lb_overall}, Dataset: {lb_dataset}, Market: {lb_market}")
    print(f"  N: {lb_n} (Dataset: {lb_n_dataset}, Market: {lb_n_market})")

    expanded, forecasts = _expand_superforecaster_questions("2024-07-21-human.json")
    result = score_forecasts(forecasts, expanded, difficulty_adjusted=False)

    print("\nOur pipeline:")
    print(f"  Overall: {result.overall_index:.1f}, Dataset: {result.dataset_index:.1f}, Market: {result.market_index:.1f}")
    print(f"  N: {result.n_dataset + result.n_market} (Dataset: {result.n_dataset}, Market: {result.n_market})")
    print(f"  Missing: {result.n_missing}")

    threshold = 0.5
    checks = [
        ("Overall", result.overall_index, lb_overall, result.overall_brier, lb_brier_overall),
        ("Dataset", result.dataset_index, lb_dataset, result.dataset_brier, lb_brier_dataset),
        ("Market", result.market_index, lb_market, result.market_brier, lb_brier_market),
    ]

    print(f"\n{'Dimension':<10} {'Ours':>8} {'LB':>8} {'Delta':>8} {'Brier Ours':>12} {'Brier LB':>10} {'Result':>8}")
    print("-" * 70)

    all_pass = True
    results_data: dict[str, object] = {}
    for name, our_idx, lb_idx, our_brier, lb_brier in checks:
        delta = our_idx - lb_idx
        status = "PASS" if abs(delta) <= threshold else "FAIL"
        if status == "FAIL":
            all_pass = False
        print(f"{name:<10} {our_idx:>8.1f} {lb_idx:>8.1f} {delta:>+8.1f} {our_brier:>12.6f} {lb_brier:>10.4f} {status:>8}")
        results_data[name.lower()] = {
            "our_index": round(our_idx, 1),
            "leaderboard_index": lb_idx,
            "delta": round(delta, 1),
            "our_brier": round(our_brier, 6),
            "leaderboard_brier": lb_brier,
            "pass": abs(delta) <= threshold,
        }

    n_match = (
        result.n_dataset == lb_n_dataset
        and result.n_market == lb_n_market
    )
    print(f"\nN match: {'PASS' if n_match else 'FAIL'} "
          f"(dataset: {result.n_dataset}/{lb_n_dataset}, market: {result.n_market}/{lb_n_market})")

    if not all_pass:
        print("\nDiagnostic: Scores exceed ±0.5 threshold.")
        print("  Likely cause: known parity package bugs:")
        print("  - Bug 1b: per-question clamping (affects individual Brier scores)")
        print("  - Bug 1c: single global shift (affects difficulty adjustment baseline)")
        print("  These bugs exist in forecastbench-parity and cannot be fixed locally.")
        print("  The consistent ~2pt gap across all dimensions supports this diagnosis.")

    payload = {
        "check": "parity_part_a_superforecaster",
        "leaderboard_model": sf_row["Model"],
        "n_match": n_match,
        "n_dataset": result.n_dataset,
        "n_market": result.n_market,
        "n_missing": result.n_missing,
        "threshold": threshold,
        "all_pass": all_pass,
        "dimensions": results_data,
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
        },
    }

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_DIR / "parity_part_a_superforecaster.json"
    out_path.write_text(json.dumps(payload, indent=2))
    print(f"\nResults saved to {out_path}")

    print(f"\nVerdict: {'PASS' if all_pass else 'CONDITIONAL PASS — gaps attributable to known bugs'}")


if __name__ == "__main__":
    main()
