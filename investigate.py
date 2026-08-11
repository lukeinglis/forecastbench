"""Diagnostic analysis of parity evaluation results.

Investigates dataset/market asymmetry, missing forecasts, multi-horizon ID
mismatches, knowledge cutoff effects, and superforecaster scoring gaps.
"""

from __future__ import annotations

import json
import math
import re
import sys
from pathlib import Path
from typing import Any

from fetch_data import MARKET_SOURCES

RESULTS_DIR = Path("results")
KNOWLEDGE_CUTOFF = "2025-03"
LEADERBOARD_SF = {"overall": 68.2, "dataset": 63.9, "market": 73.1}
LARGE_ROUND_DATES = {"2025-08-17", "2025-08-31"}
_DATE_RE = re.compile(r"\d{4}-\d{2}-\d{2}")


def load_result(path: Path) -> dict[str, Any]:
    data: dict[str, Any] = json.loads(path.read_text())
    return data


def load_all_results(results_dir: Path = RESULTS_DIR) -> list[dict[str, Any]]:
    if not results_dir.exists():
        return []
    results = []
    for p in sorted(results_dir.glob("*.json")):
        try:
            results.append(load_result(p))
        except (json.JSONDecodeError, KeyError):
            continue
    return results


def is_market_source(source: str) -> bool:
    return source.lower() in MARKET_SOURCES


def classify_source(source: str) -> str:
    return "market" if is_market_source(source) else "dataset"


def brier_score(forecast: float, outcome: int) -> float:
    return (forecast - outcome) ** 2


def brier_index(mean_brier: float) -> float:
    return (1.0 - math.sqrt(mean_brier)) * 100.0


def extract_date_suffix(key: str) -> str | None:
    m = _DATE_RE.findall(key)
    return m[-1] if m else None


def diagnose_id_mismatch(result: dict[str, Any]) -> dict[str, Any]:
    """Compare forecast keys against outcome keys to find multi-horizon mismatches."""
    forecasts: dict[str, float] = result.get("forecasts", {})
    outcomes: dict[str, int] = result.get("outcomes", {})
    sources: dict[str, str] = result.get("sources", {})

    forecast_keys = set(forecasts.keys())
    outcome_keys = set(outcomes.keys())

    matched = forecast_keys & outcome_keys
    forecast_only = forecast_keys - outcome_keys
    outcome_only = outcome_keys - forecast_keys

    forecast_only_by_type: dict[str, list[str]] = {"dataset": [], "market": []}
    outcome_only_by_type: dict[str, list[str]] = {"dataset": [], "market": []}

    for key in sorted(forecast_only):
        src = sources.get(key, "")
        cat = classify_source(src) if src else _guess_category(key, sources)
        forecast_only_by_type[cat].append(key)

    for key in sorted(outcome_only):
        src = sources.get(key, "")
        cat = classify_source(src) if src else _guess_category(key, sources)
        outcome_only_by_type[cat].append(key)

    date_mismatches: list[dict[str, str | None]] = []
    base_ids_seen: dict[str, dict[str, str | None]] = {}
    for key in sorted(forecast_only | outcome_only):
        date = extract_date_suffix(key)
        if date is None:
            continue
        base = key.rsplit("_", 1)[0] if date else key
        if base not in base_ids_seen:
            base_ids_seen[base] = {}
        side = "forecast" if key in forecast_only else "outcome"
        base_ids_seen[base][side + "_date"] = date

    for base, dates in sorted(base_ids_seen.items()):
        if "forecast_date" in dates and "outcome_date" in dates:
            date_mismatches.append({
                "base_id": base,
                "forecast_date": dates["forecast_date"],
                "outcome_date": dates["outcome_date"],
            })

    return {
        "total_forecasts": len(forecast_keys),
        "total_outcomes": len(outcome_keys),
        "matched": len(matched),
        "forecast_only": len(forecast_only),
        "outcome_only": len(outcome_only),
        "forecast_only_dataset": len(forecast_only_by_type["dataset"]),
        "forecast_only_market": len(forecast_only_by_type["market"]),
        "outcome_only_dataset": len(outcome_only_by_type["dataset"]),
        "outcome_only_market": len(outcome_only_by_type["market"]),
        "date_mismatches": date_mismatches[:20],
        "date_mismatch_count": len(date_mismatches),
        "sample_forecast_only": sorted(forecast_only)[:10],
        "sample_outcome_only": sorted(outcome_only)[:10],
    }


def _guess_category(key: str, sources: dict[str, str]) -> str:
    base = key.rsplit("_", 1)[0] if extract_date_suffix(key) else key
    src = sources.get(base, "")
    if src:
        return classify_source(src)
    for full_key, s in sources.items():
        if full_key.startswith(base):
            return classify_source(s)
    return "dataset"


def stratify_by_source(result: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Per-source statistics: counts, missing rates, Brier scores."""
    forecasts: dict[str, float] = result.get("forecasts", {})
    outcomes: dict[str, int] = result.get("outcomes", {})
    sources: dict[str, str] = result.get("sources", {})

    per_source: dict[str, dict[str, Any]] = {}

    for qid, outcome in outcomes.items():
        src = sources.get(qid, "unknown").lower()
        if src not in per_source:
            per_source[src] = {
                "category": classify_source(src),
                "forecast_count": 0,
                "outcome_count": 0,
                "missing_count": 0,
                "brier_scores": [],
            }
        entry = per_source[src]
        entry["outcome_count"] += 1
        if qid in forecasts:
            entry["forecast_count"] += 1
            entry["brier_scores"].append(brier_score(forecasts[qid], outcome))
        else:
            entry["missing_count"] += 1
            entry["brier_scores"].append(brier_score(0.5, outcome))

    for src in sorted(set(sources.get(k, "unknown").lower() for k in forecasts if k not in outcomes)):
        if src not in per_source:
            per_source[src] = {
                "category": classify_source(src),
                "forecast_count": 0,
                "outcome_count": 0,
                "missing_count": 0,
                "brier_scores": [],
            }
        per_source[src]["forecast_count"] += sum(
            1 for k in forecasts if k not in outcomes and sources.get(k, "unknown").lower() == src
        )

    result_table: dict[str, dict[str, Any]] = {}
    for src, data in sorted(per_source.items()):
        scores = data["brier_scores"]
        mean_bs = sum(scores) / len(scores) if scores else 0.0
        result_table[src] = {
            "category": data["category"],
            "forecast_count": data["forecast_count"],
            "outcome_count": data["outcome_count"],
            "missing_count": data["missing_count"],
            "missing_rate": data["missing_count"] / data["outcome_count"] if data["outcome_count"] else 0.0,
            "mean_brier": mean_bs,
            "brier_index": brier_index(mean_bs) if scores else 0.0,
        }

    return result_table


def extract_round_date(result: dict[str, Any]) -> str | None:
    meta = result.get("metadata", {})
    round_name = meta.get("round")
    if round_name:
        m = _DATE_RE.search(round_name)
        return m.group(0) if m else round_name
    return None


def per_round_breakdown(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Per-round statistics: counts, splits, scores, size category."""
    rows: list[dict[str, Any]] = []
    for result in results:
        round_date = extract_round_date(result)
        if round_date is None:
            continue

        outcomes: dict[str, int] = result.get("outcomes", {})
        forecasts: dict[str, float] = result.get("forecasts", {})
        sources: dict[str, str] = result.get("sources", {})
        sr = result.get("scoring_result", {})

        total = len(outcomes)
        n_dataset = 0
        n_market = 0
        for qid in outcomes:
            src = sources.get(qid, "unknown").lower()
            if is_market_source(src):
                n_market += 1
            else:
                n_dataset += 1

        missing = sum(1 for qid in outcomes if qid not in forecasts)
        missing_rate = missing / total if total else 0.0

        size_category = "1000q" if round_date in LARGE_ROUND_DATES else "500q"

        rows.append({
            "round_date": round_date,
            "total": total,
            "n_dataset": n_dataset,
            "n_market": n_market,
            "missing": missing,
            "missing_rate": missing_rate,
            "dataset_index": sr.get("dataset_index", 0.0),
            "market_index": sr.get("market_index", 0.0),
            "overall_index": sr.get("overall_index", 0.0),
            "size_category": size_category,
        })

    rows.sort(key=lambda r: r["round_date"])
    return rows


def knowledge_cutoff_analysis(
    result: dict[str, Any],
    cutoff: str = KNOWLEDGE_CUTOFF,
) -> dict[str, Any]:
    """Compare scores for questions resolving before vs after the knowledge cutoff."""
    forecasts: dict[str, float] = result.get("forecasts", {})
    outcomes: dict[str, int] = result.get("outcomes", {})
    sources: dict[str, str] = result.get("sources", {})

    pre_scores: list[float] = []
    post_scores: list[float] = []
    pre_market: list[float] = []
    post_market: list[float] = []

    for qid, outcome in outcomes.items():
        resolution_date = extract_date_suffix(qid)
        if resolution_date is None:
            continue

        prob = forecasts.get(qid, 0.5)
        bs = brier_score(prob, outcome)
        src = sources.get(qid, "unknown").lower()
        is_market = is_market_source(src)

        if resolution_date[:7] <= cutoff:
            pre_scores.append(bs)
            if is_market:
                pre_market.append(bs)
        else:
            post_scores.append(bs)
            if is_market:
                post_market.append(bs)

    def _stats(scores: list[float]) -> dict[str, Any]:
        if not scores:
            return {"count": 0, "mean_brier": 0.0, "brier_index": 0.0}
        mean = sum(scores) / len(scores)
        return {"count": len(scores), "mean_brier": mean, "brier_index": brier_index(mean)}

    return {
        "cutoff": cutoff,
        "pre_cutoff_all": _stats(pre_scores),
        "post_cutoff_all": _stats(post_scores),
        "pre_cutoff_market": _stats(pre_market),
        "post_cutoff_market": _stats(post_market),
    }


def superforecaster_gap(result: dict[str, Any]) -> dict[str, Any]:
    """Compute gap vs leaderboard superforecaster medians."""
    sr = result.get("scoring_result", {})
    our_overall = sr.get("overall_index", 0.0)
    our_dataset = sr.get("dataset_index", 0.0)
    our_market = sr.get("market_index", 0.0)

    return {
        "our_overall": our_overall,
        "our_dataset": our_dataset,
        "our_market": our_market,
        "lb_overall": LEADERBOARD_SF["overall"],
        "lb_dataset": LEADERBOARD_SF["dataset"],
        "lb_market": LEADERBOARD_SF["market"],
        "gap_overall": our_overall - LEADERBOARD_SF["overall"],
        "gap_dataset": our_dataset - LEADERBOARD_SF["dataset"],
        "gap_market": our_market - LEADERBOARD_SF["market"],
        "per_source": _sf_gap_by_source(result),
    }


def _sf_gap_by_source(result: dict[str, Any]) -> dict[str, dict[str, Any]]:
    source_table = stratify_by_source(result)
    per_source: dict[str, dict[str, Any]] = {}
    for src, data in source_table.items():
        per_source[src] = {
            "category": data["category"],
            "brier_index": data["brier_index"],
            "count": data["outcome_count"],
        }
    return per_source


def compare_round_sizes(rounds: list[dict[str, Any]]) -> dict[str, Any]:
    """Compare 1000q vs 500q rounds on composition and scores."""
    large: list[dict[str, Any]] = []
    small: list[dict[str, Any]] = []

    for r in rounds:
        if r["size_category"] == "1000q":
            large.append(r)
        else:
            small.append(r)

    def _agg(group: list[dict[str, Any]]) -> dict[str, Any]:
        if not group:
            return {"count": 0}
        n = len(group)
        return {
            "count": n,
            "avg_total": sum(r["total"] for r in group) / n,
            "avg_dataset_pct": sum(r["n_dataset"] / r["total"] * 100 for r in group) / n if all(r["total"] for r in group) else 0.0,
            "avg_market_pct": sum(r["n_market"] / r["total"] * 100 for r in group) / n if all(r["total"] for r in group) else 0.0,
            "avg_missing_rate": sum(r["missing_rate"] for r in group) / n,
            "avg_overall_index": sum(r["overall_index"] for r in group) / n,
            "avg_dataset_index": sum(r["dataset_index"] for r in group) / n,
            "avg_market_index": sum(r["market_index"] for r in group) / n,
            "rounds": [r["round_date"] for r in group],
        }

    return {
        "large_rounds": _agg(large),
        "small_rounds": _agg(small),
    }


def run_investigation(results: list[dict[str, Any]]) -> dict[str, Any]:
    """Run all 7 analyses and produce a structured report."""
    report: dict[str, Any] = {"analyses": {}, "summary": {}}

    if not results:
        report["summary"]["error"] = "No result files found"
        return report

    all_mismatch: list[dict[str, Any]] = []
    all_sources: dict[str, dict[str, Any]] = {}
    all_cutoff: list[dict[str, Any]] = []
    all_sf_gap: list[dict[str, Any]] = []

    for result in results:
        mismatch = diagnose_id_mismatch(result)
        all_mismatch.append(mismatch)

        sources = stratify_by_source(result)
        for src, data in sources.items():
            if src not in all_sources:
                all_sources[src] = {
                    "category": data["category"],
                    "total_forecast": 0,
                    "total_outcome": 0,
                    "total_missing": 0,
                    "all_brier_indices": [],
                }
            agg = all_sources[src]
            agg["total_forecast"] += data["forecast_count"]
            agg["total_outcome"] += data["outcome_count"]
            agg["total_missing"] += data["missing_count"]
            agg["all_brier_indices"].append(data["brier_index"])

        cutoff = knowledge_cutoff_analysis(result)
        all_cutoff.append(cutoff)

        sf = superforecaster_gap(result)
        all_sf_gap.append(sf)

    total_mismatched = sum(m["forecast_only"] + m["outcome_only"] for m in all_mismatch)
    total_forecasts = sum(m["total_forecasts"] for m in all_mismatch)
    total_outcomes = sum(m["total_outcomes"] for m in all_mismatch)
    total_matched = sum(m["matched"] for m in all_mismatch)

    report["analyses"]["id_mismatch"] = {
        "total_forecasts": total_forecasts,
        "total_outcomes": total_outcomes,
        "total_matched": total_matched,
        "total_mismatched": total_mismatched,
        "per_file": all_mismatch,
    }

    source_summary: dict[str, dict[str, Any]] = {}
    for src, agg in sorted(all_sources.items()):
        indices = agg["all_brier_indices"]
        source_summary[src] = {
            "category": agg["category"],
            "total_forecast": agg["total_forecast"],
            "total_outcome": agg["total_outcome"],
            "total_missing": agg["total_missing"],
            "missing_rate": agg["total_missing"] / agg["total_outcome"] if agg["total_outcome"] else 0.0,
            "mean_brier_index": sum(indices) / len(indices) if indices else 0.0,
        }
    report["analyses"]["source_stratification"] = source_summary

    rounds = per_round_breakdown(results)
    report["analyses"]["per_round"] = rounds

    if all_cutoff:
        pre_all = [c["pre_cutoff_all"] for c in all_cutoff if c["pre_cutoff_all"]["count"]]
        post_all = [c["post_cutoff_all"] for c in all_cutoff if c["post_cutoff_all"]["count"]]
        pre_market = [c["pre_cutoff_market"] for c in all_cutoff if c["pre_cutoff_market"]["count"]]
        post_market = [c["post_cutoff_market"] for c in all_cutoff if c["post_cutoff_market"]["count"]]
        report["analyses"]["knowledge_cutoff"] = {
            "cutoff": KNOWLEDGE_CUTOFF,
            "pre_cutoff_all_avg_index": _avg_field(pre_all, "brier_index"),
            "post_cutoff_all_avg_index": _avg_field(post_all, "brier_index"),
            "pre_cutoff_market_avg_index": _avg_field(pre_market, "brier_index"),
            "post_cutoff_market_avg_index": _avg_field(post_market, "brier_index"),
            "pre_cutoff_count": sum(c["count"] for c in pre_all),
            "post_cutoff_count": sum(c["count"] for c in post_all),
        }

    if all_sf_gap:
        avg_gap = sum(s["gap_overall"] for s in all_sf_gap) / len(all_sf_gap)
        report["analyses"]["superforecaster_gap"] = {
            "leaderboard": LEADERBOARD_SF,
            "avg_gap_overall": avg_gap,
            "per_file": all_sf_gap,
        }

    round_comparison = compare_round_sizes(rounds)
    report["analyses"]["round_size_comparison"] = round_comparison

    report["summary"] = _build_summary(report)

    return report


def _avg_field(items: list[dict[str, Any]], field: str) -> float:
    if not items:
        return 0.0
    total: float = sum(float(item[field]) for item in items)
    return total / len(items)


def _build_summary(report: dict[str, Any]) -> dict[str, Any]:
    analyses = report["analyses"]
    findings: list[str] = []
    recommendations: list[str] = []

    mismatch = analyses.get("id_mismatch", {})
    if mismatch.get("total_mismatched", 0) > 0:
        pct = mismatch["total_mismatched"] / (mismatch["total_forecasts"] + mismatch["total_outcomes"]) * 100 if (mismatch["total_forecasts"] + mismatch["total_outcomes"]) else 0
        findings.append(
            f"ID MISMATCH: {mismatch['total_mismatched']} keys unmatched "
            f"({pct:.1f}% of all keys). {mismatch['total_matched']} matched."
        )
        recommendations.append(
            "Root-cause the ID mismatch: forecast composite keys use horizon dates "
            "but outcome keys use resolution dates. Fix likely in eval.py "
            "or forecastbench-parity (join_resolved_questions). "
            "If parity package, file an issue."
        )

    sources = analyses.get("source_stratification", {})
    high_missing = [
        (src, d["missing_rate"]) for src, d in sources.items()
        if d["missing_rate"] > 0.3
    ]
    if high_missing:
        findings.append(
            f"HIGH MISSING RATE: {len(high_missing)} sources have >30% missing: "
            + ", ".join(f"{s} ({r:.0%})" for s, r in sorted(high_missing, key=lambda x: -x[1]))
        )

    cutoff = analyses.get("knowledge_cutoff", {})
    pre_idx = cutoff.get("pre_cutoff_market_avg_index", 0)
    post_idx = cutoff.get("post_cutoff_market_avg_index", 0)
    if pre_idx and post_idx and pre_idx - post_idx > 2:
        findings.append(
            f"CUTOFF EFFECT: Pre-cutoff market Brier Index {pre_idx:.1f} vs "
            f"post-cutoff {post_idx:.1f} (delta {pre_idx - post_idx:+.1f}). "
            f"Possible memorization advantage."
        )

    sf = analyses.get("superforecaster_gap", {})
    avg_gap = sf.get("avg_gap_overall", 0)
    if avg_gap:
        findings.append(
            f"SF GAP: Average gap vs superforecasters: {avg_gap:+.1f}pt overall."
        )
        if avg_gap < -1:
            recommendations.append(
                "Investigate Part A -2.1pt gap: check if leaderboard uses "
                "difficulty-adjusted scores while we use --raw."
            )

    rc = analyses.get("round_size_comparison", {})
    large = rc.get("large_rounds", {})
    small = rc.get("small_rounds", {})
    if large.get("count") and small.get("count"):
        diff = large.get("avg_overall_index", 0) - small.get("avg_overall_index", 0)
        if abs(diff) > 3:
            findings.append(
                f"ROUND SIZE: 1000q rounds avg {large['avg_overall_index']:.1f} vs "
                f"500q rounds avg {small['avg_overall_index']:.1f} (delta {diff:+.1f}pt)"
            )

    if not findings:
        findings.append("No significant anomalies detected.")

    return {"findings": findings, "recommendations": recommendations}


def format_report(report: dict[str, Any]) -> str:
    """Format investigation report for stdout."""
    lines: list[str] = []
    analyses = report.get("analyses", {})

    lines.append("=" * 72)
    lines.append("PARITY INVESTIGATION REPORT")
    lines.append("=" * 72)

    mismatch = analyses.get("id_mismatch", {})
    if mismatch:
        lines.append("")
        lines.append("1. MULTI-HORIZON ID MISMATCH DIAGNOSIS")
        lines.append("-" * 40)
        lines.append(f"  Total forecast keys:  {mismatch.get('total_forecasts', 0):,}")
        lines.append(f"  Total outcome keys:   {mismatch.get('total_outcomes', 0):,}")
        lines.append(f"  Matched:              {mismatch.get('total_matched', 0):,}")
        lines.append(f"  Forecast-only:        {mismatch.get('total_mismatched', 0) - mismatch.get('total_matched', 0):,}")
        lines.append(f"  Unmatched total:      {mismatch.get('total_mismatched', 0):,}")

        for i, pf in enumerate(mismatch.get("per_file", [])[:5]):
            lines.append(f"  File {i+1}: matched={pf['matched']}, "
                         f"forecast_only={pf['forecast_only']}, "
                         f"outcome_only={pf['outcome_only']}")
            if pf.get("date_mismatches"):
                for dm in pf["date_mismatches"][:3]:
                    lines.append(f"    Mismatch: {dm['base_id']} → "
                                 f"forecast={dm.get('forecast_date')}, "
                                 f"outcome={dm.get('outcome_date')}")
            if pf.get("sample_forecast_only"):
                lines.append(f"    Sample forecast-only: {pf['sample_forecast_only'][:5]}")
            if pf.get("sample_outcome_only"):
                lines.append(f"    Sample outcome-only:  {pf['sample_outcome_only'][:5]}")

    source_strat = analyses.get("source_stratification", {})
    if source_strat:
        lines.append("")
        lines.append("2. SOURCE-LEVEL STRATIFICATION")
        lines.append("-" * 40)
        lines.append(f"  {'Source':<14s} {'Cat':<8s} {'Forecasts':>10s} {'Outcomes':>9s} "
                     f"{'Missing':>8s} {'Miss%':>6s} {'BI':>6s}")
        lines.append(f"  {'-'*14} {'-'*8} {'-'*10} {'-'*9} {'-'*8} {'-'*6} {'-'*6}")
        for src, data in sorted(source_strat.items()):
            lines.append(
                f"  {src:<14s} {data['category']:<8s} {data['total_forecast']:>10,} "
                f"{data['total_outcome']:>9,} {data['total_missing']:>8,} "
                f"{data['missing_rate']:>5.0%} {data['mean_brier_index']:>5.1f}"
            )

    rounds = analyses.get("per_round", [])
    if rounds:
        lines.append("")
        lines.append("3. PER-ROUND BREAKDOWN")
        lines.append("-" * 40)
        lines.append(f"  {'Round':<12s} {'Total':>6s} {'DS':>5s} {'MK':>5s} "
                     f"{'Miss%':>6s} {'DS BI':>6s} {'MK BI':>6s} {'OA BI':>6s} {'Size':>5s}")
        lines.append(f"  {'-'*12} {'-'*6} {'-'*5} {'-'*5} "
                     f"{'-'*6} {'-'*6} {'-'*6} {'-'*6} {'-'*5}")
        for r in rounds:
            lines.append(
                f"  {r['round_date']:<12s} {r['total']:>6d} {r['n_dataset']:>5d} "
                f"{r['n_market']:>5d} {r['missing_rate']:>5.0%} "
                f"{r['dataset_index']:>5.1f} {r['market_index']:>5.1f} "
                f"{r['overall_index']:>5.1f} {r['size_category']:>5s}"
            )

    cutoff = analyses.get("knowledge_cutoff", {})
    if cutoff:
        lines.append("")
        lines.append("4. KNOWLEDGE CUTOFF ANALYSIS")
        lines.append("-" * 40)
        lines.append(f"  Cutoff: {cutoff.get('cutoff', KNOWLEDGE_CUTOFF)}")
        lines.append(f"  Pre-cutoff  (all):    n={cutoff.get('pre_cutoff_count', 0):,}, "
                     f"avg BI={cutoff.get('pre_cutoff_all_avg_index', 0):.1f}")
        lines.append(f"  Post-cutoff (all):    n={cutoff.get('post_cutoff_count', 0):,}, "
                     f"avg BI={cutoff.get('post_cutoff_all_avg_index', 0):.1f}")
        lines.append(f"  Pre-cutoff  (market): avg BI={cutoff.get('pre_cutoff_market_avg_index', 0):.1f}")
        lines.append(f"  Post-cutoff (market): avg BI={cutoff.get('post_cutoff_market_avg_index', 0):.1f}")

    sf = analyses.get("superforecaster_gap", {})
    if sf:
        lines.append("")
        lines.append("5. SUPERFORECASTER GAP ANALYSIS")
        lines.append("-" * 40)
        lb = sf.get("leaderboard", LEADERBOARD_SF)
        lines.append(f"  Leaderboard SF: Overall={lb['overall']}, "
                     f"Dataset={lb['dataset']}, Market={lb['market']}")
        lines.append(f"  Avg gap overall: {sf.get('avg_gap_overall', 0):+.1f}pt")
        for i, pf in enumerate(sf.get("per_file", [])[:5]):
            lines.append(f"  File {i+1}: overall={pf['our_overall']:.1f} "
                         f"(gap {pf['gap_overall']:+.1f}), "
                         f"dataset={pf['our_dataset']:.1f} "
                         f"(gap {pf['gap_dataset']:+.1f}), "
                         f"market={pf['our_market']:.1f} "
                         f"(gap {pf['gap_market']:+.1f})")

    rc = analyses.get("round_size_comparison", {})
    if rc:
        lines.append("")
        lines.append("6. 1000q VS 500q ROUND COMPARISON")
        lines.append("-" * 40)
        large = rc.get("large_rounds", {})
        small = rc.get("small_rounds", {})
        if large.get("count"):
            lines.append(f"  1000q rounds ({large['count']}): "
                         f"avg total={large.get('avg_total', 0):.0f}, "
                         f"dataset={large.get('avg_dataset_pct', 0):.1f}%, "
                         f"market={large.get('avg_market_pct', 0):.1f}%, "
                         f"avg OA BI={large.get('avg_overall_index', 0):.1f}")
        else:
            lines.append("  No 1000q rounds found.")
        if small.get("count"):
            lines.append(f"  500q rounds  ({small['count']}): "
                         f"avg total={small.get('avg_total', 0):.0f}, "
                         f"dataset={small.get('avg_dataset_pct', 0):.1f}%, "
                         f"market={small.get('avg_market_pct', 0):.1f}%, "
                         f"avg OA BI={small.get('avg_overall_index', 0):.1f}")
        else:
            lines.append("  No 500q rounds found.")

    summary = report.get("summary", {})
    if summary:
        lines.append("")
        lines.append("7. FINDINGS & RECOMMENDATIONS")
        lines.append("=" * 40)
        for f in summary.get("findings", []):
            lines.append(f"  [FINDING] {f}")
        for r in summary.get("recommendations", []):
            lines.append(f"  [ACTION]  {r}")

    lines.append("")
    return "\n".join(lines)


def main() -> None:
    if len(sys.argv) > 1:
        paths = [Path(arg) for arg in sys.argv[1:]]
        results = []
        for p in paths:
            if p.is_file() and p.suffix == ".json":
                try:
                    results.append(load_result(p))
                except (json.JSONDecodeError, KeyError) as e:
                    print(f"Error loading {p}: {e}", file=sys.stderr)
            else:
                print(f"Skipping {p}: not a JSON file", file=sys.stderr)
    else:
        results = load_all_results()

    if not results:
        print("No result files found. Run eval first to generate results.", file=sys.stderr)
        sys.exit(1)

    report = run_investigation(results)
    print(format_report(report))


if __name__ == "__main__":
    main()
