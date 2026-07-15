"""Error analysis for forecast evaluation: source grouping, calibration, and bias detection."""

from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Any

from fetch_data import ResolvedQuestion
from logging_config import get_logger
from score import brier_score, brier_index, murphy_decomposition

logger = get_logger("analyze")


def analyze_by_source(
    forecasts: dict[str, float],
    resolved: list[ResolvedQuestion],
) -> dict[str, dict[str, object]]:
    by_source: dict[str, list[tuple[float, int]]] = {}
    for q in resolved:
        f = forecasts.get(q.id, 0.5)
        by_source.setdefault(q.source, []).append((f, q.outcome))

    results: dict[str, dict[str, object]] = {}
    for source, pairs in sorted(by_source.items()):
        bs = sum(brier_score(f, o) for f, o in pairs) / len(pairs)
        results[source] = {
            "brier": bs,
            "index": brier_index(bs),
            "count": len(pairs),
        }
    return results


def analyze_calibration(
    forecasts: dict[str, float],
    resolved: list[ResolvedQuestion],
    n_bins: int = 10,
) -> list[dict[str, object]]:
    pairs = [(forecasts.get(q.id, 0.5), q.outcome) for q in resolved]
    if not pairs:
        return []

    bin_width = 1.0 / n_bins
    bins: list[dict[str, object]] = []

    for i in range(n_bins):
        low = i * bin_width
        high = (i + 1) * bin_width
        in_bin = [(f, o) for f, o in pairs if low <= f < high or (i == n_bins - 1 and f == high)]
        if not in_bin:
            continue
        fs, os_ = zip(*in_bin)
        bins.append({
            "bin_low": low,
            "bin_high": high,
            "mean_predicted": sum(fs) / len(fs),
            "mean_observed": sum(os_) / len(os_),
            "count": len(in_bin),
        })
    return bins


def calibration_metrics(
    pairs: list[tuple[float, int]],
    n_bins: int = 10,
) -> dict[str, float]:
    """Compute calibration summary metrics: ECE, MCE, and sharpness.

    Reference:
        Gneiting, T. & Raftery, A. E. (2007). 'Strictly Proper Scoring
        Rules, Prediction, and Estimation.' Journal of the American
        Statistical Association, 102(477), 359-378.

    Args:
        pairs: List of (forecast_probability, binary_outcome) tuples.
        n_bins: Number of equally-spaced bins in [0, 1]. Default 10.

    Returns:
        Dict with keys: ece, mce, sharpness.
    """
    if not pairs:
        return {"ece": 0.0, "mce": 0.0, "sharpness": 0.0}

    n = len(pairs)
    forecasts_list = [f for f, _ in pairs]
    mean_f = sum(forecasts_list) / n
    sharpness = sum((f - mean_f) ** 2 for f in forecasts_list) / n

    bin_width = 1.0 / n_bins
    ece = 0.0
    mce = 0.0
    for i in range(n_bins):
        low = i * bin_width
        high = (i + 1) * bin_width
        in_bin = [
            (f, o) for f, o in pairs
            if low <= f < high or (i == n_bins - 1 and f == high)
        ]
        if not in_bin:
            continue
        n_k = len(in_bin)
        f_k = sum(f for f, _ in in_bin) / n_k
        o_k = sum(o for _, o in in_bin) / n_k
        gap = abs(f_k - o_k)
        ece += (n_k / n) * gap
        mce = max(mce, gap)

    logger.info(
        "calibration_metrics",
        ece=round(ece, 6),
        mce=round(mce, 6),
        sharpness=round(sharpness, 6),
    )

    return {"ece": ece, "mce": mce, "sharpness": sharpness}


def analyze_biases(
    forecasts: dict[str, float],
    resolved: list[ResolvedQuestion],
) -> dict[str, object]:
    pairs = [(forecasts.get(q.id, 0.5), q.outcome) for q in resolved]
    if not pairs:
        return {"mean_forecast": 0.0, "mean_outcome": 0.0, "bias": 0.0, "low_bin": {}, "high_bin": {}}

    fs, os_ = zip(*pairs)
    mean_f = sum(fs) / len(fs)
    mean_o = sum(os_) / len(os_)

    low_pairs = [(f, o) for f, o in pairs if f < 0.3]
    high_pairs = [(f, o) for f, o in pairs if f > 0.7]

    def _bin_stats(bp: list[tuple[float, int]]) -> dict[str, object]:
        if not bp:
            return {"mean_predicted": 0.0, "mean_observed": 0.0, "count": 0, "brier": 0.0}
        bfs, bos = zip(*bp)
        bs = sum(brier_score(f, o) for f, o in bp) / len(bp)
        return {
            "mean_predicted": sum(bfs) / len(bfs),
            "mean_observed": sum(bos) / len(bos),
            "count": len(bp),
            "brier": bs,
        }

    return {
        "mean_forecast": mean_f,
        "mean_outcome": mean_o,
        "bias": mean_f - mean_o,
        "low_bin": _bin_stats(low_pairs),
        "high_bin": _bin_stats(high_pairs),
    }


def analyze_decomposition(
    forecasts: dict[str, float],
    resolved: list[ResolvedQuestion],
    n_bins: int = 10,
) -> dict[str, dict[str, float]]:
    """Run Murphy decomposition and calibration metrics on forecast/outcome pairs."""
    pairs = [(forecasts.get(q.id, 0.5), q.outcome) for q in resolved]
    if not pairs:
        return {"murphy": {}, "calibration": {}}

    murphy = murphy_decomposition(pairs, n_bins=n_bins)
    cal = calibration_metrics(pairs, n_bins=n_bins)
    return {"murphy": murphy, "calibration": cal}


def print_analysis(analysis: dict[str, Any]) -> None:
    if "by_source" in analysis:
        for source, stats in analysis["by_source"].items():
            logger.info(
                "source_performance",
                source=source,
                brier=round(stats["brier"], 4),
                index=round(stats["index"], 1),
                count=stats["count"],
            )

    if "calibration" in analysis:
        for b in analysis["calibration"]:
            logger.info(
                "calibration_bin",
                bin_low=b["bin_low"],
                bin_high=b["bin_high"],
                mean_predicted=round(b["mean_predicted"], 3),
                mean_observed=round(b["mean_observed"], 3),
                count=b["count"],
            )

    if "biases" in analysis:
        b = analysis["biases"]
        direction = "optimistic" if b["bias"] > 0 else "pessimistic"
        logger.info(
            "bias_analysis",
            mean_forecast=round(b["mean_forecast"], 4),
            mean_outcome=round(b["mean_outcome"], 4),
            bias=round(b["bias"], 4),
            direction=direction,
        )

    if "decomposition" in analysis:
        decomp = analysis["decomposition"]
        if "murphy" in decomp and decomp["murphy"]:
            m = decomp["murphy"]
            logger.info(
                "murphy_decomposition_result",
                reliability=round(m["reliability"], 6),
                resolution=round(m["resolution"], 6),
                uncertainty=round(m["uncertainty"], 6),
                brier_check=round(m["brier_check"], 6),
            )
        if "calibration" in decomp and decomp["calibration"]:
            c = decomp["calibration"]
            logger.info(
                "calibration_metrics_result",
                ece=round(c["ece"], 6),
                mce=round(c["mce"], 6),
                sharpness=round(c["sharpness"], 6),
            )


def save_analysis(analysis: dict[str, Any], path: str | Path) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(analysis, indent=2))


def analyze_worst_questions(
    forecasts: dict[str, float],
    resolved: list[ResolvedQuestion],
    top_n: int = 50,
) -> list[dict[str, object]]:
    """Find the N questions with highest individual Brier scores."""
    errors: list[dict[str, object]] = []
    for q in resolved:
        f = forecasts.get(q.id, 0.5)
        bs = brier_score(f, q.outcome)
        category = "neutral"
        if f > 0.7 and q.outcome == 0:
            category = "confident_wrong_positive"
        elif f < 0.3 and q.outcome == 1:
            category = "confident_wrong_negative"
        elif 0.4 <= f <= 0.6:
            category = "uncertain"
        errors.append({
            "id": q.id,
            "source": q.source,
            "question": q.question[:120],
            "forecast": f,
            "outcome": q.outcome,
            "brier": bs,
            "category": category,
        })
    errors.sort(key=lambda x: float(str(x["brier"])), reverse=True)
    return errors[:top_n]


def analyze_by_horizon(
    forecasts: dict[str, float],
    resolved: list[ResolvedQuestion],
) -> dict[str, dict[str, object]]:
    """Break down dataset question performance by resolution horizon."""
    horizon_pattern = re.compile(r"^(.+)_(\d{4}-\d{2}-\d{2})$")
    horizon_groups: dict[str, list[tuple[float, int]]] = {}

    for q in resolved:
        m = horizon_pattern.match(q.id)
        if m and q.resolution_date:
            horizon_groups.setdefault(q.resolution_date, []).append(
                (forecasts.get(q.id, 0.5), q.outcome)
            )

    if not horizon_groups:
        return {}

    results: dict[str, dict[str, object]] = {}
    for horizon, pairs in sorted(horizon_groups.items()):
        bs = sum(brier_score(f, o) for f, o in pairs) / len(pairs)
        results[horizon] = {
            "brier": bs,
            "index": brier_index(bs),
            "count": len(pairs),
        }
    return results


def compare_paired(
    result_a_path: str | Path,
    result_b_path: str | Path,
) -> dict[str, object]:
    """Paired comparison of two runs on shared questions."""
    data_a = json.loads(Path(result_a_path).read_text())
    data_b = json.loads(Path(result_b_path).read_text())
    forecasts_a: dict[str, float] = data_a["forecasts"]
    forecasts_b: dict[str, float] = data_b["forecasts"]

    outcomes_a: dict[str, int] = data_a.get("outcomes", {})
    outcomes_b: dict[str, int] = data_b.get("outcomes", {})

    if not outcomes_a and not outcomes_b:
        return {
            "error": "Result files missing 'outcomes' key. Re-run eval to generate results with outcomes.",
            "n_shared": 0,
        }

    outcomes = outcomes_a or outcomes_b
    shared_ids = set(forecasts_a.keys()) & set(forecasts_b.keys()) & set(outcomes.keys())
    if not shared_ids:
        return {"error": "No shared questions between runs", "n_shared": 0}

    sr_a = data_a["scoring_result"]
    sr_b = data_b["scoring_result"]

    diffs: list[float] = []
    a_wins = 0
    b_wins = 0
    for qid in sorted(shared_ids):
        outcome = outcomes[qid]
        bs_a = (forecasts_a[qid] - outcome) ** 2
        bs_b = (forecasts_b[qid] - outcome) ** 2
        diffs.append(bs_a - bs_b)
        if bs_a < bs_b:
            a_wins += 1
        elif bs_b < bs_a:
            b_wins += 1

    n = len(diffs)
    mean_diff = sum(diffs) / n
    var_diff = sum((d - mean_diff) ** 2 for d in diffs) / (n - 1) if n > 1 else 0.0
    se = math.sqrt(var_diff / n) if n > 1 else 0.0
    t_stat = mean_diff / se if se > 0 else 0.0

    return {
        "model_a": data_a["model_slug"],
        "model_b": data_b["model_slug"],
        "n_shared": n,
        "mean_brier_a": sr_a["overall_brier"],
        "mean_brier_b": sr_b["overall_brier"],
        "mean_diff": mean_diff,
        "std_err": se,
        "t_statistic": t_stat,
        "a_wins": a_wins,
        "b_wins": b_wins,
        "ties": n - a_wins - b_wins,
    }


def compare_results(results_dir: str | Path = "results") -> None:
    """Print a comparison table of all saved results."""
    p = Path(results_dir)
    if not p.exists():
        logger.warning("compare_results_no_dir", path=str(p))
        return

    files = sorted(p.glob("*.json"))
    if not files:
        logger.warning("compare_results_no_files", path=str(p))
        return

    rows: list[dict[str, Any]] = []
    for f in files:
        try:
            data = json.loads(f.read_text())
            sr = data["scoring_result"]
            rows.append({
                "timestamp": data["timestamp"],
                "model": data["model_slug"],
                "overall_brier": sr["overall_brier"],
                "overall_index": sr["overall_index"],
                "dataset_brier": sr["dataset_brier"],
                "market_brier": sr["market_brier"],
                "n_dataset": sr["n_dataset"],
                "n_market": sr["n_market"],
                "n_missing": sr["n_missing"],
                "adjusted": sr.get("difficulty_adjusted", False),
            })
        except (json.JSONDecodeError, KeyError):
            continue

    if not rows:
        logger.warning("compare_results_no_valid", path=str(p))
        return

    for r in sorted(rows, key=lambda x: x["overall_brier"]):
        logger.info(
            "compare_result",
            model=r["model"],
            timestamp=r["timestamp"],
            overall_brier=r["overall_brier"],
            overall_index=r["overall_index"],
            dataset_brier=r["dataset_brier"],
            market_brier=r["market_brier"],
            n=r["n_dataset"] + r["n_market"],
            n_missing=r["n_missing"],
            adjusted=r["adjusted"],
        )


def compare_by_round(results_dir: str | Path = "results") -> None:
    """Print a per-round score table grouped by round name."""
    p = Path(results_dir)
    if not p.exists():
        logger.warning("compare_by_round_no_dir", path=str(p))
        return

    files = sorted(p.glob("*.json"))
    if not files:
        logger.warning("compare_by_round_no_files", path=str(p))
        return

    rows: list[dict[str, Any]] = []
    for f in files:
        try:
            data = json.loads(f.read_text())
            sr = data["scoring_result"]
            meta = data.get("metadata", {})
            round_name = meta.get("round")
            if round_name is None:
                continue
            rows.append({
                "round": round_name,
                "model": data["model_slug"],
                "overall_brier": sr["overall_brier"],
                "overall_index": sr["overall_index"],
                "n": sr["n_dataset"] + sr["n_market"],
            })
        except (json.JSONDecodeError, KeyError):
            continue

    if not rows:
        print("No per-round results found. Run eval with --round to generate them.")
        return

    rows.sort(key=lambda x: (x["round"], x["overall_brier"]))
    print(f"\n{'Round':<25s} {'Model':<20s} {'Brier':>7s} {'Index':>7s} {'N':>5s}")
    print("-" * 68)
    for r in rows:
        print(
            f"{r['round']:<25s} {r['model']:<20s} "
            f"{r['overall_brier']:>7.3f} {r['overall_index']:>6.1f}% {r['n']:>5d}"
        )


def _load_result_forecasts(result_path: str | Path) -> tuple[dict[str, float], list[ResolvedQuestion]]:
    """Load forecasts from a result file and re-join with resolved questions."""
    from fetch_data import Resolution, load_data, join_resolved_questions

    data = json.loads(Path(result_path).read_text())
    forecasts: dict[str, float] = data["forecasts"]
    question_sets_used = data["metadata"]["question_sets_used"]

    all_qs, resolved = load_data()
    used_qs = [qs for qs in all_qs if qs.forecast_due_date in question_sets_used]
    resolutions = {
        q.id: Resolution(id=q.id, outcome=q.outcome, resolution_date=q.resolution_date)
        for q in resolved
    }
    iteration_resolved = join_resolved_questions(used_qs, resolutions)
    return forecasts, iteration_resolved


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="ForecastBench analysis tools")
    parser.add_argument("--compare", action="store_true", help="Compare all saved results")
    parser.add_argument("--by-round", action="store_true", help="Show per-round score comparison")
    parser.add_argument("--results-dir", default="results", help="Results directory")
    parser.add_argument("--worst", metavar="RESULT", help="Show worst questions from a result file")
    parser.add_argument("--horizons", metavar="RESULT", help="Show horizon breakdown from a result file")
    parser.add_argument("--decompose", metavar="RESULT", help="Show Brier decomposition from a result file")
    parser.add_argument("--versus", nargs=2, metavar=("A", "B"), help="Paired comparison of two result files")
    parser.add_argument("--top-n", type=int, default=50, help="Number of worst questions to show")
    args = parser.parse_args()

    if args.compare:
        compare_results(args.results_dir)
    elif args.by_round:
        compare_by_round(args.results_dir)
    elif args.worst:
        forecasts, resolved = _load_result_forecasts(args.worst)
        worst = analyze_worst_questions(forecasts, resolved, top_n=args.top_n)
        print(f"\nTop {len(worst)} Worst Questions:")
        print(f"{'Source':<12s} {'Forecast':>8s} {'Outcome':>7s} {'Brier':>7s} {'Category':<25s} {'Question'}")
        print("-" * 110)
        for w in worst:
            print(
                f"{w['source']:<12s} {w['forecast']:>8.3f} {w['outcome']:>7d} {w['brier']:>7.4f} "
                f"{w['category']:<25s} {w['question']}"
            )
    elif args.horizons:
        forecasts, resolved = _load_result_forecasts(args.horizons)
        horizons = analyze_by_horizon(forecasts, resolved)
        if not horizons:
            print("No multi-horizon questions found.")
        else:
            print(f"\n{'Horizon':<12s} {'Brier':>7s} {'Index':>7s} {'Count':>7s}")
            print("-" * 35)
            for h, stats in horizons.items():
                print(f"{h:<12s} {stats['brier']:>7.4f} {stats['index']:>6.1f}% {stats['count']:>7d}")
    elif args.decompose:
        forecasts, resolved = _load_result_forecasts(args.decompose)
        decomp = analyze_decomposition(forecasts, resolved)
        murphy = decomp["murphy"]
        cal = decomp["calibration"]
        print("\nBrier Score Decomposition (Murphy):")
        if murphy:
            print(f"  Reliability (calibration error): {murphy['reliability']:.6f}")
            print(f"  Resolution (discrimination):     {murphy['resolution']:.6f}")
            print(f"  Uncertainty (base rate):          {murphy['uncertainty']:.6f}")
            print(f"  Brier (rel - res + unc):          {murphy['brier_check']:.6f}")
        print("\nCalibration Metrics:")
        if cal:
            print(f"  ECE:                              {cal['ece']:.6f}")
            print(f"  MCE:                              {cal['mce']:.6f}")
            print(f"  Sharpness:                        {cal['sharpness']:.6f}")
    elif args.versus:
        result = compare_paired(args.versus[0], args.versus[1])
        if "error" in result:
            print(f"Error: {result['error']}")
        else:
            print(f"\nPaired Comparison: {result['model_a']} vs {result['model_b']}")
            print(f"  Shared questions:  {result['n_shared']}")
            print(f"  Mean Brier A:      {result['mean_brier_a']:.4f}")
            print(f"  Mean Brier B:      {result['mean_brier_b']:.4f}")
            print(f"  Mean diff (A-B):   {result['mean_diff']:+.6f}")
            print(f"  Std error:         {result['std_err']:.6f}")
            print(f"  t-statistic:       {result['t_statistic']:+.4f}")
            print(f"  A wins / B wins:   {result['a_wins']} / {result['b_wins']} (ties: {result['ties']})")
    else:
        parser.print_help()
