"""Error analysis for forecast evaluation: source grouping, calibration, and bias detection."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from fetch_data import ResolvedQuestion
from score import brier_score, brier_index


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


def print_analysis(analysis: dict[str, Any]) -> None:
    if "by_source" in analysis:
        print("\n--- Performance by Source ---")
        for source, stats in analysis["by_source"].items():
            print(f"  {source:20s}  Brier={stats['brier']:.4f}  Index={stats['index']:.1f}%  n={stats['count']}")

    if "calibration" in analysis:
        print("\n--- Calibration ---")
        for b in analysis["calibration"]:
            print(
                f"  [{b['bin_low']:.1f}, {b['bin_high']:.1f})  "
                f"predicted={b['mean_predicted']:.3f}  observed={b['mean_observed']:.3f}  n={b['count']}"
            )

    if "biases" in analysis:
        b = analysis["biases"]
        print("\n--- Bias Analysis ---")
        print(f"  Mean forecast:  {b['mean_forecast']:.4f}")
        print(f"  Mean outcome:   {b['mean_outcome']:.4f}")
        direction = "optimistic" if b["bias"] > 0 else "pessimistic"
        print(f"  Bias:           {b['bias']:+.4f} ({direction})")
        if b["low_bin"]["count"] > 0:
            print(f"  Low  (<0.3):    predicted={b['low_bin']['mean_predicted']:.3f}  observed={b['low_bin']['mean_observed']:.3f}  n={b['low_bin']['count']}")
        if b["high_bin"]["count"] > 0:
            print(f"  High (>0.7):    predicted={b['high_bin']['mean_predicted']:.3f}  observed={b['high_bin']['mean_observed']:.3f}  n={b['high_bin']['count']}")


def save_analysis(analysis: dict[str, Any], path: str | Path) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(analysis, indent=2))


def compare_results(results_dir: str | Path = "results") -> None:
    """Print a comparison table of all saved results."""
    p = Path(results_dir)
    if not p.exists():
        print("No results directory found.")
        return

    files = sorted(p.glob("*.json"))
    if not files:
        print("No result files found.")
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
        print("No valid result files found.")
        return

    print(f"\n{'Model':<35s} {'Date':<11s} {'Brier':>7s} {'Index':>7s} {'DS Brier':>9s} {'MK Brier':>9s} {'N':>6s} {'Miss':>5s} {'Adj':>4s}")
    print("-" * 101)
    for r in sorted(rows, key=lambda x: x["overall_brier"]):
        n = r["n_dataset"] + r["n_market"]
        date = r["timestamp"][:8] if len(r["timestamp"]) >= 8 else r["timestamp"]
        print(
            f"{r['model']:<35s} {date:<11s} {r['overall_brier']:>7.4f} {r['overall_index']:>6.1f}% "
            f"{r['dataset_brier']:>9.4f} {r['market_brier']:>9.4f} {n:>6d} {r['n_missing']:>5d} "
            f"{'yes' if r['adjusted'] else 'no':>4s}"
        )
    print()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="ForecastBench analysis tools")
    parser.add_argument("--compare", action="store_true", help="Compare all saved results")
    parser.add_argument("--results-dir", default="results", help="Results directory")
    args = parser.parse_args()

    if args.compare:
        compare_results(args.results_dir)
    else:
        parser.print_help()
