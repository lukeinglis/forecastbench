"""Error analysis for forecast evaluation: source grouping, calibration, and bias detection."""

from __future__ import annotations

import json
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
