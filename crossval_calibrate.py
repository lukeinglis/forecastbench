"""Cross-validation for calibration strategies.

Leave-one-round-out: fit calibration on N-1 rounds, evaluate on held-out round.
Tests Platt scaling, base-rate replacement, isotonic regression, and hybrids.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

from calibrate import (
    platt_calibrate as calibrate,
    fit_calibration,
    fit_beta_calibration,
    beta_calibrate,
    BetaCalibrationModel,
    fit_feature_platt,
    apply_feature_platt,
    FeaturePlattModel,
)
from score import brier_index, brier_score

RESULTS_DIR = Path("results")
TIMESERIES_SOURCES = {"fred", "dbnomics", "yfinance"}
MARKET_SOURCES = {"metaculus", "polymarket", "manifold", "infer"}


def load_all_results() -> list[dict]:
    results = []
    for p in sorted(RESULTS_DIR.glob("*.json")):
        data = json.loads(p.read_text())
        results.append(data)
    return results


def _pairs(forecasts: dict[str, float], outcomes: dict[str, int]) -> list[tuple[float, int]]:
    return [(forecasts[qid], outcomes[qid]) for qid in forecasts if qid in outcomes]


def _mean_brier(pairs: list[tuple[float, int]]) -> float:
    if not pairs:
        return float("nan")
    return sum(brier_score(f, o) for f, o in pairs) / len(pairs)


def _round_name(result: dict, idx: int) -> str:
    return result.get("metadata", {}).get("round", f"fold-{idx}")


def _merge_train(results: list[dict], skip_idx: int) -> tuple[dict[str, float], dict[str, int], dict[str, str]]:
    forecasts: dict[str, float] = {}
    outcomes: dict[str, int] = {}
    sources: dict[str, str] = {}
    for i, r in enumerate(results):
        if i == skip_idx:
            continue
        forecasts.update(r.get("forecasts", {}))
        outcomes.update(r.get("outcomes", {}))
        sources.update(r.get("sources", {}))
    return forecasts, outcomes, sources


def _source_base_rates(results: list[dict], skip_idx: int) -> dict[str, float]:
    by_source: dict[str, list[int]] = defaultdict(list)
    for i, r in enumerate(results):
        if i == skip_idx:
            continue
        for qid, outcome in r.get("outcomes", {}).items():
            src = r.get("sources", {}).get(qid, "")
            if src:
                by_source[src].append(outcome)
    return {src: sum(os_) / len(os_) for src, os_ in by_source.items() if os_}


def _by_source_scores(
    forecasts: dict[str, float],
    outcomes: dict[str, int],
    sources: dict[str, str],
) -> dict[str, list[tuple[float, int]]]:
    by_src: dict[str, list[tuple[float, int]]] = defaultdict(list)
    for qid in forecasts:
        if qid in outcomes and qid in sources:
            by_src[sources[qid]].append((forecasts[qid], outcomes[qid]))
    return dict(by_src)


def evaluate_platt(results: list[dict], min_samples: int = 10) -> list[dict]:
    folds = []
    for test_idx in range(len(results)):
        test = results[test_idx]
        test_sources = test.get("sources", {})
        if not test_sources:
            continue

        train_f, train_o, train_s = _merge_train(results, test_idx)
        params = fit_calibration(train_f, train_o, train_s, min_samples=min_samples)

        test_f = dict(test.get("forecasts", {}))
        test_o = test.get("outcomes", {})

        from calibrate import calibrate_forecasts
        calibrated = calibrate_forecasts(test_f, test_sources, params)

        uncal_bs = _mean_brier(_pairs(test_f, test_o))
        cal_bs = _mean_brier(_pairs(calibrated, test_o))

        folds.append({
            "round": _round_name(test, test_idx),
            "uncal_index": brier_index(uncal_bs),
            "cal_index": brier_index(cal_bs),
            "delta": brier_index(cal_bs) - brier_index(uncal_bs),
            "n": len([q for q in test_f if q in test_o]),
            "by_source": {
                src: {"index": brier_index(_mean_brier(pairs)), "n": len(pairs)}
                for src, pairs in _by_source_scores(calibrated, test_o, test_sources).items()
            },
        })
    return folds


def evaluate_base_rate(results: list[dict], sources_to_replace: set[str] | None = None) -> list[dict]:
    if sources_to_replace is None:
        sources_to_replace = TIMESERIES_SOURCES

    folds = []
    for test_idx in range(len(results)):
        test = results[test_idx]
        test_sources = test.get("sources", {})
        if not test_sources:
            continue

        base_rates = _source_base_rates(results, test_idx)
        test_f = dict(test.get("forecasts", {}))
        test_o = test.get("outcomes", {})

        replaced = dict(test_f)
        for qid in replaced:
            src = test_sources.get(qid, "")
            if src in sources_to_replace and src in base_rates:
                replaced[qid] = base_rates[src]

        uncal_bs = _mean_brier(_pairs(test_f, test_o))
        rep_bs = _mean_brier(_pairs(replaced, test_o))

        folds.append({
            "round": _round_name(test, test_idx),
            "uncal_index": brier_index(uncal_bs),
            "replaced_index": brier_index(rep_bs),
            "delta": brier_index(rep_bs) - brier_index(uncal_bs),
            "n": len([q for q in test_f if q in test_o]),
        })
    return folds


def evaluate_hybrid(results: list[dict], min_samples: int = 10) -> list[dict]:
    """Platt scaling for market/events, base-rate replacement for timeseries."""
    folds = []
    for test_idx in range(len(results)):
        test = results[test_idx]
        test_sources = test.get("sources", {})
        if not test_sources:
            continue

        train_f, train_o, train_s = _merge_train(results, test_idx)
        params = fit_calibration(train_f, train_o, train_s, min_samples=min_samples)
        base_rates = _source_base_rates(results, test_idx)

        test_f = dict(test.get("forecasts", {}))
        test_o = test.get("outcomes", {})

        hybrid: dict[str, float] = {}
        for qid, prob in test_f.items():
            src = test_sources.get(qid, "")
            if src in TIMESERIES_SOURCES and src in base_rates:
                hybrid[qid] = base_rates[src]
            else:
                src_params = params.get(src, params.get("_global", {"a": 1.0, "b": 0.0}))
                hybrid[qid] = calibrate(prob, src_params["a"], src_params["b"])

        uncal_bs = _mean_brier(_pairs(test_f, test_o))
        hyb_bs = _mean_brier(_pairs(hybrid, test_o))

        folds.append({
            "round": _round_name(test, test_idx),
            "uncal_index": brier_index(uncal_bs),
            "hybrid_index": brier_index(hyb_bs),
            "delta": brier_index(hyb_bs) - brier_index(uncal_bs),
            "n": len([q for q in test_f if q in test_o]),
            "by_source": {
                src: {"index": brier_index(_mean_brier(pairs)), "n": len(pairs)}
                for src, pairs in _by_source_scores(hybrid, test_o, test_sources).items()
            },
        })
    return folds


def evaluate_isotonic(results: list[dict]) -> list[dict] | None:
    try:
        from sklearn.isotonic import IsotonicRegression
    except ImportError:
        return None

    folds = []
    for test_idx in range(len(results)):
        test = results[test_idx]
        test_sources = test.get("sources", {})
        if not test_sources:
            continue

        train_f, train_o, train_s = _merge_train(results, test_idx)

        by_source: dict[str, tuple[list[float], list[int]]] = defaultdict(lambda: ([], []))
        all_train_f: list[float] = []
        all_train_o: list[int] = []
        for qid in train_f:
            if qid in train_o:
                all_train_f.append(train_f[qid])
                all_train_o.append(train_o[qid])
                src = train_s.get(qid, "_global")
                by_source[src][0].append(train_f[qid])
                by_source[src][1].append(train_o[qid])

        global_iso = IsotonicRegression(y_min=0.01, y_max=0.99, out_of_bounds="clip")
        if all_train_f:
            global_iso.fit(all_train_f, all_train_o)

        source_isos: dict[str, IsotonicRegression] = {}
        for src, (fs, os_) in by_source.items():
            if len(fs) >= 20:
                iso = IsotonicRegression(y_min=0.01, y_max=0.99, out_of_bounds="clip")
                iso.fit(fs, os_)
                source_isos[src] = iso

        test_f = dict(test.get("forecasts", {}))
        test_o = test.get("outcomes", {})

        calibrated: dict[str, float] = {}
        for qid, prob in test_f.items():
            src = test_sources.get(qid, "")
            iso = source_isos.get(src, global_iso)
            calibrated[qid] = float(iso.predict([prob])[0])

        uncal_bs = _mean_brier(_pairs(test_f, test_o))
        cal_bs = _mean_brier(_pairs(calibrated, test_o))

        folds.append({
            "round": _round_name(test, test_idx),
            "uncal_index": brier_index(uncal_bs),
            "iso_index": brier_index(cal_bs),
            "delta": brier_index(cal_bs) - brier_index(uncal_bs),
            "n": len([q for q in test_f if q in test_o]),
        })
    return folds


def evaluate_beta(results: list[dict]) -> list[dict]:
    """Leave-one-round-out cross-validation for beta calibration."""
    folds = []
    for test_idx in range(len(results)):
        test = results[test_idx]
        test_sources = test.get("sources", {})
        if not test_sources:
            continue

        train_f, train_o, train_s = _merge_train(results, test_idx)

        by_source: dict[str, tuple[list[float], list[int]]] = defaultdict(lambda: ([], []))
        all_train_f: list[float] = []
        all_train_o: list[int] = []
        for qid in train_f:
            if qid in train_o:
                all_train_f.append(train_f[qid])
                all_train_o.append(train_o[qid])
                src = train_s.get(qid, "_global")
                by_source[src][0].append(train_f[qid])
                by_source[src][1].append(train_o[qid])

        global_model = fit_beta_calibration(all_train_f, all_train_o)
        source_models: dict[str, BetaCalibrationModel] = {}
        for src, (fs, os_) in by_source.items():
            if len(fs) >= 10:
                source_models[src] = fit_beta_calibration(fs, os_)

        test_f = dict(test.get("forecasts", {}))
        test_o = test.get("outcomes", {})

        calibrated: dict[str, float] = {}
        for qid, prob in test_f.items():
            src = test_sources.get(qid, "")
            model = source_models.get(src, global_model)
            calibrated[qid] = beta_calibrate(prob, model)

        uncal_bs = _mean_brier(_pairs(test_f, test_o))
        cal_bs = _mean_brier(_pairs(calibrated, test_o))

        folds.append({
            "round": _round_name(test, test_idx),
            "uncal_index": brier_index(uncal_bs),
            "beta_index": brier_index(cal_bs),
            "delta": brier_index(cal_bs) - brier_index(uncal_bs),
            "n": len([q for q in test_f if q in test_o]),
            "by_source": {
                src: {"index": brier_index(_mean_brier(pairs)), "n": len(pairs)}
                for src, pairs in _by_source_scores(calibrated, test_o, test_sources).items()
            },
        })
    return folds


def evaluate_feature_platt(results: list[dict], min_samples: int = 10) -> list[dict]:
    """Leave-one-round-out cross-validation for feature-rich Platt scaling.

    Extracts horizon_days from result metadata when available. Threshold distance
    is set to 0 when not computable (non-timeseries questions).
    """
    folds = []
    for test_idx in range(len(results)):
        test = results[test_idx]
        test_sources = test.get("sources", {})
        if not test_sources:
            continue

        train_f, train_o, train_s = _merge_train(results, test_idx)

        by_source: dict[str, tuple[list[float], list[int], list[float], list[float]]] = (
            defaultdict(lambda: ([], [], [], []))
        )
        all_preds: list[float] = []
        all_outs: list[int] = []
        all_horizon: list[float] = []
        all_threshold: list[float] = []

        for i, r in enumerate(results):
            if i == test_idx:
                continue
            horizons = r.get("horizon_days", {})
            thresholds = r.get("threshold_distances", {})
            for qid, prob in r.get("forecasts", {}).items():
                if qid not in r.get("outcomes", {}):
                    continue
                outcome = r["outcomes"][qid]
                h = float(horizons.get(qid, 30))
                td = float(thresholds.get(qid, 0.0))
                src = train_s.get(qid, "_global")

                all_preds.append(prob)
                all_outs.append(outcome)
                all_horizon.append(h)
                all_threshold.append(td)

                by_source[src][0].append(prob)
                by_source[src][1].append(outcome)
                by_source[src][2].append(h)
                by_source[src][3].append(td)

        if not all_preds:
            all_preds = list(train_f.values())
            all_outs = [train_o[qid] for qid in train_f if qid in train_o]
            all_horizon = [30.0] * len(all_preds)
            all_threshold = [0.0] * len(all_preds)

        global_model = fit_feature_platt(
            all_preds, all_outs,
            {"horizon_days": all_horizon, "threshold_distance": all_threshold},
        )
        source_models: dict[str, FeaturePlattModel] = {}
        for src, (fs, os_, hs, tds) in by_source.items():
            if len(fs) >= min_samples:
                source_models[src] = fit_feature_platt(
                    fs, os_, {"horizon_days": hs, "threshold_distance": tds},
                )

        test_f = dict(test.get("forecasts", {}))
        test_o = test.get("outcomes", {})
        test_horizons = test.get("horizon_days", {})
        test_thresholds = test.get("threshold_distances", {})

        calibrated: dict[str, float] = {}
        for qid, prob in test_f.items():
            src = test_sources.get(qid, "")
            model = source_models.get(src, global_model)
            h = float(test_horizons.get(qid, 30))
            td = float(test_thresholds.get(qid, 0.0))
            calibrated[qid] = apply_feature_platt(
                [prob], model,
                {"horizon_days": [h], "threshold_distance": [td]},
            )[0]

        uncal_bs = _mean_brier(_pairs(test_f, test_o))
        cal_bs = _mean_brier(_pairs(calibrated, test_o))

        folds.append({
            "round": _round_name(test, test_idx),
            "uncal_index": brier_index(uncal_bs),
            "fplatt_index": brier_index(cal_bs),
            "delta": brier_index(cal_bs) - brier_index(uncal_bs),
            "n": len([q for q in test_f if q in test_o]),
            "by_source": {
                src: {"index": brier_index(_mean_brier(pairs)), "n": len(pairs)}
                for src, pairs in _by_source_scores(calibrated, test_o, test_sources).items()
            },
        })
    return folds


def _print_folds(folds: list[dict], cal_key: str) -> float:
    total_delta = 0.0
    for fold in folds:
        delta = fold["delta"]
        total_delta += delta
        print(f"  {fold['round']}: uncal={fold['uncal_index']:.1f}%, {cal_key}={fold[cal_key]:.1f}% (delta={delta:+.1f})")
        if "by_source" in fold:
            for src, info in sorted(fold["by_source"].items()):
                print(f"    {src}: {info['index']:.1f}% (N={info['n']})")
    avg_delta = total_delta / len(folds) if folds else 0.0
    print(f"  Average delta: {avg_delta:+.1f}%")
    return avg_delta


def crossval() -> None:
    results = load_all_results()
    if len(results) < 2:
        print("Need at least 2 result files for cross-validation")
        return

    print(f"Loaded {len(results)} rounds")
    for i, r in enumerate(results):
        n = len(r.get("forecasts", {}))
        has_src = bool(r.get("sources"))
        print(f"  [{i}] {_round_name(r, i)}: {n} questions, sources={'yes' if has_src else 'NO'}")
    print()

    configs: list[tuple[str, float]] = []

    # 1. Platt scaling with different min_samples
    for min_s in [5, 10, 20, 50]:
        label = f"platt_min{min_s}"
        print(f"--- {label} ---")
        folds = evaluate_platt(results, min_samples=min_s)
        if folds:
            avg = _print_folds(folds, "cal_index")
            configs.append((label, avg))
        print()

    # 2. Base-rate replacement (timeseries only)
    print("--- base_rate_timeseries ---")
    folds = evaluate_base_rate(results, TIMESERIES_SOURCES)
    if folds:
        avg = _print_folds(folds, "replaced_index")
        configs.append(("base_rate_timeseries", avg))
    print()

    # 3. Base-rate replacement (all sources)
    print("--- base_rate_all ---")
    folds = evaluate_base_rate(results, None)
    if folds:
        # replace all sources
        folds_all = []
        for test_idx in range(len(results)):
            test = results[test_idx]
            test_sources = test.get("sources", {})
            if not test_sources:
                continue
            base_rates = _source_base_rates(results, test_idx)
            test_f = dict(test.get("forecasts", {}))
            test_o = test.get("outcomes", {})
            replaced = dict(test_f)
            for qid in replaced:
                src = test_sources.get(qid, "")
                if src in base_rates:
                    replaced[qid] = base_rates[src]
            uncal_bs = _mean_brier(_pairs(test_f, test_o))
            rep_bs = _mean_brier(_pairs(replaced, test_o))
            folds_all.append({
                "round": _round_name(test, test_idx),
                "uncal_index": brier_index(uncal_bs),
                "replaced_index": brier_index(rep_bs),
                "delta": brier_index(rep_bs) - brier_index(uncal_bs),
                "n": len([q for q in test_f if q in test_o]),
            })
        avg = _print_folds(folds_all, "replaced_index")
        configs.append(("base_rate_all", avg))
    print()

    # 4. Hybrid: Platt for market/events, base-rate for timeseries
    for min_s in [5, 10, 20]:
        label = f"hybrid_min{min_s}"
        print(f"--- {label} ---")
        folds = evaluate_hybrid(results, min_samples=min_s)
        if folds:
            avg = _print_folds(folds, "hybrid_index")
            configs.append((label, avg))
        print()

    # 5. Isotonic regression
    print("--- isotonic ---")
    folds = evaluate_isotonic(results)
    if folds is None:
        print("  sklearn not available, skipping isotonic regression")
    elif folds:
        avg = _print_folds(folds, "iso_index")
        configs.append(("isotonic", avg))
    print()

    # 6. Beta calibration
    print("--- beta ---")
    folds = evaluate_beta(results)
    if folds:
        avg = _print_folds(folds, "beta_index")
        configs.append(("beta", avg))
    print()

    # 7. Feature-rich Platt scaling
    for min_s in [5, 10, 20]:
        label = f"feature_platt_min{min_s}"
        print(f"--- {label} ---")
        folds = evaluate_feature_platt(results, min_samples=min_s)
        if folds:
            avg = _print_folds(folds, "fplatt_index")
            configs.append((label, avg))
        print()

    # Summary
    print("=" * 60)
    print("SUMMARY: Average cross-validated delta (higher = better)")
    print("=" * 60)
    configs.sort(key=lambda x: x[1], reverse=True)
    for label, avg_delta in configs:
        marker = " <-- BEST" if configs and label == configs[0][0] else ""
        print(f"  {label:30s} {avg_delta:+.2f}%{marker}")


if __name__ == "__main__":
    crossval()
