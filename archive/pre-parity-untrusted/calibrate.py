"""Calibration tools for forecast probabilities.

Provides two approaches:
1. Isotonic regression (PAVA) — learned from result files, applied per-source
2. Hierarchical Platt scaling — fitted from forecasts+outcomes, saved/loaded as params
"""

from __future__ import annotations

import json
import math
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

from logging_config import get_logger

logger = get_logger("calibrate")

CALIBRATION_DIR = Path(".cache/calibration")
MIN_DATA_POINTS = 20
MIN_SAMPLES_FOR_SOURCE = 10


# ---------------------------------------------------------------------------
# Isotonic regression (PAVA) — used by eval.py _apply_calibration
# ---------------------------------------------------------------------------

def isotonic_regression(
    predictions: list[float], outcomes: list[float],
) -> list[tuple[float, float]]:
    """Pool Adjacent Violators Algorithm (PAVA) for isotonic regression."""
    if not predictions or not outcomes:
        return []

    paired = sorted(zip(predictions, outcomes))
    blocks: list[tuple[float, float, int]] = [
        (x, y, 1) for x, y in paired
    ]

    merged = True
    while merged:
        merged = False
        new_blocks: list[tuple[float, float, int]] = []
        i = 0
        while i < len(blocks):
            if i + 1 < len(blocks) and blocks[i][1] > blocks[i + 1][1]:
                x1, y1, n1 = blocks[i]
                x2, y2, n2 = blocks[i + 1]
                combined_y = (y1 * n1 + y2 * n2) / (n1 + n2)
                combined_x = (x1 * n1 + x2 * n2) / (n1 + n2)
                new_blocks.append((combined_x, combined_y, n1 + n2))
                merged = True
                i += 2
            else:
                new_blocks.append(blocks[i])
                i += 1
        blocks = new_blocks

    return [(x, y) for x, y, _ in blocks]


def calibrate(
    probability: float,
    source: str,
    models: dict[str, list[tuple[float, float]]] | None = None,
) -> float:
    """Apply isotonic calibration to a single forecast probability."""
    if models is not None:
        breakpoints = models.get(source)
    else:
        breakpoints = _load_source_model(source)

    if not breakpoints:
        return probability

    calibrated = _interpolate(probability, breakpoints)
    return max(0.001, min(0.999, calibrated))


def load_calibration_models() -> dict[str, list[tuple[float, float]]]:
    """Load all isotonic calibration models from disk."""
    models: dict[str, list[tuple[float, float]]] = {}
    if not CALIBRATION_DIR.exists():
        return models

    for path in CALIBRATION_DIR.glob("*.json"):
        source = path.stem
        try:
            data = json.loads(path.read_text())
            if "breakpoints" in data:
                breakpoints = [(float(x), float(y)) for x, y in data["breakpoints"]]
                models[source] = breakpoints
        except (json.JSONDecodeError, KeyError, ValueError):
            logger.warning("calibration_load_failed", source=source)
    return models


def learn(result_path: str) -> None:
    """Learn isotonic calibration models from a result file."""
    from fetch_data import Resolution, load_data, join_resolved_questions

    data = json.loads(Path(result_path).read_text())
    forecasts: dict[str, float] = data["forecasts"]
    outcomes: dict[str, int] = data.get("outcomes", {})
    question_sets_used: list[str] = data["metadata"]["question_sets_used"]

    all_qs, resolved = load_data()
    used_qs = [qs for qs in all_qs if qs.forecast_due_date in question_sets_used]
    resolutions = {
        q.id: Resolution(id=q.id, outcome=q.outcome, resolution_date=q.resolution_date)
        for q in resolved
    }
    iteration_resolved = join_resolved_questions(used_qs, resolutions)

    source_map: dict[str, str] = {}
    for rq in iteration_resolved:
        source_map[rq.id] = rq.source

    composite_re = re.compile(r"^(.+)_(\d{4}-\d{2}-\d{2})$")

    by_source: dict[str, list[tuple[float, float]]] = {}
    for qid, forecast in forecasts.items():
        if qid not in outcomes:
            continue
        outcome = outcomes[qid]
        if outcome not in (0, 1):
            continue

        source = source_map.get(qid)
        if source is None:
            m = composite_re.match(qid)
            if m:
                source = source_map.get(m.group(1))
        if source is None:
            continue

        by_source.setdefault(source, []).append((forecast, float(outcome)))

    CALIBRATION_DIR.mkdir(parents=True, exist_ok=True)

    for source, pairs in sorted(by_source.items()):
        if len(pairs) < MIN_DATA_POINTS:
            logger.info("calibration_skip_source", source=source, n_points=len(pairs),
                        reason=f"need_at_least_{MIN_DATA_POINTS}")
            continue

        preds = [p for p, _ in pairs]
        outs = [o for _, o in pairs]
        breakpoints = isotonic_regression(preds, outs)

        calibrated_vals = [_interpolate(p, breakpoints) for p in preds]
        deltas = [c - p for c, p in zip(calibrated_vals, preds)]
        mean_shift = sum(deltas) / len(deltas) if deltas else 0.0
        max_shift = max(abs(d) for d in deltas) if deltas else 0.0

        model_data = {
            "breakpoints": [[x, y] for x, y in breakpoints],
            "n_points": len(pairs),
            "mean_shift": mean_shift,
        }

        path = CALIBRATION_DIR / f"{source}.json"
        path.write_text(json.dumps(model_data, indent=2))
        logger.info("calibration_learned", source=source, n_points=len(pairs),
                     n_breakpoints=len(breakpoints), mean_shift=round(mean_shift, 4),
                     max_shift=round(max_shift, 4))


def show() -> None:
    """Display current calibration models."""
    models = load_calibration_models()
    if not models:
        print("No calibration models found.")
        return

    print(f"\nCalibration models ({len(models)} sources):")
    for source, breakpoints in sorted(models.items()):
        path = CALIBRATION_DIR / f"{source}.json"
        data = json.loads(path.read_text())
        n_points = data.get("n_points", "?")
        mean_shift = data.get("mean_shift", 0.0)
        print(f"\n  {source} ({n_points} training points, mean shift: {mean_shift:+.4f}):")
        print(f"    {'Input':>8s} -> {'Output':>8s}")
        for x, y in breakpoints:
            print(f"    {x:>8.4f} -> {y:>8.4f}")


def _load_source_model(source: str) -> list[tuple[float, float]] | None:
    path = CALIBRATION_DIR / f"{source}.json"
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text())
        return [(float(x), float(y)) for x, y in data["breakpoints"]]
    except (json.JSONDecodeError, KeyError, ValueError):
        return None


def _interpolate(x: float, breakpoints: list[tuple[float, float]]) -> float:
    """Piecewise-linear interpolation between breakpoints."""
    if not breakpoints:
        return x
    if len(breakpoints) == 1:
        return breakpoints[0][1]

    if x <= breakpoints[0][0]:
        return breakpoints[0][1]
    if x >= breakpoints[-1][0]:
        return breakpoints[-1][1]

    for i in range(len(breakpoints) - 1):
        x0, y0 = breakpoints[i]
        x1, y1 = breakpoints[i + 1]
        if x0 <= x <= x1:
            if x1 == x0:
                return y0
            t = (x - x0) / (x1 - x0)
            return y0 + t * (y1 - y0)

    return breakpoints[-1][1]


# ---------------------------------------------------------------------------
# Platt scaling — used by eval.py --fit-calibration / --calibrate flags
# ---------------------------------------------------------------------------

def _logit(p: float) -> float:
    EPS = 1e-6
    p = max(EPS, min(1 - EPS, p))
    return math.log(p / (1 - p))


def _logistic(x: float) -> float:
    if x > 500:
        return 1.0
    if x < -500:
        return 0.0
    return 1.0 / (1.0 + math.exp(-x))


def platt_calibrate(probability: float, a: float = 1.0, b: float = 0.0) -> float:
    """Apply Platt scaling: calibrated_p = logistic(a * logit(p) + b)."""
    return _logistic(a * _logit(probability) + b)


def _negative_log_likelihood(
    ab: Any,
    logits: Any,
    outcomes: Any,
) -> float:
    import numpy as np
    a, b = float(ab[0]), float(ab[1])
    z = a * logits + b
    z = np.clip(z, -500, 500)
    p = 1.0 / (1.0 + np.exp(-z))
    p = np.clip(p, 1e-15, 1 - 1e-15)
    nll = -np.sum(outcomes * np.log(p) + (1 - outcomes) * np.log(1 - p))
    return float(nll)


def fit_platt(
    forecasts: list[float],
    outcomes: list[int],
    prior_a: float = 1.0,
    prior_b: float = 0.0,
    regularization: float = 0.0,
) -> tuple[float, float]:
    """Fit Platt scaling parameters via maximum likelihood."""
    import numpy as np
    from scipy.optimize import minimize as sp_minimize

    logits = np.array([_logit(p) for p in forecasts])
    outs = np.array(outcomes)

    def objective(ab: Any) -> float:
        nll = _negative_log_likelihood(ab, logits, outs)
        if regularization > 0:
            reg = regularization * ((ab[0] - prior_a) ** 2 + (ab[1] - prior_b) ** 2)
            return float(nll + reg)
        return nll

    result = sp_minimize(
        objective,
        x0=np.array([prior_a, prior_b]),
        method="Nelder-Mead",
        options={"maxiter": 1000, "xatol": 1e-8, "fatol": 1e-8},
    )
    return float(result.x[0]), float(result.x[1])


def fit_calibration(
    forecasts: dict[str, float],
    outcomes: dict[str, int],
    sources: dict[str, str],
    min_samples: int = MIN_SAMPLES_FOR_SOURCE,
    horizon_indices: dict[str, int] | None = None,
) -> dict[str, dict[str, float]]:
    """Fit per-source Platt scaling with hierarchical prior.

    When horizon_indices is provided, also fits per-(source, horizon) models
    with keys like 'fred_h1', 'fred_h2'. Falls back to source-only fit as
    prior when per-horizon data is insufficient.
    """
    by_source: dict[str, tuple[list[float], list[int]]] = defaultdict(lambda: ([], []))
    by_source_horizon: dict[str, tuple[list[float], list[int]]] = defaultdict(lambda: ([], []))
    all_f: list[float] = []
    all_o: list[int] = []
    for qid, prob in forecasts.items():
        if qid not in outcomes:
            continue
        source = sources.get(qid, "unknown")
        by_source[source][0].append(prob)
        by_source[source][1].append(outcomes[qid])
        all_f.append(prob)
        all_o.append(outcomes[qid])

        if horizon_indices is not None:
            horizon = horizon_indices.get(qid, 0)
            if horizon > 0:
                key = f"{source}_h{horizon}"
                by_source_horizon[key][0].append(prob)
                by_source_horizon[key][1].append(outcomes[qid])

    if not all_f:
        return {}

    global_a, global_b = fit_platt(all_f, all_o) if len(all_f) >= min_samples else (1.0, 0.0)

    params: dict[str, dict[str, float]] = {"_global": {"a": global_a, "b": global_b}}
    for source, (fs, os_) in by_source.items():
        if len(fs) >= min_samples:
            a, b = fit_platt(fs, os_)
            params[source] = {"a": a, "b": b}
        else:
            a, b = fit_platt(
                fs, os_,
                prior_a=global_a, prior_b=global_b,
                regularization=1.0,
            )
            params[source] = {"a": a, "b": b}

    if horizon_indices is not None:
        for sh_key, (fs, os_) in by_source_horizon.items():
            source = sh_key.rsplit("_h", 1)[0]
            source_params = params.get(source, params["_global"])
            if len(fs) >= min_samples:
                a, b = fit_platt(fs, os_)
                params[sh_key] = {"a": a, "b": b}
            else:
                a, b = fit_platt(
                    fs, os_,
                    prior_a=source_params["a"], prior_b=source_params["b"],
                    regularization=1.0,
                )
                params[sh_key] = {"a": a, "b": b}

    return params


def save_calibration(params: dict[str, dict[str, float]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(params, indent=2))


def load_calibration(path: Path) -> dict[str, dict[str, float]]:
    data = json.loads(path.read_text())
    return {k: {"a": float(v["a"]), "b": float(v["b"])} for k, v in data.items()}


def calibration_path(model_slug: str) -> Path:
    return CALIBRATION_DIR / f"{model_slug}.json"


def calibrate_forecasts(
    forecasts: dict[str, float],
    sources: dict[str, str],
    params: dict[str, dict[str, float]],
    horizon_indices: dict[str, int] | None = None,
) -> dict[str, float]:
    """Apply per-source Platt calibration to a dict of forecasts.

    When horizon_indices is provided, tries per-horizon key (e.g. 'fred_h1')
    first, falling back to source-only, then _global.
    """
    result: dict[str, float] = {}
    for qid, prob in forecasts.items():
        src = sources.get(qid, "_global")
        horizon = horizon_indices.get(qid, 0) if horizon_indices is not None else 0
        if horizon > 0:
            sh_key = f"{src}_h{horizon}"
            p = params.get(sh_key, params.get(src, params.get("_global", {"a": 1.0, "b": 0.0})))
        else:
            p = params.get(src, params.get("_global", {"a": 1.0, "b": 0.0}))
        result[qid] = platt_calibrate(prob, p["a"], p["b"])
    return result


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Calibration tools for ForecastBench")
    subparsers = parser.add_subparsers(dest="command")

    learn_parser = subparsers.add_parser("learn", help="Learn calibration from a result file")
    learn_parser.add_argument("--result", required=True, help="Path to result JSON file")

    subparsers.add_parser("show", help="Display current calibration models")

    args = parser.parse_args()

    if args.command == "learn":
        learn(args.result)
    elif args.command == "show":
        show()
    else:
        parser.print_help()
