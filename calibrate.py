"""Hierarchical Platt scaling calibration for forecast probabilities."""

from __future__ import annotations

import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from scipy.optimize import minimize


CALIBRATION_DIR = Path(".cache/calibration")
MIN_SAMPLES_FOR_SOURCE = 10


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


def calibrate(probability: float, a: float = 1.0, b: float = 0.0) -> float:
    """Apply Platt scaling: calibrated_p = logistic(a * logit(p) + b)."""
    return _logistic(a * _logit(probability) + b)


def _negative_log_likelihood(
    ab: np.ndarray[Any, np.dtype[np.floating[Any]]],
    logits: np.ndarray[Any, np.dtype[np.floating[Any]]],
    outcomes: np.ndarray[Any, np.dtype[np.integer[Any]]],
) -> float:
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
    logits = np.array([_logit(p) for p in forecasts])
    outs = np.array(outcomes)

    def objective(ab: np.ndarray[Any, np.dtype[np.floating[Any]]]) -> float:
        nll = _negative_log_likelihood(ab, logits, outs)
        if regularization > 0:
            reg = regularization * ((ab[0] - prior_a) ** 2 + (ab[1] - prior_b) ** 2)
            return float(nll + reg)
        return nll

    result = minimize(
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
) -> dict[str, dict[str, float]]:
    """Fit per-source Platt scaling with hierarchical prior.

    Sources with >= min_samples pairs get independent fits.
    Sources with fewer samples use global params as a regularized prior.
    """
    by_source: dict[str, tuple[list[float], list[int]]] = defaultdict(lambda: ([], []))
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
) -> dict[str, float]:
    """Apply per-source calibration to a dict of forecasts."""
    result: dict[str, float] = {}
    for qid, prob in forecasts.items():
        src = sources.get(qid, "_global")
        src_params = params.get(src, params.get("_global", {"a": 1.0, "b": 0.0}))
        result[qid] = calibrate(prob, src_params["a"], src_params["b"])
    return result
