"""Source-specific calibration post-processing via isotonic regression.

Learns calibration curves from prior results and applies them as a
post-processing step before scoring. Uses Pool Adjacent Violators
Algorithm (PAVA) — pure Python, no scikit-learn dependency.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from logging_config import get_logger

logger = get_logger("calibrate")

CALIBRATION_DIR = Path(".cache/calibration")
MIN_DATA_POINTS = 20


def isotonic_regression(
    predictions: list[float], outcomes: list[float],
) -> list[tuple[float, float]]:
    """Pool Adjacent Violators Algorithm (PAVA) for isotonic regression.

    Input: parallel lists of (prediction, binary_outcome), need not be sorted.
    Output: monotone non-decreasing breakpoints as (x, y) tuples.
    """
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
    """Apply calibration to a single forecast probability.

    Loads model from disk if not provided. Returns input unchanged
    if no model exists for the source (identity fallback).
    """
    if models is not None:
        breakpoints = models.get(source)
    else:
        breakpoints = _load_source_model(source)

    if not breakpoints:
        return probability

    calibrated = _interpolate(probability, breakpoints)
    return max(0.001, min(0.999, calibrated))


def load_calibration_models() -> dict[str, list[tuple[float, float]]]:
    """Load all calibration models from disk."""
    models: dict[str, list[tuple[float, float]]] = {}
    if not CALIBRATION_DIR.exists():
        return models

    for path in CALIBRATION_DIR.glob("*.json"):
        source = path.stem
        try:
            data = json.loads(path.read_text())
            breakpoints = [(float(x), float(y)) for x, y in data["breakpoints"]]
            models[source] = breakpoints
        except (json.JSONDecodeError, KeyError, ValueError):
            logger.warning("calibration_load_failed", source=source)
    return models


def learn(result_path: str) -> None:
    """Learn calibration models from a result file.

    Re-joins forecasts with questions to recover per-question source,
    groups by source, fits isotonic regression per source.
    """
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

        calibrated = [_interpolate(p, breakpoints) for p in preds]
        deltas = [c - p for c, p in zip(calibrated, preds)]
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
