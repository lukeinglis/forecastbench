"""Graded Brier Index gate. This is the outer-loop fitness function.

Scores FULL rounds, one at a time, and averages per-round Brier Index.
See gate/make_manifest.py::score_rounds for why per-round rather than pooled.

Two independent ladders — dataset and market — so the outer loop gets a
gradient on each category separately. This matches the competition's
equal-weight scoring: overall = (dataset_brier + market_brier) / 2.
A candidate that improves one category while hurting the other will show
a split signal instead of a masked overall gain — the failure mode that
made gen-0 calibration look like an improvement when it was a regression.

remote-factory scores a candidate by piping this command's stdout through
parse_pytest_stdout, which regexes "(\\d+) passed" / "(\\d+) failed" /
"(\\d+) error", then computes `passed / total - 0.01 * node_count`. With
two 10-rung ladders the total is 20, giving finer resolution (0.05) than
the original single 10-rung ladder (0.10).

Run:  uv run pytest gate/ -q

Invisible to `uv run pytest` because pyproject sets testpaths = ["tests"].
Listed Read-Only in factory.md: this file defines how the factory is scored.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
ROUNDS = REPO / "gate_rounds.json"
BASELINE = REPO / "gate_baseline.json"

if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

# Offsets, not absolute thresholds. Parametrize arguments are evaluated at
# import time, and pytest.skip() at module level is a collection error rather
# than a skip. The baseline is read inside a fixture instead.
LADDER_BELOW = 4
LADDER_ABOVE = 5
_OFFSETS = list(range(-LADDER_BELOW, LADDER_ABOVE + 1))


def _load_baselines() -> dict[str, float] | None:
    """Load per-category baselines from gate_baseline.json.

    Returns None (→ skip) if the file is missing or uses the old
    single-value format. The caller should re-run --set-baseline.
    """
    if not BASELINE.exists():
        return None
    try:
        data = json.loads(BASELINE.read_text())
        if "dataset_index" not in data or "market_index" not in data:
            return None  # Old format — needs re-baselining
        return {
            "dataset": float(data["dataset_index"]),
            "market": float(data["market_index"]),
            "overall": float(data.get("overall_index", (data["dataset_index"] + data["market_index"]) / 2)),
        }
    except (json.JSONDecodeError, KeyError, ValueError, TypeError):
        return None


@pytest.fixture(scope="session")
def baselines() -> dict[str, float]:
    """Per-category Brier Index baselines the ladders are centered on."""
    bl = _load_baselines()
    if bl is None:
        pytest.skip(
            "gate_baseline.json missing per-category baselines; "
            "run: python gate/make_manifest.py --set-baseline"
        )
    return bl


@pytest.fixture(scope="session")
def scores(baselines: dict[str, float]) -> dict[str, float]:
    """Score every pinned round once per session, not once per rung.

    Depends on `baselines` so a missing baseline skips before any API calls.
    Returns mean per-category Brier Index across all pinned rounds.
    Respects GATE_ROUNDS_OVERRIDE env var for holdout evaluation.
    """
    from gate.make_manifest import load_rounds, score_rounds

    rounds = load_rounds()
    _, per_round = score_rounds(rounds)

    mean_dataset = sum(r.dataset_index for _, r in per_round) / len(per_round)
    mean_market = sum(r.market_index for _, r in per_round) / len(per_round)
    mean_overall = sum(r.overall_index for _, r in per_round) / len(per_round)

    print()
    for name, res in per_round:
        total = res.n_dataset + res.n_market
        print(
            f"gate: {name} overall={res.overall_index:.3f} "
            f"dataset={res.dataset_index:.3f} (n={res.n_dataset}) "
            f"market={res.market_index:.3f} (n={res.n_market}) "
            f"missing={res.n_missing} rows={total}"
        )
    print(
        f"gate: mean across {len(per_round)} rounds — "
        f"overall={mean_overall:.3f} dataset={mean_dataset:.3f} market={mean_market:.3f}"
    )
    return {"dataset": mean_dataset, "market": mean_market, "overall": mean_overall}


# ---------------------------------------------------------------------------
# Two independent ladders: dataset and market.
# Equal rung count per category = equal weight, matching competition scoring.
# Total rungs = 2 * len(_OFFSETS) = 20.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("offset", _OFFSETS)
def test_dataset_index_at_or_above(scores: dict[str, float], baselines: dict[str, float], offset: int) -> None:
    """Dataset Brier Index ladder. Pass count is half the fitness signal."""
    threshold = baselines["dataset"] + offset
    assert scores["dataset"] >= threshold, (
        f"Dataset Brier Index {scores['dataset']:.3f} below rung {threshold:.1f} "
        f"(baseline {baselines['dataset']:.3f}{offset:+d})"
    )


@pytest.mark.parametrize("offset", _OFFSETS)
def test_market_index_at_or_above(scores: dict[str, float], baselines: dict[str, float], offset: int) -> None:
    """Market Brier Index ladder. Pass count is half the fitness signal."""
    threshold = baselines["market"] + offset
    assert scores["market"] >= threshold, (
        f"Market Brier Index {scores['market']:.3f} below rung {threshold:.1f} "
        f"(baseline {baselines['market']:.3f}{offset:+d})"
    )


def test_pinned_rounds_well_formed() -> None:
    """Guard against a shrunken evaluation.

    Missing forecasts default to 0.5 under ForecastBench rules, so a run that
    silently drops rounds produces a plausible number rather than an obvious
    failure. score_rounds() raises on zero rows; this checks the pinned set.
    """
    from gate.make_manifest import load_rounds

    rounds = load_rounds()
    assert len(rounds) >= 1
    assert len(set(rounds)) == len(rounds), "duplicate rounds pinned"
