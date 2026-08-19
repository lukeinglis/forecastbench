"""Graded Brier Index gate. This is the outer-loop fitness function.

Scores FULL rounds, one at a time, and averages the per-round Brier Index.
See gate/make_manifest.py::score_rounds for why per-round rather than pooled.

remote-factory scores a candidate by piping this command's stdout through
parse_pytest_stdout, which regexes "(\d+) passed" / "(\d+) failed" /
"(\d+) error", then computes `passed / total - 0.01 * node_count`. A binary
pass/fail gives that search no gradient, and the contrastive reflector returns
an empty report when top-K and bottom-K don't differ. Hence a 10-rung ladder
centered on a committed baseline: at baseline the score is 0.5, with room to
move both ways.

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


def _load_baseline() -> float | None:
    if not BASELINE.exists():
        return None
    try:
        return float(json.loads(BASELINE.read_text())["brier_index"])
    except (json.JSONDecodeError, KeyError, ValueError, TypeError):
        return None


@pytest.fixture(scope="session")
def baseline() -> float:
    """Brier Index the ladder is centered on."""
    base = _load_baseline()
    if base is None:
        pytest.skip("gate_baseline.json missing; run gate/make_manifest.py --set-baseline")
    return base


@pytest.fixture(scope="session")
def brier_index(baseline: float) -> float:
    """Score every pinned round once per session, not once per rung.

    Depends on `baseline` so a missing baseline skips before any API calls.
    """
    if not ROUNDS.exists():
        pytest.skip("gate_rounds.json missing; run gate/make_manifest.py --rounds ...")

    from gate.make_manifest import score_rounds

    rounds = json.loads(ROUNDS.read_text())["rounds"]
    index, per_round = score_rounds(rounds)

    print()
    for name, res in per_round:
        print(
            f"gate: {name} index={res.overall_index:.3f} "
            f"rows={res.n_dataset + res.n_market} "
            f"(dataset={res.n_dataset} market={res.n_market} missing={res.n_missing})"
        )
    print(f"gate: mean index={index:.3f} across {len(per_round)} rounds")
    return index


@pytest.mark.parametrize("offset", _OFFSETS)
def test_brier_index_at_or_above(brier_index: float, baseline: float, offset: int) -> None:
    """One rung of the ladder. Pass count is the fitness signal."""
    threshold = baseline + offset
    assert brier_index >= threshold, (
        f"Brier Index {brier_index:.3f} below rung {threshold:.1f} "
        f"(baseline {baseline:.3f}{offset:+d})"
    )


def test_pinned_rounds_well_formed() -> None:
    """Guard against a shrunken evaluation.

    Missing forecasts default to 0.5 under ForecastBench rules, so a run that
    silently drops rounds produces a plausible number rather than an obvious
    failure. score_rounds() raises on zero rows; this checks the pinned set.
    """
    if not ROUNDS.exists():
        pytest.skip("gate_rounds.json missing")
    rounds = json.loads(ROUNDS.read_text())["rounds"]
    assert len(rounds) >= 1
    assert len(set(rounds)) == len(rounds), "duplicate rounds pinned"
