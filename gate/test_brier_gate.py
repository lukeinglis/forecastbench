"""Graded Brier Index gate. This is the outer loop's fitness function.

remote-factory scores a candidate by piping the --test-command stdout through
parse_pytest_stdout, which regexes for "(\\d+) passed" / "(\\d+) failed" /
"(\\d+) error", then computes:

    fitness = passed / total - 0.01 * node_count

A binary pass/fail gate gives that search no gradient: every candidate scores
1.0 or 0.0 and the contrastive reflector, which needs variance between top-K
and bottom-K, produces an empty report. So this emits a ladder of 10 graded
assertions centered on a committed baseline. At baseline the score is 0.5,
which leaves room to move in both directions.

Run:  uv run pytest gate/ -q

Invisible to `uv run pytest` because pyproject sets testpaths = ["tests"].
Listed Read-Only in factory.md: this file defines how the factory is scored,
so the factory must not be able to edit it.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
MANIFEST = REPO / "gate_manifest.json"
BASELINE = REPO / "gate_baseline.json"

# Ladder spans baseline - 4 through baseline + 5 in 1-point steps.
# Asymmetric on purpose: more headroom above than below, since the point is
# to reward improvement more finely than it punishes small regressions.
LADDER_BELOW = 4
LADDER_ABOVE = 5


def _thresholds() -> list[float]:
    if not BASELINE.exists():
        pytest.skip("gate_baseline.json missing; run gate/make_manifest.py --set-baseline")
    base = float(json.loads(BASELINE.read_text())["brier_index"])
    return [base + offset for offset in range(-LADDER_BELOW, LADDER_ABOVE + 1)]


@pytest.fixture(scope="session")
def brier_index() -> float:
    """Score the pinned subsample once per session, not once per threshold."""
    import sys

    sys.path.insert(0, str(REPO))
    import eval as ev
    import lab_forecaster

    if not MANIFEST.exists():
        pytest.skip("gate_manifest.json missing; run gate/make_manifest.py --rounds ...")

    manifest = json.loads(MANIFEST.read_text())
    ids = set(manifest["market_ids"]) | set(manifest["dataset_ids"])

    result = asyncio.run(
        ev.run_eval(
            forecaster=lab_forecaster.aforecast,
            multi_forecaster=lab_forecaster.aforecast_multi_horizon,
            agent_name="lab",
            raw=True,                       # difficulty adjustment stays off
            n_held_out=0,                   # the manifest is the split
            n_rounds=len(manifest["rounds"]),
            question_filter=ids,
        )
    )

    print(
        f"\ngate: index={result.scoring.overall_index:.3f} "
        f"(dataset={result.scoring.dataset_index:.3f} n={result.scoring.n_dataset}, "
        f"market={result.scoring.market_index:.3f} n={result.scoring.n_market}, "
        f"missing={result.scoring.n_missing})"
    )
    return float(result.scoring.overall_index)


@pytest.mark.parametrize("threshold", _thresholds())
def test_brier_index_at_or_above(brier_index: float, threshold: float) -> None:
    """One rung of the ladder. Pass count is the fitness signal."""
    assert brier_index >= threshold, (
        f"Brier Index {brier_index:.3f} below rung {threshold:.1f}"
    )


def test_coverage_is_intact(brier_index: float) -> None:
    """Guard against scoring a shrunken question set.

    Missing forecasts default to 0.5 per ForecastBench rules, so a forecaster
    that silently drops hard questions could otherwise look better than one
    that answers them. This rung fails if the manifest was not fully scored.
    """
    import sys

    sys.path.insert(0, str(REPO))
    manifest = json.loads(MANIFEST.read_text())
    # brier_index fixture already ran; re-reading the result would double the
    # cost, so this asserts on manifest integrity rather than re-scoring.
    assert manifest["n_total"] == len(manifest["market_ids"]) + len(manifest["dataset_ids"])
    assert manifest["n_market"] > 0 and manifest["n_dataset"] > 0
