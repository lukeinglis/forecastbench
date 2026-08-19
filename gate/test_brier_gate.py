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


# Offsets, not absolute thresholds. Collection must not depend on
# gate_baseline.json existing: parametrize arguments are evaluated at import
# time, and pytest.skip() at module level is a collection error, not a skip.
# The baseline is read inside the fixture instead.
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
    """Score the pinned subsample once per session, not once per rung.

    Depends on `baseline` so a missing baseline skips before any API calls.
    """
    import sys

    if not MANIFEST.exists():
        pytest.skip("gate_manifest.json missing; run gate/make_manifest.py --rounds ...")

    sys.path.insert(0, str(REPO))
    import eval as ev
    import lab_forecaster

    manifest = json.loads(MANIFEST.read_text())
    ids = set(manifest["market_ids"]) | set(manifest["dataset_ids"])

    result = asyncio.run(
        ev.run_eval(
            forecaster=lab_forecaster.aforecast,
            multi_forecaster=lab_forecaster.aforecast_multi_horizon,
            agent_name="lab",
            raw=True,                       # difficulty adjustment stays off
            n_held_out=0,                   # the manifest is the split
            question_filter=ids,
        )
    )

    scored = result.scoring.n_dataset + result.scoring.n_market
    assert scored > 0, (
        "gate scored zero rows: the manifest IDs did not match any resolved "
        "questions. Check that gate_manifest.json rounds have resolution sets."
    )

    print(
        f"\ngate: index={result.scoring.overall_index:.3f} "
        f"(dataset={result.scoring.dataset_index:.3f} n={result.scoring.n_dataset}, "
        f"market={result.scoring.market_index:.3f} n={result.scoring.n_market}, "
        f"missing={result.scoring.n_missing})"
    )
    return float(result.scoring.overall_index)


@pytest.mark.parametrize("offset", _OFFSETS)
def test_brier_index_at_or_above(brier_index: float, baseline: float, offset: int) -> None:
    """One rung of the ladder. Pass count is the fitness signal."""
    threshold = baseline + offset
    assert brier_index >= threshold, (
        f"Brier Index {brier_index:.3f} below rung {threshold:.1f} "
        f"(baseline {baseline:.3f}{offset:+d})"
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
