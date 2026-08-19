"""Pin the rounds the Brier gate scores, and set its baseline.

    --rounds 2026-03-01 2026-04-12 2026-05-10
        Writes gate_rounds.json. Full rounds, no sampling. Run once, commit
        the result. Changing it invalidates every comparison you have made.

    --set-baseline
        Scores the pinned rounds with the current lab forecaster and writes
        gate_baseline.json. The ladder in test_brier_gate.py is centered on
        this value, so the score sits mid-ladder with room to move both ways.
        Re-run deliberately, as a commit, after accepting a real gain.

    --dry-run
        Same scoring path with the dummy forecaster. No API calls. Use it to
        check row counts before spending anything.

Kept outside tests/ so `uv run pytest` (testpaths = ["tests"]) never triggers
it, and outside factory.md's Mutable list so the factory cannot edit the thing
that scores it.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any

REPO = Path(__file__).resolve().parent.parent
ROUNDS = REPO / "gate_rounds.json"
BASELINE = REPO / "gate_baseline.json"

# gate/ is not the repo root, so `python gate/make_manifest.py` puts gate/ on
# sys.path rather than the root and `import eval` fails.
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

if TYPE_CHECKING:
    from score import ScoringResult


def load_rounds() -> list[str]:
    if not ROUNDS.exists():
        raise SystemExit(
            f"{ROUNDS} not found. Run: python gate/make_manifest.py "
            "--rounds 2026-03-01 2026-04-12 2026-05-10"
        )
    rounds: list[str] = json.loads(ROUNDS.read_text())["rounds"]
    return rounds


def score_rounds(
    rounds: list[str],
    dummy: bool = False,
) -> tuple[float, list[tuple[str, ScoringResult]]]:
    """Score each round separately, return the mean index and the per-round results.

    One run_eval call per round with round_name set. Without round_name,
    run_eval loads every published round at once; question IDs recur across
    rounds and are forecast only once, keyed on the first occurrence's
    resolution dates, so rows from other rounds match no forecast and fall
    through to the 0.5 default. Per-round calls keep identity unambiguous.

    Averaging round indices rather than pooling rows is also the honest
    statistic: questions within a round are correlated, so the effective
    sample size for generalization tracks the round count.
    """
    import eval as ev  # noqa: PLC0415

    if dummy:
        import dummy_forecaster  # noqa: PLC0415

        forecaster: Any = dummy_forecaster.forecast
        multi: Any = None
        agent = "dummy"
    else:
        import lab_forecaster  # noqa: PLC0415

        forecaster = lab_forecaster.aforecast
        multi = lab_forecaster.aforecast_multi_horizon
        agent = "lab"

    per_round: list[tuple[str, ScoringResult]] = []
    for name in rounds:
        result = asyncio.run(
            ev.run_eval(
                forecaster=forecaster,
                multi_forecaster=multi,
                agent_name=agent,
                raw=True,                    # difficulty adjustment off, always
                round_name=f"{name}-llm",    # loads exactly ONE question set
                n_held_out=0,                # the pinned rounds are the split
            )
        )
        scoring = result.scoring
        rows = scoring.n_dataset + scoring.n_market
        if rows == 0:
            raise SystemExit(
                f"Round {name} scored zero rows. Check that it has a published "
                "resolution set."
            )
        per_round.append((name, scoring))

    if not per_round:
        raise SystemExit("No rounds scored.")

    mean_index = sum(r.overall_index for _, r in per_round) / len(per_round)
    return mean_index, per_round


def _report(mean_index: float, per_round: list[tuple[str, ScoringResult]]) -> None:
    for name, r in per_round:
        print(
            f"{name}  index={r.overall_index:7.3f}  "
            f"rows={r.n_dataset + r.n_market:5d}  "
            f"dataset={r.n_dataset:5d} ({r.dataset_index:6.2f})  "
            f"market={r.n_market:4d} ({r.market_index:6.2f})  "
            f"missing={r.n_missing}"
        )
    print(f"mean index across {len(per_round)} rounds: {mean_index:.3f}")


def main() -> None:
    p = argparse.ArgumentParser(description="Brier gate round pinning")
    p.add_argument("--rounds", nargs="+", help="Round names, e.g. 2026-03-01 2026-04-12")
    p.add_argument("--set-baseline", action="store_true", help="Score rounds, write baseline")
    p.add_argument("--dry-run", action="store_true", help="Score with dummy forecaster, no API calls")
    args = p.parse_args()

    if args.rounds:
        if len(set(args.rounds)) != len(args.rounds):
            p.error("duplicate rounds")
        ROUNDS.write_text(json.dumps({"rounds": sorted(args.rounds)}, indent=2))
        print(f"wrote {ROUNDS.name}: {len(args.rounds)} rounds {sorted(args.rounds)}")

    if args.dry_run:
        mean_index, per_round = score_rounds(load_rounds(), dummy=True)
        print("\n--- dry run (dummy forecaster, no API calls) ---")
        _report(mean_index, per_round)

    if args.set_baseline:
        mean_index, per_round = score_rounds(load_rounds())
        _report(mean_index, per_round)
        BASELINE.write_text(json.dumps({"brier_index": round(mean_index, 3)}, indent=2))
        print(f"\nwrote {BASELINE.name}: brier_index = {mean_index:.3f}")
        print(f"ladder spans {mean_index - 4:.1f} to {mean_index + 5:.1f}")

    if not (args.rounds or args.dry_run or args.set_baseline):
        p.error("nothing to do: pass --rounds, --dry-run, and/or --set-baseline")


if __name__ == "__main__":
    main()
