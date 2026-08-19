"""Build and maintain the pinned question subsample used by the Brier gate.

Two jobs:

    --rounds R1 R2 --market 100 --dataset 100
        Writes gate_manifest.json: a fixed, stratified, deterministic set of
        question IDs. Run once. Commit the result. Changing it invalidates
        every comparison you have made.

    --set-baseline
        Runs the current forecaster over the manifest and writes
        gate_baseline.json. The gate ladder is centered on this value, so the
        score sits mid-ladder and the search has room in both directions.
        Re-run deliberately, as a commit, after accepting a real gain.

Kept outside tests/ so `uv run pytest` (testpaths = ["tests"]) never touches
it, and outside factory.md's Mutable list so the factory cannot edit the
thing that scores it.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import random
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
MANIFEST = REPO / "gate_manifest.json"
BASELINE = REPO / "gate_baseline.json"

MARKET_SOURCES = {"metaculus", "polymarket", "manifold", "infer"}

# Fixed. Do not change without re-baselining, or comparisons across runs
# stop meaning anything.
SEED = 20260818


def build_manifest(rounds: list[str], n_market: int, n_dataset: int) -> dict:
    from fetch_data import fetch_question_set

    by_source: dict[str, list[str]] = defaultdict(list)
    for round_name in rounds:
        filename = round_name if round_name.endswith(".json") else f"{round_name}-llm.json"
        qs = fetch_question_set(filename)
        for q in qs.questions:
            by_source[q.source].append(q.id)

    rng = random.Random(SEED)
    market_pool = sorted({s for s in by_source if s in MARKET_SOURCES})
    dataset_pool = sorted({s for s in by_source if s not in MARKET_SOURCES})

    def stratify(pool: list[str], total: int) -> list[str]:
        """Proportional allocation across sources, deterministic."""
        if not pool:
            return []
        sizes = {s: len(by_source[s]) for s in pool}
        grand = sum(sizes.values())
        picked: list[str] = []
        for source in pool:
            share = round(total * sizes[source] / grand)
            ids = sorted(set(by_source[source]))
            rng.shuffle(ids)
            picked.extend(ids[:share])
        return sorted(set(picked))

    market_ids = stratify(market_pool, n_market)
    dataset_ids = stratify(dataset_pool, n_dataset)

    return {
        "seed": SEED,
        "rounds": sorted(rounds),
        "market_ids": market_ids,
        "dataset_ids": dataset_ids,
        "n_market": len(market_ids),
        "n_dataset": len(dataset_ids),
        "n_total": len(market_ids) + len(dataset_ids),
    }


def load_manifest() -> dict:
    if not MANIFEST.exists():
        raise SystemExit(
            f"{MANIFEST} not found. Run: python gate/make_manifest.py "
            "--rounds 2026-06-21 2026-07-05 --market 100 --dataset 100"
        )
    return json.loads(MANIFEST.read_text())


def score_manifest() -> float:
    """Run the lab forecaster over the pinned subsample. Returns Brier Index."""
    import eval as ev
    import lab_forecaster

    manifest = load_manifest()
    ids = set(manifest["market_ids"]) | set(manifest["dataset_ids"])

    result = asyncio.run(
        ev.run_eval(
            forecaster=lab_forecaster.aforecast,
            multi_forecaster=lab_forecaster.aforecast_multi_horizon,
            agent_name="lab",
            raw=True,                       # difficulty adjustment off, always
            n_held_out=0,                   # the manifest IS the split
            n_rounds=len(manifest["rounds"]),
            question_filter=ids,            # requires the eval.py patch
        )
    )
    return float(result.scoring.overall_index)


def main() -> None:
    p = argparse.ArgumentParser(description="Brier gate manifest tooling")
    p.add_argument("--rounds", nargs="+", help="Round names, e.g. 2026-06-21 2026-07-05")
    p.add_argument("--market", type=int, default=100, help="Market questions to pin")
    p.add_argument("--dataset", type=int, default=100, help="Dataset questions to pin")
    p.add_argument("--set-baseline", action="store_true", help="Score manifest, write baseline")
    args = p.parse_args()

    if args.rounds:
        manifest = build_manifest(args.rounds, args.market, args.dataset)
        MANIFEST.write_text(json.dumps(manifest, indent=2))
        print(
            f"wrote {MANIFEST.name}: {manifest['n_total']} questions "
            f"({manifest['n_market']} market, {manifest['n_dataset']} dataset) "
            f"from {len(manifest['rounds'])} rounds"
        )

    if args.set_baseline:
        index = score_manifest()
        BASELINE.write_text(json.dumps({"brier_index": round(index, 3)}, indent=2))
        print(f"wrote {BASELINE.name}: brier_index = {index:.3f}")
        print("ladder will span", f"{index - 4:.1f}", "to", f"{index + 5:.1f}")

    if not args.rounds and not args.set_baseline:
        p.error("nothing to do: pass --rounds and/or --set-baseline")


if __name__ == "__main__":
    main()
