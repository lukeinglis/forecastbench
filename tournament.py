"""Tournament analysis for multi-model ForecastBench comparison.

Provides pairwise bootstrap comparisons, model×source Brier score matrices,
cost-accuracy summaries, and formatted tournament reports.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from logging_config import get_logger
from score import brier_score, brier_index

logger = get_logger("tournament")

SMALL_N_THRESHOLD = 100


@dataclass
class ModelResult:
    model_slug: str
    forecasts: dict[str, float]
    outcomes: dict[str, int]
    sources: dict[str, str]
    costs: dict[str, float]
    scoring_result: dict[str, Any]
    metadata: dict[str, Any]
    timestamp: str = ""


def load_tournament_results(results_dir: str | Path = "results") -> list[ModelResult]:
    logger.info("load_tournament_results", results_dir=str(results_dir))
    p = Path(results_dir)
    if not p.exists():
        logger.warning("load_tournament_results_no_dir", path=str(p))
        return []
    results: list[ModelResult] = []
    for f in sorted(p.glob("*.json")):
        try:
            data = json.loads(f.read_text())
            if "forecasts" not in data or "scoring_result" not in data:
                continue
            results.append(ModelResult(
                model_slug=data.get("model_slug", f.stem),
                forecasts=data["forecasts"],
                outcomes=data.get("outcomes", {}),
                sources=data.get("sources", {}),
                costs=data.get("costs", {}),
                scoring_result=data["scoring_result"],
                metadata=data.get("metadata", {}),
                timestamp=data.get("timestamp", ""),
            ))
        except (json.JSONDecodeError, KeyError):
            logger.warning("load_tournament_result_error", path=str(f))
            continue
    logger.info("load_tournament_results_complete", n_loaded=len(results))
    return results


def _source_pairs(
    result: ModelResult,
) -> dict[str, list[tuple[float, int]]]:
    logger.debug("source_pairs", model_slug=result.model_slug, n_outcomes=len(result.outcomes))
    by_source: dict[str, list[tuple[float, int]]] = {}
    for qid, outcome in result.outcomes.items():
        forecast = result.forecasts.get(qid, 0.5)
        source = result.sources.get(qid, "unknown")
        by_source.setdefault(source, []).append((forecast, outcome))
    return by_source


@dataclass
class CellStats:
    brier: float
    brier_index: float
    ci_low: float
    ci_high: float
    count: int
    small_n: bool


def model_source_matrix(
    results: list[ModelResult],
    n_bootstrap: int = 1000,
    seed: int = 42,
) -> dict[str, dict[str, CellStats]]:
    logger.info("model_source_matrix_start", n_models=len(results), n_bootstrap=n_bootstrap)
    matrix: dict[str, dict[str, CellStats]] = {}
    for result in results:
        by_source = _source_pairs(result)
        row: dict[str, CellStats] = {}
        for source, pairs in sorted(by_source.items()):
            if not pairs:
                continue
            mean_bs = sum(brier_score(f, o) for f, o in pairs) / len(pairs)
            bi = brier_index(mean_bs)
            ci_lo, ci_hi = _bootstrap_ci_brier(pairs, n_bootstrap=n_bootstrap, seed=seed)
            row[source] = CellStats(
                brier=mean_bs,
                brier_index=bi,
                ci_low=brier_index(ci_hi),
                ci_high=brier_index(ci_lo),
                count=len(pairs),
                small_n=len(pairs) < SMALL_N_THRESHOLD,
            )
        all_pairs = [(result.forecasts.get(qid, 0.5), o) for qid, o in result.outcomes.items()]
        if all_pairs:
            overall_bs = sum(brier_score(f, o) for f, o in all_pairs) / len(all_pairs)
            overall_bi = brier_index(overall_bs)
            ci_lo, ci_hi = _bootstrap_ci_brier(all_pairs, n_bootstrap=n_bootstrap, seed=seed)
            row["overall"] = CellStats(
                brier=overall_bs,
                brier_index=overall_bi,
                ci_low=brier_index(ci_hi),
                ci_high=brier_index(ci_lo),
                count=len(all_pairs),
                small_n=False,
            )
        matrix[result.model_slug] = row
    return matrix


def _bootstrap_ci_brier(
    pairs: list[tuple[float, int]],
    n_bootstrap: int = 1000,
    ci: float = 0.95,
    seed: int = 42,
) -> tuple[float, float]:
    rng = random.Random(seed)
    n = len(pairs)
    if n == 0:
        return (0.0, 0.0)
    means: list[float] = []
    for _ in range(n_bootstrap):
        sample = rng.choices(pairs, k=n)
        mean_bs = sum((f - o) ** 2 for f, o in sample) / n
        means.append(mean_bs)
    means.sort()
    alpha = (1 - ci) / 2
    lo = means[int(alpha * n_bootstrap)]
    hi = means[min(int((1 - alpha) * n_bootstrap), len(means) - 1)]
    return (lo, hi)


@dataclass
class BootstrapResult:
    mean_diff: float
    ci_low: float
    ci_high: float
    p_value: float
    n_questions: int


def paired_bootstrap_test(
    forecasts_a: dict[str, float],
    forecasts_b: dict[str, float],
    outcomes: dict[str, int],
    n_bootstrap: int = 10000,
    seed: int = 42,
) -> BootstrapResult:
    shared_ids = sorted(set(forecasts_a) & set(forecasts_b) & set(outcomes))
    n = len(shared_ids)
    logger.debug("paired_bootstrap_test", n_shared=n, n_bootstrap=n_bootstrap)
    if n == 0:
        return BootstrapResult(0.0, 0.0, 0.0, 1.0, 0)

    diffs = [
        brier_score(forecasts_a[qid], outcomes[qid]) - brier_score(forecasts_b[qid], outcomes[qid])
        for qid in shared_ids
    ]
    observed_diff = sum(diffs) / n

    rng = random.Random(seed)
    boot_diffs: list[float] = []
    count_extreme = 0
    for _ in range(n_bootstrap):
        sample = rng.choices(diffs, k=n)
        boot_mean = sum(sample) / n
        boot_diffs.append(boot_mean)
        if abs(boot_mean - observed_diff) >= abs(observed_diff):
            count_extreme += 1

    boot_diffs.sort()
    alpha = 0.025
    ci_low = boot_diffs[int(alpha * n_bootstrap)]
    ci_high = boot_diffs[min(int((1 - alpha) * n_bootstrap), len(boot_diffs) - 1)]
    p_value = count_extreme / n_bootstrap

    return BootstrapResult(
        mean_diff=observed_diff,
        ci_low=ci_low,
        ci_high=ci_high,
        p_value=p_value,
        n_questions=n,
    )


@dataclass
class PairwiseEntry:
    model_a: str
    model_b: str
    bootstrap: BootstrapResult
    a_wins: int
    b_wins: int
    significant: bool


def pairwise_comparison_table(
    results: list[ModelResult],
    n_bootstrap: int = 10000,
    seed: int = 42,
) -> list[PairwiseEntry]:
    logger.info("pairwise_comparison_start", n_models=len(results))
    entries: list[PairwiseEntry] = []
    for i, ra in enumerate(results):
        for rb in results[i + 1:]:
            shared_outcomes = {
                qid: o for qid, o in ra.outcomes.items()
                if qid in rb.outcomes
            }
            boot = paired_bootstrap_test(
                ra.forecasts, rb.forecasts, shared_outcomes,
                n_bootstrap=n_bootstrap, seed=seed,
            )
            shared_ids = set(ra.forecasts) & set(rb.forecasts) & set(shared_outcomes)
            a_wins = 0
            b_wins = 0
            for qid in shared_ids:
                bs_a = brier_score(ra.forecasts[qid], shared_outcomes[qid])
                bs_b = brier_score(rb.forecasts[qid], shared_outcomes[qid])
                if bs_a < bs_b:
                    a_wins += 1
                elif bs_b < bs_a:
                    b_wins += 1
            entries.append(PairwiseEntry(
                model_a=ra.model_slug,
                model_b=rb.model_slug,
                bootstrap=boot,
                a_wins=a_wins,
                b_wins=b_wins,
                significant=boot.p_value < 0.05,
            ))
    return entries


@dataclass
class CostEntry:
    model_slug: str
    total_cost: float
    mean_cost: float
    n_costed: int
    by_source: dict[str, dict[str, float]]


def cost_accuracy_summary(results: list[ModelResult]) -> list[CostEntry]:
    logger.info("cost_accuracy_summary_start", n_models=len(results))
    entries: list[CostEntry] = []
    for r in results:
        if not r.costs:
            continue
        source_costs: dict[str, list[float]] = {}
        for qid, cost in r.costs.items():
            src = r.sources.get(qid, "unknown")
            source_costs.setdefault(src, []).append(cost)
        by_source = {
            src: {"total": sum(vals), "mean": sum(vals) / len(vals), "count": len(vals)}
            for src, vals in sorted(source_costs.items())
        }
        entries.append(CostEntry(
            model_slug=r.model_slug,
            total_cost=sum(r.costs.values()),
            mean_cost=sum(r.costs.values()) / len(r.costs),
            n_costed=len(r.costs),
            by_source=by_source,
        ))
    return entries


def tournament_report(results: list[ModelResult]) -> str:
    logger.info("tournament_report_start", n_models=len(results))
    if not results:
        return "No results to report."

    lines: list[str] = []
    lines.append("=" * 80)
    lines.append("TOURNAMENT REPORT")
    lines.append("=" * 80)

    matrix = model_source_matrix(results)
    all_sources = sorted({s for row in matrix.values() for s in row if s != "overall"})

    lines.append("")
    lines.append("OVERALL RANKINGS (Brier Index)")
    lines.append("-" * 60)
    rankings = []
    for model, row in matrix.items():
        overall = row.get("overall")
        if overall:
            rankings.append((model, overall))
    rankings.sort(key=lambda x: x[1].brier_index, reverse=True)

    lines.append(f"  {'Rank':<5s} {'Model':<40s} {'Index':>7s} {'95% CI':>15s} {'N':>6s}")
    for rank, (model, stats) in enumerate(rankings, 1):
        ci_str = f"[{stats.ci_low:.1f}, {stats.ci_high:.1f}]"
        lines.append(f"  {rank:<5d} {model:<40s} {stats.brier_index:>6.1f}% {ci_str:>15s} {stats.count:>6d}")

    lines.append("")
    lines.append("MODEL x SOURCE MATRIX (Brier Index)")
    lines.append("-" * 80)

    src_header = "".join(f"{s:>12s}" for s in all_sources)
    lines.append(f"  {'Model':<30s}{src_header}")
    for model, stats in rankings:
        row = matrix[model]
        cells: list[str] = []
        for src in all_sources:
            cell = row.get(src)
            if cell:
                flag = "*" if cell.small_n else " "
                cells.append(f"{cell.brier_index:>10.1f}%{flag}")
            else:
                cells.append(f"{'—':>12s}")
        lines.append(f"  {model:<30s}{''.join(cells)}")
    lines.append("  * = small N (fewer than 100 questions), wider CIs expected")

    lines.append("")
    lines.append("PAIRWISE COMPARISONS (Brier Score Difference, A - B)")
    lines.append("-" * 80)
    pairwise = pairwise_comparison_table(results, n_bootstrap=1000, seed=42)
    if pairwise:
        lines.append(f"  {'Model A':<25s} {'Model B':<25s} {'Diff':>8s} {'p-val':>7s} {'Sig':>5s} {'A wins':>7s} {'B wins':>7s}")
        for entry in sorted(pairwise, key=lambda e: e.bootstrap.mean_diff):
            sig = "***" if entry.bootstrap.p_value < 0.001 else (
                "**" if entry.bootstrap.p_value < 0.01 else (
                    "*" if entry.significant else ""
                )
            )
            lines.append(
                f"  {entry.model_a:<25s} {entry.model_b:<25s} "
                f"{entry.bootstrap.mean_diff:>+7.4f} {entry.bootstrap.p_value:>7.3f} {sig:>5s} "
                f"{entry.a_wins:>7d} {entry.b_wins:>7d}"
            )
    else:
        lines.append("  (Need 2+ models for pairwise comparison)")

    cost_entries = cost_accuracy_summary(results)
    if cost_entries:
        lines.append("")
        lines.append("COST SUMMARY")
        lines.append("-" * 60)
        lines.append(f"  {'Model':<40s} {'Total $':>10s} {'Mean $/Q':>10s} {'N':>6s}")
        for ce in sorted(cost_entries, key=lambda x: x.mean_cost):
            lines.append(
                f"  {ce.model_slug:<40s} ${ce.total_cost:>9.2f} ${ce.mean_cost:>9.4f} {ce.n_costed:>6d}"
            )

    lines.append("")
    lines.append("=" * 80)
    return "\n".join(lines)
