"""Error analysis module for ForecastBench backtest results."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from fetch_data import ResolvedQuestion
from score import brier_score


@dataclass
class AnalysisResult:
    per_source_scores: dict[str, float] = field(default_factory=dict)
    worst_questions: list[dict[str, Any]] = field(default_factory=list)
    calibration_buckets: list[dict[str, Any]] = field(default_factory=list)


def run_analysis(
    forecasts: dict[str, float],
    resolved: list[ResolvedQuestion],
) -> AnalysisResult:
    """Compute per-source Brier scores, worst questions, and calibration buckets."""
    source_scores: dict[str, list[float]] = {}
    scored_questions: list[dict[str, Any]] = []

    for q in resolved:
        prob = forecasts.get(q.id, 0.5)
        bs = brier_score(prob, q.outcome)

        source_scores.setdefault(q.source, []).append(bs)
        scored_questions.append({
            "id": q.id,
            "question": q.question,
            "brier_score": bs,
            "forecast": prob,
            "outcome": q.outcome,
        })

    per_source = {
        source: sum(scores) / len(scores)
        for source, scores in source_scores.items()
    }

    scored_questions.sort(key=lambda x: x["brier_score"], reverse=True)
    worst = scored_questions[:20]

    buckets = _compute_calibration_buckets(forecasts, resolved)

    return AnalysisResult(
        per_source_scores=per_source,
        worst_questions=worst,
        calibration_buckets=buckets,
    )


def _compute_calibration_buckets(
    forecasts: dict[str, float],
    resolved: list[ResolvedQuestion],
) -> list[dict[str, Any]]:
    bucket_data: dict[int, list[tuple[float, int]]] = {i: [] for i in range(10)}

    for q in resolved:
        prob = forecasts.get(q.id, 0.5)
        bucket_idx = min(int(prob * 10), 9)
        bucket_data[bucket_idx].append((prob, q.outcome))

    buckets: list[dict[str, Any]] = []
    for i in range(10):
        lo = i / 10.0
        hi = (i + 1) / 10.0
        items = bucket_data[i]
        if items:
            predicted_mean = sum(p for p, _ in items) / len(items)
            actual_frequency = sum(o for _, o in items) / len(items)
        else:
            predicted_mean = 0.0
            actual_frequency = 0.0

        buckets.append({
            "bucket_range": f"{lo:.1f}-{hi:.1f}",
            "predicted_mean": round(predicted_mean, 4),
            "actual_frequency": round(actual_frequency, 4),
            "count": len(items),
        })

    return buckets


def save_analysis(result: AnalysisResult, output_dir: Path) -> None:
    """Write analysis results to JSON and markdown files."""
    output_dir.mkdir(parents=True, exist_ok=True)

    json_path = output_dir / "analysis.json"
    json_path.write_text(json.dumps(asdict(result), indent=2))

    md_path = output_dir / "analysis.md"
    md_path.write_text(_format_markdown(result))


def _format_markdown(result: AnalysisResult) -> str:
    lines = ["# ForecastBench Error Analysis", ""]

    lines.append("## Per-Source Brier Scores")
    lines.append("")
    lines.append("| Source | Mean Brier |")
    lines.append("|--------|-----------|")
    for source, score in sorted(result.per_source_scores.items(), key=lambda x: x[1]):
        lines.append(f"| {source} | {score:.4f} |")
    lines.append("")

    lines.append("## Worst 20 Questions (Highest Brier Score)")
    lines.append("")
    lines.append("| ID | Question | Brier | Forecast | Outcome |")
    lines.append("|----|----------|-------|----------|---------|")
    for q in result.worst_questions:
        question_text = q["question"][:60]
        lines.append(
            f"| {q['id']} | {question_text} | {q['brier_score']:.4f} "
            f"| {q['forecast']:.2f} | {q['outcome']} |"
        )
    lines.append("")

    lines.append("## Calibration")
    lines.append("")
    lines.append("| Bucket | Predicted Mean | Actual Frequency | Count |")
    lines.append("|--------|---------------|-----------------|-------|")
    for b in result.calibration_buckets:
        lines.append(
            f"| {b['bucket_range']} | {b['predicted_mean']:.4f} "
            f"| {b['actual_frequency']:.4f} | {b['count']} |"
        )
    lines.append("")

    return "\n".join(lines)
