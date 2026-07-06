"""Error analysis and reporting for ForecastBench evaluation runs."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from pydantic import BaseModel

from fetch_data import ResolvedQuestion
from score import brier_score, brier_index


class QuestionResult(BaseModel):
    question_id: str
    question_text: str
    source: str
    forecast: float
    outcome: int
    brier_score: float
    error_magnitude: float
    error_type: str
    direction: str


class CalibrationBin(BaseModel):
    range_low: float
    range_high: float
    n_questions: int
    mean_forecast: float
    actual_rate: float


class AggregateMetrics(BaseModel):
    mean_brier: float
    brier_index: float
    n_questions: int
    n_overconfident: int
    n_underconfident: int
    n_well_calibrated: int
    mean_abs_error: float


class RunResult(BaseModel):
    model: str
    timestamp: str
    aggregate: AggregateMetrics
    questions: list[QuestionResult]
    calibration: list[CalibrationBin]
    worst_predictions: list[QuestionResult]


def classify_error(forecast: float, outcome: int) -> str:
    error = abs(forecast - outcome)
    predicted_yes = forecast >= 0.5
    actual_yes = outcome == 1

    if error <= 0.3 + 1e-9 and predicted_yes == actual_yes:
        return "well_calibrated"
    if predicted_yes != actual_yes:
        return "overconfident"
    return "underconfident"


def compute_calibration(results: list[QuestionResult], n_bins: int = 10) -> list[CalibrationBin]:
    bins: list[CalibrationBin] = []
    bin_width = 1.0 / n_bins

    for i in range(n_bins):
        low = i * bin_width
        high = (i + 1) * bin_width
        in_bin = [r for r in results if low <= r.forecast < high or (i == n_bins - 1 and r.forecast == high)]

        if in_bin:
            mean_f = sum(r.forecast for r in in_bin) / len(in_bin)
            actual = sum(r.outcome for r in in_bin) / len(in_bin)
        else:
            mean_f = (low + high) / 2
            actual = 0.0

        bins.append(
            CalibrationBin(
                range_low=low,
                range_high=high,
                n_questions=len(in_bin),
                mean_forecast=mean_f,
                actual_rate=actual,
            )
        )

    return bins


def analyze_run(
    forecasts: dict[str, float],
    resolved: list[ResolvedQuestion],
    model: str,
) -> RunResult:
    results: list[QuestionResult] = []

    for q in resolved:
        f = forecasts.get(q.id, 0.5)
        bs = brier_score(f, q.outcome)
        error_mag = abs(f - q.outcome)
        error_type = classify_error(f, q.outcome)
        direction = "over" if f > q.outcome else ("under" if f < q.outcome else "exact")

        results.append(
            QuestionResult(
                question_id=q.id,
                question_text=q.question,
                source=q.source,
                forecast=f,
                outcome=q.outcome,
                brier_score=bs,
                error_magnitude=error_mag,
                error_type=error_type,
                direction=direction,
            )
        )

    brier_scores = [r.brier_score for r in results]
    mean_bs = sum(brier_scores) / len(brier_scores) if brier_scores else 0.0
    bi = brier_index(mean_bs) if brier_scores else 0.0

    n_over = sum(1 for r in results if r.error_type == "overconfident")
    n_under = sum(1 for r in results if r.error_type == "underconfident")
    n_well = sum(1 for r in results if r.error_type == "well_calibrated")
    mean_abs = sum(r.error_magnitude for r in results) / len(results) if results else 0.0

    aggregate = AggregateMetrics(
        mean_brier=mean_bs,
        brier_index=bi,
        n_questions=len(results),
        n_overconfident=n_over,
        n_underconfident=n_under,
        n_well_calibrated=n_well,
        mean_abs_error=mean_abs,
    )

    calibration = compute_calibration(results)

    worst = sorted(results, key=lambda r: r.brier_score, reverse=True)[:10]

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")

    return RunResult(
        model=model,
        timestamp=timestamp,
        aggregate=aggregate,
        questions=results,
        calibration=calibration,
        worst_predictions=worst,
    )


def save_results(result: RunResult, output_dir: str = "results") -> str:
    path = Path(output_dir)
    path.mkdir(parents=True, exist_ok=True)
    filename = f"run_{result.timestamp}.json"
    filepath = path / filename
    filepath.write_text(result.model_dump_json(indent=2))
    return str(filepath)


def print_summary(result: RunResult) -> None:
    agg = result.aggregate
    print("=" * 60)
    print(f"ForecastBench Baseline Run: {result.model}")
    print("=" * 60)
    print(f"  Questions:       {agg.n_questions}")
    print(f"  Mean Brier:      {agg.mean_brier:.4f}")
    print(f"  Brier Index:     {agg.brier_index:.1f}%")
    print(f"  Mean Abs Error:  {agg.mean_abs_error:.4f}")
    print(f"  Well calibrated: {agg.n_well_calibrated}")
    print(f"  Overconfident:   {agg.n_overconfident}")
    print(f"  Underconfident:  {agg.n_underconfident}")
    print("-" * 60)
    print("Worst predictions:")
    for w in result.worst_predictions[:5]:
        print(f"  [{w.source}] {w.question_text[:60]}...")
        print(f"    forecast={w.forecast:.2f}  outcome={w.outcome}  brier={w.brier_score:.4f}")
    print("=" * 60)
