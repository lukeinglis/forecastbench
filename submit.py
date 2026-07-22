"""ForecastBench submission assembly, validation, and upload."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from fetch_data import MARKET_SOURCES, ResolvedQuestion


@dataclass
class SubmissionMetadata:
    organization: str
    model: str
    model_organization: str
    question_set: str
    sequence_number: int = 1


@dataclass
class CoverageResult:
    market_coverage: float
    dataset_coverage: float
    market_total: int
    market_covered: int
    dataset_total: int
    dataset_covered: int
    passes: bool


def validate_forecasts(entries: list[dict[str, Any]]) -> None:
    """Validate forecast entries for submission compliance.

    Raises ValueError if any entry has:
    - forecast value outside [0, 1]
    - resolution_date set for a market-source question
    """
    for entry in entries:
        prob = entry.get("forecast")
        if not isinstance(prob, (int, float)) or not math.isfinite(prob) or prob < 0 or prob > 1:
            raise ValueError(
                f"Forecast for {entry.get('id', '?')} is out of range [0, 1]: {prob}"
            )
        source = entry.get("source", "")
        if source.lower() in MARKET_SOURCES and entry.get("resolution_date") is not None:
            raise ValueError(
                f"Market question {entry.get('id', '?')} must not have resolution_date"
            )


def assemble_submission(
    forecasts: dict[str, float],
    questions: list[ResolvedQuestion],
    metadata: SubmissionMetadata,
    reasoning: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Build ForecastBench-format submission JSON.

    Schema per: https://github.com/forecastingresearch/forecastbench/wiki/How-to-submit-to-ForecastBench
    """
    entries = []
    for q in questions:
        prob = forecasts.get(q.id, 0.5)
        entry: dict[str, Any] = {
            "id": q.id,
            "source": q.source,
            "forecast": prob,
        }
        if q.resolution_date and q.source.lower() not in MARKET_SOURCES:
            entry["resolution_date"] = q.resolution_date
        if reasoning and q.id in reasoning:
            entry["reasoning"] = reasoning[q.id]
        entries.append(entry)

    validate_forecasts(entries)

    return {
        "organization": metadata.organization,
        "model": metadata.model,
        "model_organization": metadata.model_organization,
        "question_set": metadata.question_set,
        "forecasts": entries,
    }


def validate_coverage(
    submission: dict[str, Any],
    questions: list[ResolvedQuestion],
    threshold: float = 0.95,
) -> CoverageResult:
    """Check that submission covers 95%+ of market and dataset questions."""
    forecast_ids = {f["id"] for f in submission["forecasts"]}
    market_qs = [q for q in questions if q.source.lower() in MARKET_SOURCES]
    dataset_qs = [q for q in questions if q.source.lower() not in MARKET_SOURCES]

    mkt_covered = sum(1 for q in market_qs if q.id in forecast_ids)
    ds_covered = sum(1 for q in dataset_qs if q.id in forecast_ids)
    mkt_cov = mkt_covered / len(market_qs) if market_qs else 1.0
    ds_cov = ds_covered / len(dataset_qs) if dataset_qs else 1.0

    return CoverageResult(
        market_coverage=mkt_cov,
        dataset_coverage=ds_cov,
        market_total=len(market_qs),
        market_covered=mkt_covered,
        dataset_total=len(dataset_qs),
        dataset_covered=ds_covered,
        passes=mkt_cov >= threshold and ds_cov >= threshold,
    )


def save_submission(
    submission: dict[str, Any],
    output_dir: Path = Path("submissions"),
) -> Path:
    """Save submission JSON to local staging directory.

    File naming per ForecastBench: {forecast_due_date}.{organization}.{N}.json
    """
    due_date = submission["question_set"]
    org = submission["organization"]
    # Determine sequence number: count existing files for this date+org
    output_dir.mkdir(parents=True, exist_ok=True)
    existing = list(output_dir.glob(f"{due_date}.{org}.*.json"))
    n = len(existing) + 1
    if n > 3:
        raise ValueError(
            f"ForecastBench allows max 3 submissions per round, found {len(existing)} existing"
        )
    filename = f"{due_date}.{org}.{n}.json"
    path = output_dir / filename
    path.write_text(json.dumps(submission, indent=2))
    return path


def upload_to_gcs(
    path: Path,
    bucket: str,
    folder: str = "",
) -> str:
    """Upload submission to GCS bucket. Requires google-cloud-storage.

    Install with: uv pip install 'forecastbench[gcs]'
    """
    from google.cloud import storage

    client = storage.Client()
    blob_name = f"{folder}/{path.name}" if folder else path.name
    blob = client.bucket(bucket).blob(blob_name)
    blob.upload_from_filename(str(path))
    return f"gs://{bucket}/{blob_name}"


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="ForecastBench submission tools")
    sub = parser.add_subparsers(dest="command")

    assemble_p = sub.add_parser("assemble", help="Assemble submission from latest result")
    assemble_p.add_argument("--org", required=True, help="Organization name")
    assemble_p.add_argument("--model", required=True, help="Model name")
    assemble_p.add_argument("--model-org", required=True, help="Model organization")
    assemble_p.add_argument("--result", required=True, help="Path to result JSON from eval pipeline")
    assemble_p.add_argument("--output-dir", default="submissions", help="Output directory")

    validate_p = sub.add_parser("validate", help="Validate coverage of a submission")
    validate_p.add_argument("submission", help="Path to submission JSON")
    validate_p.add_argument("--threshold", type=float, default=0.95, help="Coverage threshold")

    args = parser.parse_args()

    if args.command == "assemble":
        result_data = json.loads(Path(args.result).read_text())
        forecasts = result_data["forecasts"]
        question_sets_used = result_data["metadata"]["question_sets_used"]

        from fetch_data import load_data, join_resolved_questions, Resolution
        all_qs, resolved = load_data()
        used_qs = [qs for qs in all_qs if qs.forecast_due_date in question_sets_used]
        resolutions = {q.id: Resolution(id=q.id, outcome=q.outcome, resolution_date=q.resolution_date)
                       for q in resolved}
        iteration_resolved = join_resolved_questions(used_qs, resolutions)

        meta = SubmissionMetadata(
            organization=args.org,
            model=args.model,
            model_organization=args.model_org,
            question_set=question_sets_used[-1] if question_sets_used else "unknown",
        )
        submission = assemble_submission(forecasts, iteration_resolved, meta)

        coverage = validate_coverage(submission, iteration_resolved)
        print(f"Market coverage:  {coverage.market_covered}/{coverage.market_total} ({coverage.market_coverage:.1%})")
        print(f"Dataset coverage: {coverage.dataset_covered}/{coverage.dataset_total} ({coverage.dataset_coverage:.1%})")
        print(f"Passes threshold: {'YES' if coverage.passes else 'NO'}")

        path = save_submission(submission, Path(args.output_dir))
        print(f"Saved to {path}")

    elif args.command == "validate":
        submission = json.loads(Path(args.submission).read_text())
        validate_forecasts(submission.get("forecasts", []))
        from fetch_data import load_data, join_resolved_questions, Resolution
        all_qs, resolved = load_data()
        resolutions = {q.id: Resolution(id=q.id, outcome=q.outcome, resolution_date=q.resolution_date)
                       for q in resolved}
        all_resolved = join_resolved_questions(all_qs, resolutions)

        coverage = validate_coverage(submission, all_resolved, threshold=args.threshold)
        print(f"Market coverage:  {coverage.market_covered}/{coverage.market_total} ({coverage.market_coverage:.1%})")
        print(f"Dataset coverage: {coverage.dataset_covered}/{coverage.dataset_total} ({coverage.dataset_coverage:.1%})")
        print(f"Passes threshold: {'YES' if coverage.passes else 'NO'}")
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
