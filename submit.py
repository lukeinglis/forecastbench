"""Thin re-export wrapper for submission — all logic lives in forecastbench-parity."""

from forecastbench_parity.submission import (
    CoverageResult,
    SubmissionMetadata,
    assemble_submission,
    save_submission,
    upload_to_gcs,
    validate_coverage,
    validate_forecasts,
)

__all__ = [
    "CoverageResult",
    "SubmissionMetadata",
    "assemble_submission",
    "save_submission",
    "upload_to_gcs",
    "validate_coverage",
    "validate_forecasts",
]


def main() -> None:
    import argparse
    import json
    from pathlib import Path

    parser = argparse.ArgumentParser(description="ForecastBench submission tools")
    sub = parser.add_subparsers(dest="command")

    assemble_p = sub.add_parser("assemble", help="Assemble submission from latest result")
    assemble_p.add_argument("--org", required=True, help="Organization name")
    assemble_p.add_argument("--model", required=True, help="Model name")
    assemble_p.add_argument("--model-org", required=True, help="Model organization")
    assemble_p.add_argument("--result", required=True, help="Path to result JSON from eval pipeline")
    assemble_p.add_argument("--output-dir", default="submissions", help="Output directory")
    assemble_p.add_argument(
        "--track",
        choices=["baseline", "tournament"],
        default="baseline",
        help="Competition track: baseline or tournament",
    )

    validate_p = sub.add_parser("validate", help="Validate coverage of a submission")
    validate_p.add_argument("submission", help="Path to submission JSON")
    validate_p.add_argument("--threshold", type=float, default=0.95, help="Coverage threshold")

    args = parser.parse_args()

    if args.command == "assemble":
        from fetch_data import Resolution, join_resolved_questions, load_data

        result_data = json.loads(Path(args.result).read_text())
        forecasts = result_data["forecasts"]
        question_sets_used = result_data["metadata"]["question_sets_used"]

        all_qs, resolved = load_data()
        used_qs = [qs for qs in all_qs if qs.forecast_due_date in question_sets_used]
        resolutions: dict[str, list[Resolution]] = {}
        for q in resolved:
            resolutions.setdefault(q.id, []).append(
                Resolution(id=q.id, outcome=q.outcome, resolution_date=q.resolution_date)
            )
        iteration_resolved = join_resolved_questions(used_qs, resolutions)

        meta = SubmissionMetadata(
            organization=args.org,
            model=args.model,
            model_organization=args.model_org,
            question_set=question_sets_used[-1] if question_sets_used else "unknown",
            track=args.track,
        )
        submission = assemble_submission(forecasts, iteration_resolved, meta)

        coverage = validate_coverage(submission, [q for qs in used_qs for q in qs.questions])
        print(f"Market coverage:  {coverage.market_covered}/{coverage.market_total} ({coverage.market_coverage:.1%})")
        print(f"Dataset coverage: {coverage.dataset_covered}/{coverage.dataset_total} ({coverage.dataset_coverage:.1%})")
        print(f"Passes threshold: {'YES' if coverage.passes else 'NO'}")

        path = save_submission(submission, Path(args.output_dir))
        print(f"Saved to {path}")

    elif args.command == "validate":
        from fetch_data import load_data

        submission = json.loads(Path(args.submission).read_text())
        validate_forecasts(submission.get("forecasts", []))
        all_qs, _ = load_data()

        coverage = validate_coverage(
            submission, [q for qs in all_qs for q in qs.questions], threshold=args.threshold,
        )
        print(f"Market coverage:  {coverage.market_covered}/{coverage.market_total} ({coverage.market_coverage:.1%})")
        print(f"Dataset coverage: {coverage.dataset_covered}/{coverage.dataset_total} ({coverage.dataset_coverage:.1%})")
        print(f"Passes threshold: {'YES' if coverage.passes else 'NO'}")
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
