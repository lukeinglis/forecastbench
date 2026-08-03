"""Behavioral parity checks against ForecastBench competition pipeline.

Run standalone: uv run python verify_parity.py
Exits with the count of failed checks (0 = all pass).
"""

from __future__ import annotations

import sys

from dummy_forecaster import forecast as dummy_forecast
from eval import _expand_resolved_for_horizons
from fetch_data import (
    Question,
    Resolution,
    fetch_all_question_sets,
    fetch_all_resolutions,
    fetch_resolution,
    join_resolved_questions,
    list_resolution_files,
)
from score import score_forecasts


def check_dummy_score() -> bool:
    """Dummy forecaster (always 0.5) must score overall_index == 50.0 ± 0.01.

    This catches any scoring, resolution, or pipeline bug since
    brier_score(0.5, _) == 0.25 always, and brier_index(0.25) == 50.0 exactly.
    """
    question_sets = fetch_all_question_sets()
    if not question_sets:
        print("FAIL check_dummy_score: no question sets fetched")
        return False

    qs = question_sets[0]
    resolutions = fetch_all_resolutions()
    resolved = join_resolved_questions([qs], resolutions)

    if not resolved:
        print("FAIL check_dummy_score: no resolved questions after join")
        return False

    expanded = _expand_resolved_for_horizons(resolved)

    forecasts: dict[str, float] = {}
    for rq in expanded:
        q = Question(
            id=rq.id,
            source=rq.source,
            question=rq.question,
            background=rq.background,
            resolution_criteria=rq.resolution_criteria,
            freeze_datetime=rq.freeze_datetime,
            freeze_datetime_value=rq.freeze_datetime_value,
            resolution_dates=rq.resolution_dates,
            url=rq.url,
            forecast_due_date=rq.forecast_due_date,
        )
        prob = dummy_forecast(q, resolution_date=rq.resolution_date, source=rq.source)
        forecasts[rq.id] = prob

    result = score_forecasts(forecasts, expanded, difficulty_adjusted=False)

    if abs(result.overall_index - 50.0) <= 0.01:
        print(
            f"PASS check_dummy_score: overall_index={result.overall_index:.4f} "
            f"(n={result.n_dataset + result.n_market})"
        )
        return True
    else:
        print(
            f"FAIL check_dummy_score: overall_index={result.overall_index:.4f}, "
            f"expected 50.0 ± 0.01"
        )
        return False


def _fetch_all_resolutions_as_lists() -> dict[str, list[Resolution]]:
    """Fetch all resolutions preserving every entry per question ID."""
    filenames = list_resolution_files()
    resolutions: dict[str, list[Resolution]] = {}
    for f in filenames:
        try:
            res_list = fetch_resolution(f)
            for r in res_list:
                resolutions.setdefault(r.id, []).append(r)
        except Exception:
            continue
    return resolutions


def check_resolution_outcome_diversity() -> bool:
    """Resolution entries with the same ID but different dates must have diverse outcomes.

    The PR #102 overwrite bug collapsed per-ID entries, losing outcome diversity.
    At least 30% of multi-entry IDs should have different outcomes across entries.
    """
    resolutions = _fetch_all_resolutions_as_lists()

    multi_entry_ids: dict[str, set[int | None]] = {}
    for qid, res_list in resolutions.items():
        if len(res_list) > 1:
            multi_entry_ids[qid] = {r.outcome for r in res_list}

    if not multi_entry_ids:
        print("PASS check_resolution_outcome_diversity: no multi-entry IDs found")
        return True

    diverse_count = sum(1 for outcomes in multi_entry_ids.values() if len(outcomes) > 1)
    total = len(multi_entry_ids)
    ratio = diverse_count / total if total > 0 else 0.0

    if ratio >= 0.30:
        print(
            f"PASS check_resolution_outcome_diversity: {diverse_count}/{total} "
            f"({ratio:.1%}) multi-entry IDs have diverse outcomes"
        )
        return True
    else:
        print(
            f"FAIL check_resolution_outcome_diversity: {diverse_count}/{total} "
            f"({ratio:.1%}) < 30% threshold"
        )
        return False


def check_resolution_entry_preservation() -> bool:
    """Total resolution entries must significantly exceed unique question IDs.

    If resolution fetching overwrites entries per-ID, the ratio drops to ~1.0.
    With proper preservation, the ratio should be > 3.0.
    """
    resolutions = _fetch_all_resolutions_as_lists()

    unique_ids = len(resolutions)
    total_entries = sum(len(entries) for entries in resolutions.values())

    ratio = total_entries / unique_ids if unique_ids > 0 else 0.0

    if ratio > 3.0:
        print(
            f"PASS check_resolution_entry_preservation: "
            f"{total_entries} entries / {unique_ids} IDs = {ratio:.1f}x"
        )
        return True
    else:
        print(
            f"FAIL check_resolution_entry_preservation: "
            f"{total_entries} entries / {unique_ids} IDs = {ratio:.1f}x (< 3.0)"
        )
        return False


def check_cross_round_filtering() -> bool:
    """Resolved questions must only contain resolution_dates from the original question.

    After join_resolved_questions(), every ResolvedQuestion with a resolution_date
    must have that date appear in the original question's resolution_dates list.
    This catches cross-round contamination (fixed in commit 746f545).
    """
    question_sets = fetch_all_question_sets()
    resolutions = fetch_all_resolutions()

    if not question_sets:
        print("FAIL check_cross_round_filtering: no question sets")
        return False

    resolved = join_resolved_questions(question_sets, resolutions)

    original_res_dates: dict[str, set[str]] = {}
    for qs in question_sets:
        for q in qs.questions:
            rd = q.resolution_dates
            if isinstance(rd, list):
                dates = {str(d) for d in rd if d and str(d).upper() != "N/A"}
                if dates:
                    original_res_dates[q.id] = dates

    violations = 0
    checked = 0
    for rq in resolved:
        if rq.id not in original_res_dates:
            continue
        if rq.resolution_date is None:
            continue
        checked += 1
        if rq.resolution_date not in original_res_dates[rq.id]:
            violations += 1

    if violations == 0:
        print(
            f"PASS check_cross_round_filtering: "
            f"{checked} resolution dates checked, 0 violations"
        )
        return True
    else:
        print(
            f"FAIL check_cross_round_filtering: "
            f"{violations}/{checked} resolution dates not in original question's list"
        )
        return False


def main() -> None:
    checks = [
        ("check_dummy_score", check_dummy_score),
        ("check_resolution_outcome_diversity", check_resolution_outcome_diversity),
        ("check_resolution_entry_preservation", check_resolution_entry_preservation),
        ("check_cross_round_filtering", check_cross_round_filtering),
    ]

    failures = 0
    for name, check_fn in checks:
        try:
            passed = check_fn()
            if not passed:
                failures += 1
        except Exception as e:
            print(f"FAIL {name}: {e}")
            failures += 1

    print(f"\n{len(checks) - failures}/{len(checks)} checks passed")
    sys.exit(failures)


if __name__ == "__main__":
    main()
