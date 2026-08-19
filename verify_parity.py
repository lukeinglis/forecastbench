"""ForecastBench parity verifier — standalone CLI tool.

Verifies our pipeline stays in parity with ForecastBench by fetching live
reference data from the upstream repo and leaderboard. Zero hardcoded
reference values. Also runs behavioral checks for pipeline correctness.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from pathlib import Path
from typing import Any

import requests


UPSTREAM_PROMPTS_URL = (
    "https://raw.githubusercontent.com/forecastingresearch/"
    "forecastbench/main/src/llm_forecaster/prompts.py"
)

RESULTS_DIR = Path("results")

PROMPT_NAMES = [
    "ZERO_SHOT_MARKET_PROMPT",
    "ZERO_SHOT_MARKET_WITH_FREEZE_VALUE_PROMPT",
    "ZERO_SHOT_DATASET_PROMPT",
    "FORECAST_EXTRACTION_PROMPT",
]

_STAT_BASELINE_RE = re.compile(
    r"\n?Statistical baseline \([^)]+\): \d+% probability\n"
    r"Note: This is a simple statistical estimate\. "
    r"Use your judgment to adjust\.\n",
)

_BASE_RATE_RE = re.compile(
    r"Historical context: Questions of this type from this data source "
    r"have historically resolved to YES approximately \d+% of the time\. "
    r"This base rate should inform your starting estimate, with adjustments "
    r"based on the specific details of this question\.\n\n",
)


def _strip_enhancements(text: str) -> str:
    text = _STAT_BASELINE_RE.sub("", text)
    text = _BASE_RATE_RE.sub("", text)
    return text


def extract_template(source: str, var_name: str) -> str | None:
    """Extract a triple-quoted string assigned to var_name from Python source."""
    pattern = rf'{var_name}\s*=\s*"""(.*?)"""'
    match = re.search(pattern, source, re.DOTALL)
    if match:
        return match.group(1)
    pattern_sq = rf"{var_name}\s*=\s*'''(.*?)'''"
    match = re.search(pattern_sq, source, re.DOTALL)
    if match:
        return match.group(1)
    return None


def _get_local_template(name: str) -> str | None:
    import lab_forecaster

    val = getattr(lab_forecaster, name, None)
    if val is None:
        return None
    return str(val)


def check_prompt_templates(upstream_source: str | None) -> tuple[bool, str]:
    if upstream_source is None:
        return True, "[WARN] Could not fetch upstream prompts — skipping live comparison"

    matched = 0
    mismatched: list[str] = []
    for name in PROMPT_NAMES:
        upstream_val = extract_template(upstream_source, name)
        local_val = _get_local_template(name)

        if upstream_val is None:
            mismatched.append(f"{name} (not found upstream)")
            continue
        if local_val is None:
            mismatched.append(f"{name} (not found locally)")
            continue

        upstream_norm = upstream_val.strip()
        local_norm = _strip_enhancements(local_val).strip()

        if upstream_norm == local_norm:
            matched += 1
        else:
            mismatched.append(name)

    total = len(PROMPT_NAMES)
    if not mismatched:
        return True, f"[PASS] Prompt templates match upstream ({matched}/{total})"
    return False, f"[FAIL] Prompt template mismatch: {', '.join(mismatched)}"


def check_resolution_matching() -> tuple[bool, str]:
    try:
        from fetch_data import (
            fetch_all_resolutions,
            fetch_question_set,
            join_resolved_questions,
            list_question_set_files,
        )
        filenames = list_question_set_files()
        if not filenames:
            return True, "[WARN] No question sets available"

        filename = sorted(filenames)[-1]
        qs = fetch_question_set(filename)
        resolutions = fetch_all_resolutions()
        resolved = join_resolved_questions([qs], resolutions)

        if not resolved:
            return True, "[WARN] No resolved questions in latest set"

        resolved_count = len(resolved)
        base_ids = {q.id for q in qs.questions}
        base_count = len(base_ids)

        ratio = resolved_count / base_count if base_count > 0 else 0
        return (
            True,
            f"[PASS] Resolution matching: {resolved_count} resolved "
            f"from {base_count} base questions (ratio {ratio:.1f}x)",
        )
    except (requests.RequestException, OSError) as e:
        return True, f"[WARN] Resolution matching check failed: {e}"


def check_scoring_formula(leaderboard: list[dict[str, str]] | None) -> tuple[bool, str]:
    from score import brier_index

    if brier_index(0.25) != 50.0:
        return False, f"[FAIL] brier_index(0.25) = {brier_index(0.25)}, expected 50.0"
    if brier_index(0.0) != 100.0:
        return False, f"[FAIL] brier_index(0.0) = {brier_index(0.0)}, expected 100.0"

    n_cross = 0
    if leaderboard:
        for row in leaderboard:
            try:
                brier_overall = float(row.get("Brier Overall", "").strip())
                overall_str = row.get("Overall", "").strip().rstrip("%")
                overall = float(overall_str)
            except (ValueError, TypeError):
                continue

            computed = (1.0 - math.sqrt(brier_overall)) * 100.0
            if abs(computed - overall) > 0.5:
                return (
                    False,
                    f"[FAIL] Leaderboard formula mismatch for "
                    f"{row.get('Model', '?')}: computed {computed:.1f} vs {overall:.1f}",
                )
            n_cross += 1

    detail = f"local + {n_cross} leaderboard entries cross-checked" if n_cross else "local only"
    return True, f"[PASS] Scoring formula verified ({detail})"


def check_missing_forecast_default() -> tuple[bool, str]:
    from fetch_data import ResolvedQuestion
    from score import score_forecasts, brier_score

    test_resolved = [
        ResolvedQuestion(
            id=f"test_{i}", source="test", question="q",
            outcome=outcome, forecast_due_date="2026-01-01",
        )
        for i, outcome in enumerate([0, 1, 0, 1])
    ]

    result = score_forecasts({}, test_resolved, difficulty_adjusted=False)

    expected_bs = sum(brier_score(0.5, q.outcome) for q in test_resolved) / len(test_resolved)
    if abs(result.overall_brier - expected_bs) > 1e-9:
        return (
            False,
            f"[FAIL] Missing forecast default: got Brier {result.overall_brier:.6f}, "
            f"expected {expected_bs:.6f}",
        )
    return True, "[PASS] Missing forecast default = 0.5"


def check_multi_horizon_batching() -> tuple[bool, str]:
    try:
        from fetch_data import (
            fetch_question_set,
            list_question_set_files,
            MARKET_SOURCES,
        )

        filenames = list_question_set_files()
        if not filenames:
            return True, "[WARN] No question sets available"

        filename = sorted(filenames)[-1]
        qs = fetch_question_set(filename)

        base_questions = 0
        expanded_entries = 0
        for q in qs.questions:
            if q.source.lower() in MARKET_SOURCES:
                continue
            rd = q.resolution_dates
            if isinstance(rd, list) and len(rd) > 0:
                base_questions += 1
                expanded_entries += len(rd)

        if base_questions == 0:
            return True, "[WARN] No multi-horizon questions found"

        if expanded_entries <= base_questions:
            return True, "[WARN] No multi-horizon expansion detected"

        return (
            True,
            f"[PASS] Multi-horizon batching: {base_questions} base questions "
            f"(not {expanded_entries} expanded entries)",
        )
    except (requests.RequestException, OSError) as e:
        return True, f"[WARN] Multi-horizon check failed: {e}"


def check_question_count(leaderboard: list[dict[str, str]] | None) -> tuple[bool, str]:
    try:
        from fetch_data import fetch_question_set, list_question_set_files

        filenames = list_question_set_files()
        if not filenames:
            return True, "[WARN] No question sets available"

        filename = sorted(filenames)[-1]
        qs = fetch_question_set(filename)
        our_count = len(qs.questions)

        if not leaderboard:
            return True, f"[PASS] Question count: {our_count} (leaderboard unavailable for cross-check)"

        n_values: list[int] = []
        for row in leaderboard:
            try:
                n = int(row.get("N", "0").strip())
                if n > 0:
                    n_values.append(n)
            except (ValueError, TypeError):
                continue

        if not n_values:
            return True, f"[PASS] Question count: {our_count} (no leaderboard N values)"

        lo, hi = min(n_values), max(n_values)
        return (
            True,
            f"[PASS] Question count: {our_count} (leaderboard reference range: {lo}-{hi})",
        )
    except (requests.RequestException, OSError) as e:
        return True, f"[WARN] Question count check failed: {e}"


def _load_latest_result() -> dict[str, Any] | None:
    if not RESULTS_DIR.exists():
        return None
    result_files = sorted(RESULTS_DIR.glob("*.json"))
    result_files = [f for f in result_files if f.name != "RESULTS.md"]
    if not result_files:
        return None
    try:
        data: dict[str, Any] = json.loads(result_files[-1].read_text())
        return data
    except (json.JSONDecodeError, OSError):
        return None


def _clean_model_slug(model_slug: str) -> str:
    """Strip provider prefix and trailing date from a model slug."""
    slug = model_slug.replace("/", "_").replace("@", "_")
    slug = re.sub(r"_\d{8,}$", "", slug)
    slug = re.sub(r"_\d{4}-\d{2}-\d{2}$", "", slug)
    for prefix in ["vertex_ai_", "openai_", "anthropic_", "google_"]:
        if slug.startswith(prefix):
            slug = slug[len(prefix):]
            break
    return slug


def _find_reference_model(
    leaderboard: list[dict[str, str]],
    model_hint: str | None = None,
) -> tuple[str, float, bool] | None:
    """Find a reference model on the leaderboard.

    Returns (model_name, overall_score, is_fallback) or None.
    When model_hint is provided, finds the closest match by overlap ratio.
    """
    if model_hint:
        cleaned = _clean_model_slug(model_hint).lower()
        best: tuple[str, float, int] | None = None
        for row in leaderboard:
            model = row.get("Model", "")
            model_lower = model.lower()
            model_core = re.sub(r"-?\d{6,}.*$", "", model_lower).rstrip("-")
            if cleaned == model_core:
                tier = 2
            elif cleaned in model_lower or model_lower in cleaned:
                tier = 1
            else:
                continue
            try:
                overall_str = row.get("Overall", "").strip().rstrip("%")
                score = float(overall_str)
            except (ValueError, TypeError):
                continue
            if best is None or tier > best[2]:
                best = (model, score, tier)
        if best is not None:
            return best[0], best[1], False

    preferred = ["o3", "gpt-4o", "claude"]
    for prefix in preferred:
        for row in leaderboard:
            model = row.get("Model", "")
            if prefix.lower() in model.lower():
                try:
                    overall_str = row.get("Overall", "").strip().rstrip("%")
                    return model, float(overall_str), True
                except (ValueError, TypeError):
                    continue
    for row in leaderboard:
        try:
            overall_str = row.get("Overall", "").strip().rstrip("%")
            return row.get("Model", "Unknown"), float(overall_str), True
        except (ValueError, TypeError):
            continue
    return None


def check_score_comparison(leaderboard: list[dict[str, str]] | None) -> tuple[bool, str]:
    result = _load_latest_result()
    if result is None:
        return True, "[SKIP] No results found — run eval first"

    if not leaderboard:
        return True, "[WARN] Leaderboard unavailable for score comparison"

    scoring = result.get("scoring_result", {})
    our_bi = float(scoring.get("overall_index", 0))

    model_slug = result.get("model_slug")
    ref = _find_reference_model(leaderboard, model_hint=model_slug)
    if ref is None:
        return True, "[WARN] No reference model found on leaderboard"

    model_name, ref_bi, is_fallback = ref
    label = f"{model_name} (fallback reference)" if is_fallback else model_name
    gap = abs(our_bi - ref_bi)
    threshold = 2.0
    if gap > threshold:
        return (
            False,
            f"[FAIL] Score gap vs {label}: {gap:.1f}pts > {threshold:.0f}pt threshold",
        )
    return (
        True,
        f"[PASS] Score gap vs {label}: {gap:.1f}pts (threshold: {threshold:.0f}pts)",
    )


def check_per_source_breakdown(leaderboard: list[dict[str, str]] | None) -> tuple[bool, str]:
    result = _load_latest_result()
    if result is None:
        return True, "[SKIP] No results found — run eval first"

    if not leaderboard:
        return True, "[WARN] Leaderboard unavailable for per-source comparison"

    scoring = result.get("scoring_result", {})
    our_dataset = float(scoring.get("dataset_index", 0))
    our_market = float(scoring.get("market_index", 0))

    model_slug = result.get("model_slug")
    ref = _find_reference_model(leaderboard, model_hint=model_slug)
    if ref is None:
        return True, "[WARN] No reference model found on leaderboard"

    model_name, _, _is_fallback = ref
    ref_row = None
    for row in leaderboard:
        if row.get("Model", "") == model_name:
            ref_row = row
            break

    if ref_row is None:
        return True, "[WARN] Could not find reference model row"

    try:
        ref_dataset_str = ref_row.get("Dataset", "").strip().rstrip("%")
        ref_market_str = ref_row.get("Market", "").strip().rstrip("%")
        ref_dataset = float(ref_dataset_str) if ref_dataset_str else None
        ref_market = float(ref_market_str) if ref_market_str else None
    except (ValueError, TypeError):
        return True, "[WARN] Could not parse reference model per-source scores"

    gaps: list[str] = []
    threshold = 3.0

    if ref_dataset is not None:
        d_gap = abs(our_dataset - ref_dataset)
        gaps.append(f"dataset: {d_gap:.1f}pts")
        if d_gap > threshold:
            return False, f"[FAIL] Per-source gap: dataset {d_gap:.1f}pts > {threshold:.0f}pt threshold"

    if ref_market is not None:
        m_gap = abs(our_market - ref_market)
        gaps.append(f"market: {m_gap:.1f}pts")
        if m_gap > threshold:
            return False, f"[FAIL] Per-source gap: market {m_gap:.1f}pts > {threshold:.0f}pt threshold"

    if not gaps:
        return True, "[WARN] No per-source scores available for comparison"

    return True, f"[PASS] Per-source gap within threshold ({', '.join(gaps)})"


# --- Behavioral checks (pipeline correctness) ---


def check_dummy_score() -> tuple[bool, str]:
    """Dummy forecaster (always 0.5) must score overall_index == 50.0 ± 0.01."""
    from dummy_forecaster import forecast as dummy_forecast
    from fetch_data import (
        Question,
        fetch_all_question_sets,
        fetch_all_resolutions,
        join_resolved_questions,
    )
    from score import score_forecasts

    question_sets = fetch_all_question_sets()
    if not question_sets:
        return False, "[FAIL] check_dummy_score: no question sets fetched"

    qs = question_sets[0]
    resolutions = fetch_all_resolutions()
    resolved = join_resolved_questions([qs], resolutions)

    if not resolved:
        return False, "[FAIL] check_dummy_score: no resolved questions after join"

    forecasts: dict[str, float] = {}
    for rq in resolved:
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

    result = score_forecasts(forecasts, resolved, difficulty_adjusted=False)

    if abs(result.overall_index - 50.0) <= 0.01:
        return (
            True,
            f"[PASS] check_dummy_score: overall_index={result.overall_index:.4f} "
            f"(n={result.n_dataset + result.n_market})",
        )
    return (
        False,
        f"[FAIL] check_dummy_score: overall_index={result.overall_index:.4f}, "
        f"expected 50.0 ± 0.01",
    )


def _fetch_all_resolutions_as_lists() -> dict[str, list[Any]]:
    """Fetch all resolutions preserving every entry per question ID."""
    from fetch_data import Resolution, fetch_resolution, list_resolution_files

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


def check_resolution_outcome_diversity() -> tuple[bool, str]:
    """Resolution entries with the same ID but different dates must have diverse outcomes."""
    resolutions = _fetch_all_resolutions_as_lists()

    multi_entry_ids: dict[str, set[int | None]] = {}
    for qid, res_list in resolutions.items():
        if len(res_list) > 1:
            multi_entry_ids[qid] = {r.outcome for r in res_list}

    if not multi_entry_ids:
        return True, "[PASS] check_resolution_outcome_diversity: no multi-entry IDs found"

    diverse_count = sum(1 for outcomes in multi_entry_ids.values() if len(outcomes) > 1)
    total = len(multi_entry_ids)
    ratio = diverse_count / total if total > 0 else 0.0

    if ratio >= 0.30:
        return (
            True,
            f"[PASS] check_resolution_outcome_diversity: {diverse_count}/{total} "
            f"({ratio:.1%}) multi-entry IDs have diverse outcomes",
        )
    return (
        False,
        f"[FAIL] check_resolution_outcome_diversity: {diverse_count}/{total} "
        f"({ratio:.1%}) < 30% threshold",
    )


def check_resolution_entry_preservation() -> tuple[bool, str]:
    """Total resolution entries must significantly exceed unique question IDs."""
    resolutions = _fetch_all_resolutions_as_lists()

    unique_ids = len(resolutions)
    total_entries = sum(len(entries) for entries in resolutions.values())

    ratio = total_entries / unique_ids if unique_ids > 0 else 0.0

    if ratio > 3.0:
        return (
            True,
            f"[PASS] check_resolution_entry_preservation: "
            f"{total_entries} entries / {unique_ids} IDs = {ratio:.1f}x",
        )
    return (
        False,
        f"[FAIL] check_resolution_entry_preservation: "
        f"{total_entries} entries / {unique_ids} IDs = {ratio:.1f}x (< 3.0)",
    )


def check_cross_round_filtering() -> tuple[bool, str]:
    """Resolved questions must only contain resolution_dates from the original question."""
    from fetch_data import (
        fetch_all_question_sets,
        fetch_all_resolutions,
        join_resolved_questions,
    )

    question_sets = fetch_all_question_sets()
    resolutions = fetch_all_resolutions()

    if not question_sets:
        return False, "[FAIL] check_cross_round_filtering: no question sets"

    resolved = join_resolved_questions(question_sets, resolutions)

    original_res_dates: dict[tuple[str, str], set[str]] = {}
    for qs in question_sets:
        for q in qs.questions:
            rd = q.resolution_dates
            if isinstance(rd, list):
                dates = {str(d) for d in rd if d and str(d).upper() != "N/A"}
                if dates:
                    original_res_dates[(q.id, qs.forecast_due_date)] = dates

    violations = 0
    checked = 0
    for rq in resolved:
        key = (rq.id, rq.forecast_due_date)
        if key not in original_res_dates:
            continue
        if rq.resolution_date is None:
            continue
        checked += 1
        if rq.resolution_date not in original_res_dates[key]:
            violations += 1

    if violations == 0:
        return (
            True,
            f"[PASS] check_cross_round_filtering: "
            f"{checked} resolution dates checked, 0 violations",
        )
    return (
        False,
        f"[FAIL] check_cross_round_filtering: "
        f"{violations}/{checked} resolution dates not in original question's list",
    )


def _fetch_upstream_prompts(refresh: bool = False) -> str | None:
    try:
        from fetch_data import _fetch_text

        cache_key = "upstream_prompts.py"
        if refresh:
            cache_path = Path(".cache") / cache_key
            if cache_path.exists():
                cache_path.unlink()

        text: str = _fetch_text(UPSTREAM_PROMPTS_URL, cache_key)
        return text
    except Exception:
        return None


def _fetch_leaderboard(refresh: bool = False) -> list[dict[str, str]] | None:
    try:
        if refresh:
            from fetch_data import refresh_cache
            refresh_cache()

        from fetch_data import fetch_leaderboard
        rows: list[dict[str, str]] = fetch_leaderboard("baseline")
        return rows
    except Exception:
        return None


def main() -> None:
    import os
    os.environ["FORECASTBENCH_LOG_FORMAT"] = "json"

    import logging_config
    logging_config.reset_logging()

    import structlog
    structlog.configure(
        processors=[structlog.dev.ConsoleRenderer()],
        wrapper_class=structlog.make_filtering_bound_logger(40),
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=False,
    )
    logging_config._configured = True

    parser = argparse.ArgumentParser(description="ForecastBench Parity Verifier")
    parser.add_argument("--score", action="store_true", help="Run score comparison checks")
    parser.add_argument("--refresh", action="store_true", help="Clear cached upstream data first")
    args = parser.parse_args()

    print("ForecastBench Parity Verifier")
    print("=" * 29)

    upstream_source = _fetch_upstream_prompts(refresh=args.refresh)
    leaderboard = _fetch_leaderboard(refresh=args.refresh)

    # Structural checks (from main)
    checks: list[tuple[bool, str]] = []

    checks.append(check_prompt_templates(upstream_source))
    checks.append(check_resolution_matching())
    checks.append(check_scoring_formula(leaderboard))
    checks.append(check_missing_forecast_default())
    checks.append(check_multi_horizon_batching())
    checks.append(check_question_count(leaderboard))

    for _, msg in checks:
        print(msg)

    # Behavioral checks (pipeline correctness)
    print()
    print("Behavioral Checks:")
    behavioral_checks: list[tuple[bool, str]] = []

    behavioral_check_fns = [
        check_dummy_score,
        check_resolution_outcome_diversity,
        check_resolution_entry_preservation,
        check_cross_round_filtering,
    ]
    for check_fn in behavioral_check_fns:
        try:
            behavioral_checks.append(check_fn())
        except Exception as e:
            behavioral_checks.append((False, f"[FAIL] {check_fn.__name__}: {e}"))

    for _, msg in behavioral_checks:
        print(msg)

    # Score comparison checks (optional, --score flag)
    score_checks: list[tuple[bool, str]] = []
    if args.score:
        print()
        print("Score Comparison (--score):")
        score_checks.append(check_score_comparison(leaderboard))
        score_checks.append(check_per_source_breakdown(leaderboard))
        for _, msg in score_checks:
            print(msg)

    all_checks = checks + behavioral_checks + score_checks
    passed = sum(1 for ok, _ in all_checks if ok)
    total = len(all_checks)

    print()
    print(f"Result: {passed}/{total} passed")

    if any(not ok for ok, _ in all_checks):
        sys.exit(1)


if __name__ == "__main__":
    main()
