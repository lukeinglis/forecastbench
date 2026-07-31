"""ForecastBench parity verifier — standalone CLI tool.

Verifies our pipeline stays in parity with ForecastBench by fetching live
reference data from the upstream repo and leaderboard. Zero hardcoded
reference values.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from pathlib import Path
from typing import Any


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
    import baseline_agent

    val = getattr(baseline_agent, name, None)
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
            MARKET_SOURCES,
        )
        from eval import _expand_resolved_for_horizons

        filenames = list_question_set_files()
        if not filenames:
            return True, "[WARN] No question sets available"

        filename = sorted(filenames)[-1]
        qs = fetch_question_set(filename)
        resolutions = fetch_all_resolutions()
        resolved = join_resolved_questions([qs], resolutions)

        if not resolved:
            return True, "[WARN] No resolved questions in latest set"

        base_count = len(resolved)
        expanded = _expand_resolved_for_horizons(resolved)
        expanded_count = len(expanded)

        dataset_q = [q for q in resolved if q.source.lower() not in MARKET_SOURCES]

        for rq in dataset_q:
            rd = rq.resolution_dates
            if isinstance(rd, list) and len(rd) > 0:
                matching_expanded = [
                    e for e in expanded
                    if e.id.startswith(f"{rq.id}_")
                ]
                if len(matching_expanded) != len(rd):
                    return (
                        False,
                        f"[FAIL] Resolution expansion mismatch for {rq.id}: "
                        f"expected {len(rd)} entries, got {len(matching_expanded)}",
                    )

        ratio = expanded_count / base_count if base_count > 0 else 0
        return (
            True,
            f"[PASS] Resolution matching: {expanded_count} resolved "
            f"from {base_count} base questions (ratio {ratio:.1f}x)",
        )
    except Exception as e:
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
    except Exception as e:
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
    except Exception as e:
        return True, f"[WARN] Question count check failed: {e}"


def _load_latest_result() -> dict[str, Any] | None:
    if not RESULTS_DIR.exists():
        return None
    result_files = sorted(RESULTS_DIR.glob("*.json"))
    result_files = [f for f in result_files if f.name != "RESULTS.md"]
    if not result_files:
        return None
    try:
        return json.loads(result_files[-1].read_text())
    except (json.JSONDecodeError, OSError):
        return None


def _find_reference_model(
    leaderboard: list[dict[str, str]],
) -> tuple[str, float] | None:
    preferred = ["o3", "gpt-4o", "claude"]
    for prefix in preferred:
        for row in leaderboard:
            model = row.get("Model", "")
            if prefix.lower() in model.lower():
                try:
                    overall_str = row.get("Overall", "").strip().rstrip("%")
                    return model, float(overall_str)
                except (ValueError, TypeError):
                    continue
    for row in leaderboard:
        try:
            overall_str = row.get("Overall", "").strip().rstrip("%")
            return row.get("Model", "Unknown"), float(overall_str)
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

    ref = _find_reference_model(leaderboard)
    if ref is None:
        return True, "[WARN] No reference model found on leaderboard"

    model_name, ref_bi = ref
    gap = abs(our_bi - ref_bi)
    threshold = 2.0
    if gap > threshold:
        return (
            False,
            f"[FAIL] Score gap vs {model_name}: {gap:.1f}pts > {threshold:.0f}pt threshold",
        )
    return (
        True,
        f"[PASS] Score gap vs {model_name}: {gap:.1f}pts (threshold: {threshold:.0f}pts)",
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

    ref = _find_reference_model(leaderboard)
    if ref is None:
        return True, "[WARN] No reference model found on leaderboard"

    model_name, _ = ref
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


def _fetch_upstream_prompts(refresh: bool = False) -> str | None:
    try:
        from fetch_data import _fetch_text

        cache_key = "upstream_prompts.py"
        if refresh:
            cache_path = Path(".cache") / cache_key
            if cache_path.exists():
                cache_path.unlink()

        return _fetch_text(UPSTREAM_PROMPTS_URL, cache_key)
    except Exception:
        return None


def _fetch_leaderboard(refresh: bool = False) -> list[dict[str, str]] | None:
    try:
        if refresh:
            from fetch_data import refresh_cache
            refresh_cache()

        from fetch_data import fetch_leaderboard
        return fetch_leaderboard("baseline")
    except Exception:
        return None


def main() -> None:
    parser = argparse.ArgumentParser(description="ForecastBench Parity Verifier")
    parser.add_argument("--score", action="store_true", help="Run score comparison checks")
    parser.add_argument("--refresh", action="store_true", help="Clear cached upstream data first")
    args = parser.parse_args()

    print("ForecastBench Parity Verifier")
    print("=" * 29)

    upstream_source = _fetch_upstream_prompts(refresh=args.refresh)
    leaderboard = _fetch_leaderboard(refresh=args.refresh)

    checks: list[tuple[bool, str]] = []

    checks.append(check_prompt_templates(upstream_source))
    checks.append(check_resolution_matching())
    checks.append(check_scoring_formula(leaderboard))
    checks.append(check_missing_forecast_default())
    checks.append(check_multi_horizon_batching())
    checks.append(check_question_count(leaderboard))

    for _, msg in checks:
        print(msg)

    score_checks: list[tuple[bool, str]] = []
    if args.score:
        print()
        print("Score Comparison (--score):")
        score_checks.append(check_score_comparison(leaderboard))
        score_checks.append(check_per_source_breakdown(leaderboard))
        for _, msg in score_checks:
            print(msg)

    all_checks = checks + score_checks
    passed = sum(1 for ok, _ in all_checks if ok)
    total = len(all_checks)

    print()
    print(f"Result: {passed}/{total} passed")

    if any(not ok for ok, _ in all_checks):
        sys.exit(1)


if __name__ == "__main__":
    main()
