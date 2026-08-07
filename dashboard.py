"""Interactive experiment results dashboard for ForecastBench.

Launch: uv run --extra dashboard streamlit run dashboard.py
"""

from __future__ import annotations

import json
import math
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, cast

try:
    import pandas as pd
    import streamlit as st
    import plotly.graph_objects as go
except ImportError:
    print(
        "Dashboard requires streamlit and plotly.\n"
        "Install with: uv pip install 'forecastbench[dashboard]'\n"
        "Or run: uv run --extra dashboard streamlit run dashboard.py"
    )
    sys.exit(1)

from analyze import (
    _lookup_forecast,
    analyze_by_source,
    analyze_calibration,
    calibration_metrics,
)
from fetch_data import MARKET_SOURCES, ResolvedQuestion, fetch_leaderboard, load_data
from score import brier_index, brier_score


ResultData = dict[str, Any]

RESULTS_DIR = Path("results")

LEADERBOARD_REFERENCE: dict[str, dict[str, float]] = {
    "human_superforecaster": {"overall_index": 68.2, "overall_brier": 0.101},
    "sonnet_4_official": {
        "overall_index": 60.0,
        "dataset_index": 59.0,
        "market_index": 61.0,
        "overall_brier": 0.160,
    },
}


@dataclass
class AggregateRun:
    slug: str
    n_rounds: int
    rounds: list[str]
    combined_forecasts: dict[str, float]
    combined_outcomes: dict[str, int]
    scoring_result: dict[str, Any]
    per_round_results: list[ResultData]
    combined_sources: dict[str, str] = field(default_factory=dict)

    @property
    def label(self) -> str:
        return self.slug


@st.cache_data  # type: ignore[untyped-decorator]
def load_all_results() -> list[ResultData]:
    if not RESULTS_DIR.exists():
        return []
    results: list[ResultData] = []
    for f in sorted(RESULTS_DIR.glob("*.json")):
        try:
            data = json.loads(f.read_text())
            if "model_slug" in data and "forecasts" in data:
                data["_filename"] = f.name
                results.append(data)
        except (json.JSONDecodeError, KeyError):
            continue
    return results


@st.cache_data  # type: ignore[untyped-decorator]
def load_resolved_questions() -> list[dict[str, Any]]:
    _, resolved = load_data()
    return [
        {
            "id": q.id,
            "source": q.source,
            "question": q.question,
            "outcome": q.outcome,
        }
        for q in resolved
    ]


@st.cache_data  # type: ignore[untyped-decorator]
def load_leaderboard(name: str) -> list[dict[str, str]]:
    return fetch_leaderboard(name)


def _round_name_from_result(result: ResultData) -> str:
    meta = result.get("metadata", {})
    rnd = meta.get("round")
    if rnd:
        return str(rnd)
    qsets = meta.get("question_sets_used", [])
    if qsets:
        return ", ".join(sorted(qsets))
    return result.get("_filename", "unknown")


def _compute_aggregate_scoring(
    forecasts: dict[str, float],
    outcomes: dict[str, int],
    sources: dict[str, str],
) -> dict[str, Any]:
    """Compute scoring result from combined forecasts and outcomes."""
    shared_ids = set(forecasts.keys()) & set(outcomes.keys())
    if not shared_ids:
        return {
            "overall_brier": 1.0, "overall_index": 0.0,
            "dataset_brier": 1.0, "dataset_index": 0.0,
            "market_brier": 1.0, "market_index": 0.0,
            "n_dataset": 0, "n_market": 0, "n_missing": 0,
        }

    dataset_pairs: list[tuple[float, int]] = []
    market_pairs: list[tuple[float, int]] = []
    n_missing = 0

    for qid in shared_ids:
        f = forecasts.get(qid, 0.5)
        o = outcomes[qid]
        source = sources.get(qid, "")
        if source in MARKET_SOURCES:
            market_pairs.append((f, o))
        else:
            dataset_pairs.append((f, o))
        if qid not in forecasts:
            n_missing += 1

    def _mean_brier(pairs: list[tuple[float, int]]) -> float:
        if not pairs:
            return 0.25
        return sum(brier_score(f, o) for f, o in pairs) / len(pairs)

    ds_brier = _mean_brier(dataset_pairs)
    mk_brier = _mean_brier(market_pairs)
    overall_brier = (ds_brier + mk_brier) / 2

    return {
        "overall_brier": overall_brier,
        "overall_index": brier_index(overall_brier),
        "dataset_brier": ds_brier,
        "dataset_index": brier_index(ds_brier),
        "market_brier": mk_brier,
        "market_index": brier_index(mk_brier),
        "n_dataset": len(dataset_pairs),
        "n_market": len(market_pairs),
        "n_missing": n_missing,
    }


def _group_results_into_runs(results: list[ResultData]) -> list[AggregateRun]:
    """Group result files by model_slug into aggregate runs."""
    grouped: dict[str, list[ResultData]] = {}
    for r in results:
        slug = str(r["model_slug"])
        grouped.setdefault(slug, []).append(r)

    runs: list[AggregateRun] = []
    for slug, result_files in sorted(grouped.items()):
        rounds: list[str] = []
        combined_forecasts: dict[str, float] = {}
        combined_outcomes: dict[str, int] = {}
        combined_sources: dict[str, str] = {}

        for rf in sorted(result_files, key=lambda r: _round_name_from_result(r)):
            rnd = _round_name_from_result(rf)
            rounds.append(rnd)
            combined_forecasts.update(rf.get("forecasts", {}))
            combined_outcomes.update(rf.get("outcomes", {}))
            combined_sources.update(rf.get("sources", {}))

        scoring = _compute_aggregate_scoring(
            combined_forecasts, combined_outcomes, combined_sources,
        )

        runs.append(AggregateRun(
            slug=slug,
            n_rounds=len(result_files),
            rounds=rounds,
            combined_forecasts=combined_forecasts,
            combined_outcomes=combined_outcomes,
            scoring_result=scoring,
            per_round_results=result_files,
            combined_sources=combined_sources,
        ))

    return runs


@st.cache_data  # type: ignore[untyped-decorator]
def group_results(results: list[ResultData]) -> list[dict[str, Any]]:
    """Cached wrapper that returns serializable dicts (Streamlit requirement)."""
    runs = _group_results_into_runs(results)
    return [
        {
            "slug": run.slug,
            "n_rounds": run.n_rounds,
            "rounds": run.rounds,
            "combined_forecasts": run.combined_forecasts,
            "combined_outcomes": run.combined_outcomes,
            "scoring_result": run.scoring_result,
            "per_round_results": run.per_round_results,
            "combined_sources": run.combined_sources,
        }
        for run in runs
    ]


def _dict_to_aggregate(d: dict[str, Any]) -> AggregateRun:
    return AggregateRun(
        slug=d["slug"],
        n_rounds=d["n_rounds"],
        rounds=d["rounds"],
        combined_forecasts=d["combined_forecasts"],
        combined_outcomes=d["combined_outcomes"],
        scoring_result=d["scoring_result"],
        per_round_results=d["per_round_results"],
        combined_sources=d["combined_sources"],
    )


def _leaderboard_reference_from_live() -> dict[str, dict[str, float]] | None:
    """Pull top reference entries from the live baseline leaderboard."""
    try:
        rows = load_leaderboard("baseline")
    except Exception:
        return None
    ref: dict[str, dict[str, float]] = {}
    for row in rows:
        model = row.get("Model", "")
        if "superforecaster" in model.lower():
            ref["human_superforecaster"] = {
                "overall_index": float(row.get("Overall", 0)),
                "overall_brier": float(row.get("Brier Overall", 0)),
            }
        if "claude-sonnet-4-20250514" == model:
            ref["sonnet_4_official"] = {
                "overall_index": float(row.get("Overall", 0)),
                "dataset_index": float(row.get("Dataset", 0)),
                "market_index": float(row.get("Market", 0)),
                "overall_brier": float(row.get("Brier Overall", 0)),
            }
    return ref if ref else None


def _model_matches_slug(leaderboard_model: str, model_slug: str) -> bool:
    """Check if a leaderboard model name approximately matches our model slug."""
    lb = leaderboard_model.lower().replace("-", "_").replace(" ", "_")
    slug = model_slug.lower().replace("-", "_").replace(" ", "_")
    lb_parts = lb.split("_")
    slug_parts = slug.split("_")
    provider_prefixes = {"vertex_ai", "openai", "anthropic", "google", "litellm"}
    clean_slug_parts = []
    skipping = True
    for part in slug_parts:
        if skipping and part in provider_prefixes:
            continue
        skipping = False
        clean_slug_parts.append(part)
    if not clean_slug_parts:
        clean_slug_parts = slug_parts
    clean_slug = "_".join(clean_slug_parts)
    clean_lb = "_".join(lb_parts)
    return clean_slug in clean_lb or clean_lb in clean_slug


def _resolved_to_objects(resolved_dicts: list[dict[str, Any]]) -> list[ResolvedQuestion]:
    return [
        ResolvedQuestion(
            id=d["id"],
            source=d["source"],
            question=d["question"],
            outcome=d["outcome"],
        )
        for d in resolved_dicts
    ]


def _classify_error(forecast: float, outcome: int) -> str:
    if forecast >= 0.7 and outcome == 1:
        return "Correct confident"
    if forecast <= 0.3 and outcome == 0:
        return "Correct confident"
    if forecast >= 0.7 and outcome == 0:
        return "Confident wrong (high)"
    if forecast <= 0.3 and outcome == 1:
        return "Confident wrong (low)"
    if 0.3 < forecast < 0.7:
        correct = (forecast >= 0.5 and outcome == 1) or (forecast < 0.5 and outcome == 0)
        if correct:
            return "Correct uncertain"
        return "Uncertain"
    return "Correct uncertain"


def _build_source_brier_matrix(
    runs: list[AggregateRun],
    resolved_dicts: list[dict[str, Any]],
) -> tuple[list[str], list[str], list[list[float | None]], list[list[int]]]:
    resolved = _resolved_to_objects(resolved_dicts)
    all_sources: set[str] = set()
    run_source_scores: dict[str, dict[str, float]] = {}
    run_source_counts: dict[str, dict[str, int]] = {}

    for run in runs:
        label = run.label
        by_source = analyze_by_source(run.combined_forecasts, resolved)
        run_source_scores[label] = {}
        run_source_counts[label] = {}
        for source, stats in by_source.items():
            brier_val: Any = stats["brier"]
            count_val: Any = stats["count"]
            run_source_scores[label][source] = float(brier_val)
            run_source_counts[label][source] = int(count_val)
            all_sources.add(source)

    sources = sorted(all_sources)
    overall_brier: dict[str, float] = {}
    for run in runs:
        overall_brier[run.label] = run.scoring_result.get("overall_brier", 1.0)
    run_labels = sorted(run_source_scores.keys(), key=lambda r: overall_brier.get(r, 1.0))

    matrix: list[list[float | None]] = []
    counts: list[list[int]] = []
    for run_label in run_labels:
        row: list[float | None] = []
        count_row: list[int] = []
        for source in sources:
            row.append(run_source_scores[run_label].get(source))
            count_row.append(run_source_counts[run_label].get(source, 0))
        matrix.append(row)
        counts.append(count_row)

    return run_labels, sources, matrix, counts


def _sort_sources_by_track(sources: list[str]) -> list[str]:
    dataset_sources = [s for s in sources if s not in MARKET_SOURCES]
    market_sources = [s for s in sources if s in MARKET_SOURCES]
    return sorted(dataset_sources) + sorted(market_sources)


def view_overview(runs: list[AggregateRun], resolved_dicts: list[dict[str, Any]]) -> None:
    st.header("Overview")

    if not runs:
        st.warning("No result files found in results/ directory.")
        return

    best_run = min(runs, key=lambda r: r.scoring_result.get("overall_brier", 1.0))
    best_sr = best_run.scoring_result
    best_index = best_sr.get("overall_index", 0.0)

    total_questions = best_sr.get("n_dataset", 0) + best_sr.get("n_market", 0)

    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Best Run", f"{best_run.label}", f"Index: {best_index:.1f}")
    m2.metric("Total Runs", len(runs))
    m3.metric("Total Questions", f"{total_questions:,}")
    m4.metric("Dataset Index", f"{best_sr.get('dataset_index', 0):.1f}")
    m5.metric("Market Index", f"{best_sr.get('market_index', 0):.1f}")

    with st.expander("How to read these numbers"):
        st.markdown(
            "- **Brier Index**: 0 = no skill (always guessing 0.5), 100 = perfect. "
            "Higher is better. Human superforecasters score 60–70. "
            "Compare against the leaderboard reference below.\n"
            "- **Dataset vs Market**: Dataset questions are time-series numeric predictions. "
            "Market questions come from prediction platforms. "
            "Performance often differs significantly between tracks.\n"
            "- **Overall Brier**: Raw error metric — 0 = perfect, 0.25 = no skill, "
            "1 = worst. Lower is better. The Brier Index is derived from this."
        )

    st.subheader("All Runs")
    rows: list[dict[str, Any]] = []
    for run in runs:
        sr = run.scoring_result
        n_total = sr.get("n_dataset", 0) + sr.get("n_market", 0)
        rows.append({
            "Run": run.label,
            "Rounds": run.n_rounds,
            "Overall Index": round(sr.get("overall_index", 0), 1),
            "Overall Brier": round(sr.get("overall_brier", 0), 4),
            "Dataset Index": round(sr.get("dataset_index", 0), 1),
            "Dataset Brier": round(sr.get("dataset_brier", 0), 4),
            "Market Index": round(sr.get("market_index", 0), 1),
            "Market Brier": round(sr.get("market_brier", 0), 4),
            "N Questions": n_total,
        })
    df = pd.DataFrame(rows).sort_values("Overall Index", ascending=False)
    st.dataframe(df, hide_index=True, use_container_width=True)

    # Round drill-down
    st.subheader("Round Drill-Down")
    run_labels = [r.label for r in runs]
    selected_drill = st.selectbox(
        "Select run to view rounds", run_labels, index=0, key="drill_run"
    )
    drill_run = next(r for r in runs if r.label == selected_drill)

    if drill_run.n_rounds <= 1:
        st.info("This run has only 1 round — no drill-down available.")
    else:
        round_rows: list[dict[str, Any]] = []
        for rf in sorted(
            drill_run.per_round_results,
            key=lambda r: _round_name_from_result(r),
        ):
            rsr = rf.get("scoring_result", {})
            rn = rsr.get("n_dataset", 0) + rsr.get("n_market", 0)
            round_rows.append({
                "Round": _round_name_from_result(rf),
                "Overall Index": round(rsr.get("overall_index", 0), 1),
                "Overall Brier": round(rsr.get("overall_brier", 0), 4),
                "Dataset Index": round(rsr.get("dataset_index", 0), 1),
                "Market Index": round(rsr.get("market_index", 0), 1),
                "N Questions": rn,
            })
        round_df = pd.DataFrame(round_rows)
        st.dataframe(round_df, hide_index=True, use_container_width=True)

    all_qsets: set[str] = set()
    for run in runs:
        for rf in run.per_round_results:
            qsets = rf.get("metadata", {}).get("question_sets_used", [])
            all_qsets.update(qsets)
    if all_qsets:
        st.subheader("Rounds Covered")
        st.markdown(", ".join(sorted(all_qsets)))

    st.subheader("Leaderboard Reference")
    live_ref = _leaderboard_reference_from_live()
    ref_data = live_ref if live_ref else LEADERBOARD_REFERENCE
    if live_ref:
        st.caption("Live reference points from the official ForecastBench baseline leaderboard.")
    else:
        st.caption("Approximate reference points (live leaderboard unavailable, using cached values).")
    ref_rows: list[dict[str, Any]] = []
    for name, ref in ref_data.items():
        ref_rows.append({
            "Reference": name.replace("_", " ").title(),
            "Overall Index": ref.get("overall_index", 0),
            "Overall Brier": ref.get("overall_brier", 0),
            "Dataset Index": ref.get("dataset_index"),
            "Market Index": ref.get("market_index"),
        })
    ref_df = pd.DataFrame(ref_rows)
    st.dataframe(ref_df, hide_index=True, use_container_width=True)


def view_leaderboard(runs: list[AggregateRun]) -> None:
    st.header("Official ForecastBench Leaderboard")

    lb_name = st.radio(
        "Leaderboard",
        ["baseline", "tournament", "dataset", "preliminary"],
        horizontal=True,
        key="lb_select",
    )

    try:
        rows = load_leaderboard(lb_name)
    except Exception as e:
        st.error(f"Failed to fetch leaderboard: {e}")
        return

    if not rows:
        st.warning("No leaderboard data returned.")
        return

    column_map = {
        "Rank": "Rank",
        "Model": "Model",
        "Model Organization": "Organization",
        "Overall": "Overall Index",
        "Dataset": "Dataset Index",
        "Market": "Market Index",
        "Brier Overall": "Overall Brier",
        "N": "N",
        "Supers > Forecaster?": "vs Supers",
    }
    display_rows: list[dict[str, Any]] = []
    for row in rows:
        display: dict[str, Any] = {}
        for src_col, dst_col in column_map.items():
            val = row.get(src_col, "")
            if src_col in ("Rank", "N"):
                try:
                    display[dst_col] = int(val)
                except (ValueError, TypeError):
                    display[dst_col] = val
            elif src_col in ("Overall", "Dataset", "Market", "Brier Overall"):
                try:
                    display[dst_col] = float(val)
                except (ValueError, TypeError):
                    display[dst_col] = val
            else:
                display[dst_col] = val
        display_rows.append(display)

    our_slugs = [r.slug for r in runs]
    matched_rows: list[dict[str, Any]] = []
    for display in display_rows:
        lb_model = str(display.get("Model", ""))
        for slug in our_slugs:
            if _model_matches_slug(lb_model, slug):
                matched_row = dict(display)
                matched_row["Our Run"] = slug
                matched_rows.append(matched_row)
                break

    if matched_rows:
        st.subheader("Our Models on the Leaderboard")
        matched_df = pd.DataFrame(matched_rows)
        st.dataframe(matched_df, hide_index=True, use_container_width=True)

    st.subheader(f"Full {lb_name.title()} Leaderboard")
    full_df = pd.DataFrame(display_rows)
    st.dataframe(full_df, hide_index=True, use_container_width=True, height=600)

    st.caption(
        "Live data from the official ForecastBench leaderboard "
        "(github.com/forecastingresearch/forecastbench-datasets). "
        "Scores are Brier Index (higher = better)."
    )

    with st.expander("Interpretation"):
        st.markdown(
            "The **baseline** leaderboard ranks all models using zero-shot prompts. "
            "The **tournament** leaderboard includes models with custom scaffolding. "
            "Rank 1 is the superforecaster median (human benchmark). "
            "Models significantly worse than random (always predicting 0.5) are at the bottom."
        )


def view_heatmap(runs: list[AggregateRun], resolved_dicts: list[dict[str, Any]]) -> None:
    st.header("Run × Source Heatmap")

    if not runs:
        st.warning("No result files found in results/ directory.")
        return

    col_metric, col_group = st.columns(2)
    with col_metric:
        metric_mode = st.radio(
            "Display metric",
            ["Brier Score", "Brier Index"],
            horizontal=True,
            key="heatmap_metric",
        )
    with col_group:
        group_mode = st.radio(
            "Group sources",
            ["By Source", "By Track"],
            horizontal=True,
            key="heatmap_group",
        )

    show_index = metric_mode == "Brier Index"

    if group_mode == "By Track":
        _view_heatmap_by_track(runs, resolved_dicts, show_index)
    else:
        _view_heatmap_by_source(runs, resolved_dicts, show_index)


def _view_heatmap_by_source(
    runs: list[AggregateRun],
    resolved_dicts: list[dict[str, Any]],
    show_index: bool,
) -> None:
    run_labels, sources, matrix, counts = _build_source_brier_matrix(runs, resolved_dicts)

    sources_sorted = _sort_sources_by_track(sources)
    source_idx_map = {s: i for i, s in enumerate(sources)}
    col_order = [source_idx_map[s] for s in sources_sorted]

    display_matrix: list[list[float | None]] = []
    for row in matrix:
        reordered = [row[j] for j in col_order]
        if show_index:
            reordered = [brier_index(v) if v is not None else None for v in reordered]
        display_matrix.append(reordered)

    display_counts: list[list[int]] = []
    for row in counts:
        display_counts.append([row[j] for j in col_order])

    display_sources = sources_sorted
    label_sources = []
    for s in display_sources:
        track = "market" if s in MARKET_SOURCES else "dataset"
        label_sources.append(f"{s} ({track})")

    metric_label = "Index" if show_index else "Brier"
    caption = (
        "Brier Index by run and source. Higher = better."
        if show_index
        else "Brier score by run and source. Darker = better (lower score)."
    )
    st.caption(caption)

    hover_text: list[list[str]] = []
    for i, run in enumerate(run_labels):
        row: list[str] = []
        for j, source in enumerate(label_sources):
            val = display_matrix[i][j]
            n = display_counts[i][j]
            if val is not None:
                row.append(
                    f"Run: {run}<br>Source: {source}<br>{metric_label}: {val:.3f}<br>N: {n}"
                )
            else:
                row.append(f"Run: {run}<br>Source: {source}<br>No data")
        hover_text.append(row)

    annotations: list[dict[str, Any]] = []
    for i, run in enumerate(run_labels):
        for j in range(len(label_sources)):
            val = display_matrix[i][j]
            if val is not None:
                fmt = f"{val:.1f}" if show_index else f"{val:.3f}"
                threshold = 50 if show_index else 0.15
                annotations.append(
                    dict(
                        x=j,
                        y=i,
                        text=fmt,
                        showarrow=False,
                        font=dict(
                            size=10,
                            color="white" if (show_index and val < threshold) or (not show_index and val > threshold) else "black",
                        ),
                    )
                )

    colorscale = "Viridis" if show_index else "Viridis_r"

    fig = go.Figure(
        data=go.Heatmap(
            z=display_matrix,
            x=label_sources,
            y=run_labels,
            colorscale=colorscale,
            hovertext=hover_text,
            hoverinfo="text",
            colorbar=dict(title=metric_label),
        )
    )
    fig.update_layout(
        annotations=annotations,
        xaxis_title="Source",
        yaxis_title="Run",
        height=max(300, 60 * len(run_labels) + 100),
    )
    st.plotly_chart(fig, use_container_width=True)

    with st.expander("Understanding the scale"):
        if show_index:
            st.markdown(
                "100 = perfect, 0 = no skill. "
                "Typical AI model range: 45–65. Human superforecasters: 60–70."
            )
        else:
            st.markdown(
                "0.00 = perfect, 0.25 = no skill (predicting 0.5 every time), "
                "0.50+ = worse than guessing. Most models score between 0.10 and 0.30."
            )

    st.subheader("Marginal Averages")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**By Run**")
        for run in sorted(runs, key=lambda r: r.scoring_result.get("overall_brier", 1.0)):
            sr = run.scoring_result
            bs = sr.get("overall_brier", 0)
            idx = sr.get("overall_index", 0)
            st.text(f"{run.label}: Brier {bs:.4f} | Index {idx:.1f}")
    with col2:
        st.markdown("**By Source (avg across runs)**")
        source_avgs: dict[str, list[float]] = {}
        for i, _run in enumerate(run_labels):
            for j, source in enumerate(label_sources):
                orig_j = col_order[j]
                val = matrix[i][orig_j]
                if val is not None:
                    source_avgs.setdefault(source, []).append(val)
        for source in label_sources:
            vals = source_avgs.get(source, [])
            if vals:
                avg_bs = sum(vals) / len(vals)
                avg_idx = brier_index(avg_bs)
                st.text(f"{source}: Brier {avg_bs:.4f} | Index {avg_idx:.1f} (n={len(vals)} runs)")


def _view_heatmap_by_track(
    runs: list[AggregateRun],
    resolved_dicts: list[dict[str, Any]],
    show_index: bool,
) -> None:
    overall_brier_map: dict[str, float] = {}
    for run in runs:
        overall_brier_map[run.label] = run.scoring_result.get("overall_brier", 1.0)
    run_labels = sorted(overall_brier_map.keys(), key=lambda r: overall_brier_map.get(r, 1.0))

    tracks = ["dataset", "market"]
    track_matrix: list[list[float | None]] = []
    track_counts: list[list[int]] = []

    for run_label in run_labels:
        run = next(r for r in runs if r.label == run_label)
        sr = run.scoring_result
        ds_bs = sr.get("dataset_brier")
        mk_bs = sr.get("market_brier")
        ds_n = sr.get("n_dataset", 0)
        mk_n = sr.get("n_market", 0)

        row: list[float | None] = []
        count_row: list[int] = []
        for track in tracks:
            if track == "dataset":
                val = ds_bs
                n = ds_n
            else:
                val = mk_bs
                n = mk_n
            if show_index and val is not None:
                val = brier_index(val)
            row.append(val)
            count_row.append(n)
        track_matrix.append(row)
        track_counts.append(count_row)

    metric_label = "Index" if show_index else "Brier"
    colorscale = "Viridis" if show_index else "Viridis_r"

    hover_text: list[list[str]] = []
    annotations: list[dict[str, Any]] = []
    for i, run in enumerate(run_labels):
        row_hover: list[str] = []
        for j, track in enumerate(tracks):
            val = track_matrix[i][j]
            n = track_counts[i][j]
            if val is not None:
                row_hover.append(f"Run: {run}<br>Track: {track}<br>{metric_label}: {val:.3f}<br>N: {n}")
                fmt = f"{val:.1f}" if show_index else f"{val:.3f}"
                threshold = 50 if show_index else 0.15
                annotations.append(
                    dict(
                        x=j,
                        y=i,
                        text=fmt,
                        showarrow=False,
                        font=dict(
                            size=12,
                            color="white" if (show_index and val < threshold) or (not show_index and val > threshold) else "black",
                        ),
                    )
                )
            else:
                row_hover.append(f"Run: {run}<br>Track: {track}<br>No data")
        hover_text.append(row_hover)

    fig = go.Figure(
        data=go.Heatmap(
            z=track_matrix,
            x=tracks,
            y=run_labels,
            colorscale=colorscale,
            hovertext=hover_text,
            hoverinfo="text",
            colorbar=dict(title=metric_label),
        )
    )
    fig.update_layout(
        annotations=annotations,
        xaxis_title="Track",
        yaxis_title="Run",
        height=max(300, 60 * len(run_labels) + 100),
    )
    st.plotly_chart(fig, use_container_width=True)



def view_failures(runs: list[AggregateRun], resolved_dicts: list[dict[str, Any]]) -> None:
    st.header("Failure Explorer")

    if not runs:
        st.warning("No result files found.")
        return

    run_names = [r.label for r in runs]
    selected_run = st.selectbox("Select run", run_names, index=0, key="failures_run")
    run = next(r for r in runs if r.label == selected_run)
    forecasts: dict[str, float] = run.combined_forecasts

    question_data: list[dict[str, Any]] = []
    for d in resolved_dicts:
        qid = d["id"]
        f = _lookup_forecast(forecasts, qid)
        outcome = d["outcome"]
        bs = brier_score(f, outcome)
        error_type = _classify_error(f, outcome)
        question_data.append({
            "Question": d["question"][:120],
            "Source": d["source"],
            "Forecast": round(f, 3),
            "Outcome": outcome,
            "Brier Score": round(bs, 4),
            "Error Type": error_type,
            "_brier_raw": bs,
        })

    source_stats: dict[str, dict[str, float | int]] = {}
    for q in question_data:
        src = q["Source"]
        if src not in source_stats:
            source_stats[src] = {"brier_sum": 0.0, "count": 0, "cw_count": 0}
        source_stats[src]["brier_sum"] += q["_brier_raw"]
        source_stats[src]["count"] += 1
        if q["Error Type"] in ("Confident wrong (high)", "Confident wrong (low)"):
            source_stats[src]["cw_count"] += 1

    if source_stats:
        worst_cw_source = max(
            source_stats,
            key=lambda s: source_stats[s]["cw_count"] / source_stats[s]["count"]
            if source_stats[s]["count"]
            else 0,
        )
        worst_cw_pct = (
            source_stats[worst_cw_source]["cw_count"]
            / source_stats[worst_cw_source]["count"]
            * 100
            if source_stats[worst_cw_source]["count"]
            else 0
        )
        worst_brier_source = max(
            source_stats,
            key=lambda s: source_stats[s]["brier_sum"] / source_stats[s]["count"]
            if source_stats[s]["count"]
            else 0,
        )
        worst_brier = (
            source_stats[worst_brier_source]["brier_sum"]
            / source_stats[worst_brier_source]["count"]
            if source_stats[worst_brier_source]["count"]
            else 0
        )
        total_cw = sum(int(s["cw_count"]) for s in source_stats.values())
        total_q = len(question_data)
        total_cw_pct = (total_cw / total_q * 100) if total_q else 0

        summary = (
            f"Highest failure rate on **{worst_cw_source}** "
            f"({worst_cw_pct:.1f}% confident wrong). "
            f"**{worst_brier_source}** has the worst Brier score "
            f"({worst_brier:.4f}). "
            f"Overall {total_cw_pct:.1f}% of questions are confidently wrong. "
            f"Focus calibration improvements here."
        )
        st.info(summary)

    st.subheader("Error Type Summary")
    with st.expander("Error type definitions"):
        st.markdown(
            "- **Confident wrong (high)**: Model predicted ≥ 70% likely, "
            "but the event did NOT happen (outcome=0). "
            "The model was confidently wrong in the 'yes' direction.\n"
            "- **Confident wrong (low)**: Model predicted ≤ 30% likely, "
            "but the event DID happen (outcome=1). "
            "The model was confidently wrong in the 'no' direction.\n"
            "- **Uncertain**: Model hedged (30–70%) and got the direction wrong. "
            "Less costly per question but indicates lack of signal.\n"
            "- **Correct confident**: Model was confident (>70% or <30%) and got it right.\n"
            "- **Correct uncertain**: Model hedged (30–70%) and happened to get the "
            "direction right."
        )

    error_counts: dict[str, int] = {}
    for q in question_data:
        et = q["Error Type"]
        error_counts[et] = error_counts.get(et, 0) + 1

    total = len(question_data)
    summary_cols = st.columns(min(len(error_counts), 5))
    for i, (et, count) in enumerate(sorted(error_counts.items(), key=lambda x: -x[1])):
        col = summary_cols[i % len(summary_cols)]
        pct = (count / total * 100) if total > 0 else 0
        col.metric(et, f"{count}", f"{pct:.1f}%")

    st.subheader("Error Type by Source")
    source_error: dict[str, dict[str, int]] = {}
    for q in question_data:
        src = q["Source"]
        et = q["Error Type"]
        source_error.setdefault(src, {})
        source_error[src][et] = source_error[src].get(et, 0) + 1

    all_error_types = sorted(error_counts.keys())
    error_colors = {
        "Confident wrong (high)": "#d62728",
        "Confident wrong (low)": "#ff7f0e",
        "Uncertain": "#bcbd22",
        "Correct confident": "#2ca02c",
        "Correct uncertain": "#17becf",
    }

    fig = go.Figure()
    source_list = sorted(source_error.keys())
    for et in all_error_types:
        values = [source_error[src].get(et, 0) for src in source_list]
        fig.add_trace(go.Bar(
            name=et,
            x=source_list,
            y=values,
            marker_color=error_colors.get(et, "#999999"),
        ))
    fig.update_layout(
        barmode="stack",
        xaxis_title="Source",
        yaxis_title="Count",
        title="Error Type Breakdown by Source",
        height=400,
    )
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Source-Level Failure Summary")
    source_summary_rows: list[dict[str, Any]] = []
    for src in source_list:
        src_questions = [q for q in question_data if q["Source"] == src]
        n_src = len(src_questions)
        mean_bs = sum(q["_brier_raw"] for q in src_questions) / n_src if n_src else 0
        confident_wrong = sum(
            1 for q in src_questions if q["Error Type"] in ("Confident wrong (high)", "Confident wrong (low)")
        )
        pct_cw = (confident_wrong / n_src * 100) if n_src else 0
        worst_q = max(src_questions, key=lambda q: q["_brier_raw"]) if src_questions else None
        track = "market" if src in MARKET_SOURCES else "dataset"
        source_summary_rows.append({
            "Source": src,
            "Track": track,
            "N Questions": n_src,
            "Mean Brier": round(mean_bs, 4),
            "Index": round(brier_index(mean_bs), 1),
            "% Confident Wrong": round(pct_cw, 1),
            "Worst Question": worst_q["Question"] if worst_q else "",
        })
    summary_df = pd.DataFrame(source_summary_rows).sort_values("Mean Brier", ascending=False)
    st.dataframe(summary_df, hide_index=True, use_container_width=True)
    st.caption(
        'A high "% Confident Wrong" rate on a source means the model is '
        "systematically overconfident on that question type — "
        "this is the most actionable signal for improvement."
    )

    st.subheader("Worst Questions")
    top_n = st.slider("Show top N worst", 10, 200, 50, key="worst_n")
    question_data.sort(key=lambda q: -q["_brier_raw"])
    worst_rows = [
        {k: v for k, v in q.items() if k != "_brier_raw"}
        for q in question_data[:top_n]
    ]
    worst_df = pd.DataFrame(worst_rows)
    st.dataframe(worst_df, hide_index=True, use_container_width=True, height=600)

    st.divider()
    st.subheader("Compare Failures Across Runs")

    other_runs = [r for r in run_names if r != selected_run]
    compare_run_name = st.selectbox(
        "Compare against run",
        ["(none)"] + other_runs,
        index=0,
        key="failures_compare",
    )

    if compare_run_name != "(none)":
        run_b = next(r for r in runs if r.label == compare_run_name)
        forecasts_b: dict[str, float] = run_b.combined_forecasts

        source_compare_rows: list[dict[str, Any]] = []
        for src in source_list:
            src_questions = [q for q in question_data if q["Source"] == src]
            n_src = len(src_questions)
            if not n_src:
                continue

            mean_a = sum(q["_brier_raw"] for q in src_questions) / n_src
            cw_a = sum(
                1 for q in src_questions
                if q["Error Type"] in ("Confident wrong (high)", "Confident wrong (low)")
            )
            pct_cw_a = cw_a / n_src * 100

            brier_b_vals: list[float] = []
            cw_b = 0
            for q_dict in resolved_dicts:
                if q_dict["source"] != src:
                    continue
                fb = _lookup_forecast(forecasts_b, q_dict["id"])
                bs_b = brier_score(fb, q_dict["outcome"])
                brier_b_vals.append(bs_b)
                err_b = _classify_error(fb, q_dict["outcome"])
                if err_b in ("Confident wrong (high)", "Confident wrong (low)"):
                    cw_b += 1

            if not brier_b_vals:
                continue

            n_b = len(brier_b_vals)
            mean_b = sum(brier_b_vals) / n_b
            pct_cw_b = cw_b / n_b * 100

            track = "market" if src in MARKET_SOURCES else "dataset"
            source_compare_rows.append({
                "Source": src,
                "Track": track,
                f"{selected_run} Brier": round(mean_a, 4),
                f"{compare_run_name} Brier": round(mean_b, 4),
                "Delta Brier": round(mean_b - mean_a, 4),
                f"{selected_run} % CW": round(pct_cw_a, 1),
                f"{compare_run_name} % CW": round(pct_cw_b, 1),
                "Delta % CW": round(pct_cw_b - pct_cw_a, 1),
            })

        if source_compare_rows:
            compare_df = pd.DataFrame(source_compare_rows)
            st.dataframe(compare_df, hide_index=True, use_container_width=True)

            n_sources = len(source_compare_rows)
            improved = sum(1 for r in source_compare_rows if r["Delta Brier"] < 0)
            worsened = sum(1 for r in source_compare_rows if r["Delta Brier"] > 0)
            best_row = min(source_compare_rows, key=lambda r: r["Delta Brier"])
            worst_row = max(source_compare_rows, key=lambda r: r["Delta Brier"])
            summary_parts = [
                f"**{compare_run_name}** improved on **{improved}/{n_sources}** sources, "
                f"worsened on **{worsened}/{n_sources}**.",
            ]
            if best_row["Delta Brier"] < 0:
                summary_parts.append(
                    f"Biggest improvement: **{best_row['Source']}** "
                    f"({best_row['Delta Brier']:+.4f} Brier)."
                )
            if worst_row["Delta Brier"] > 0:
                summary_parts.append(
                    f"Biggest regression: **{worst_row['Source']}** "
                    f"({worst_row['Delta Brier']:+.4f} Brier)."
                )
            st.info(" ".join(summary_parts))
            st.caption(
                "Delta columns show (Compare run - Selected run). "
                "Negative delta = the comparison run is better on that source."
            )


def view_calibration(runs: list[AggregateRun], resolved_dicts: list[dict[str, Any]]) -> None:
    st.header("Calibration Curves")

    if not runs:
        st.warning("No result files found.")
        return

    run_names = [r.label for r in runs]
    selected_runs = st.multiselect("Select runs to compare", run_names, default=run_names[:3])

    if not selected_runs:
        st.info("Select at least one run.")
        return

    resolved = _resolved_to_objects(resolved_dicts)
    n_bins = st.slider("Number of bins", 5, 20, 10)

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=[0, 1],
            y=[0, 1],
            mode="lines",
            line=dict(dash="dash", color="gray"),
            name="Perfect calibration",
            showlegend=True,
        )
    )

    metrics_rows: list[dict[str, Any]] = []

    for run_name in selected_runs:
        run = next(r for r in runs if r.label == run_name)
        cal_bins = analyze_calibration(run.combined_forecasts, resolved, n_bins=n_bins)
        if not cal_bins:
            continue

        x_vals = [float(cast(Any, b["mean_predicted"])) for b in cal_bins]
        y_vals = [float(cast(Any, b["mean_observed"])) for b in cal_bins]
        sizes = [max(5, min(30, int(cast(Any, b["count"])) // 50 + 5)) for b in cal_bins]
        hover = [
            f"Predicted: {x:.3f}<br>Observed: {y:.3f}<br>N: {int(cast(Any, b['count']))}"
            for x, y, b in zip(x_vals, y_vals, cal_bins)
        ]

        fig.add_trace(
            go.Scatter(
                x=x_vals,
                y=y_vals,
                mode="lines+markers",
                marker=dict(size=sizes),
                name=run_name,
                hovertext=hover,
                hoverinfo="text",
            )
        )

        pairs = [(_lookup_forecast(run.combined_forecasts, q.id), q.outcome) for q in resolved]
        cal = calibration_metrics(pairs, n_bins=n_bins)
        metrics_rows.append(
            {"Run": run_name, "ECE": cal["ece"], "MCE": cal["mce"], "Sharpness": cal["sharpness"]}
        )

    fig.update_layout(
        xaxis_title="Mean Predicted Probability",
        yaxis_title="Mean Observed Frequency",
        xaxis=dict(range=[0, 1]),
        yaxis=dict(range=[0, 1]),
        height=500,
    )
    st.plotly_chart(fig, use_container_width=True)

    with st.expander("How to read calibration curves"):
        st.markdown(
            "- Points on the **diagonal** = perfectly calibrated "
            "(when the model says 70%, events happen 70% of the time)\n"
            "- Points **above** the diagonal = underconfident "
            "(events happen more often than predicted — model should be bolder)\n"
            "- Points **below** the diagonal = overconfident "
            "(events happen less often than predicted — model is too confident)\n"
            "- **Point size** = number of questions in that probability bin "
            "(larger = more data = more reliable)"
        )

    if metrics_rows:
        st.subheader("Calibration Metrics")
        st.markdown(
            "- **ECE** (Expected Calibration Error): Average miscalibration across all bins. "
            "Lower is better. 0 = perfect calibration. Typical range: 0.02–0.10.\n"
            "- **MCE** (Max Calibration Error): Worst single-bin miscalibration. "
            "Lower is better. Identifies the probability range where the model is "
            "most miscalibrated.\n"
            "- **Sharpness**: How far from 0.5 the forecasts are on average. "
            "Higher = more decisive. A model that always predicts 0.5 has sharpness of 0. "
            "Good models are both sharp AND well-calibrated."
        )
        df = pd.DataFrame(metrics_rows)
        df["ECE"] = df["ECE"].map("{:.4f}".format)
        df["MCE"] = df["MCE"].map("{:.4f}".format)
        df["Sharpness"] = df["Sharpness"].map("{:.6f}".format)
        st.dataframe(df, hide_index=True, use_container_width=True)


def view_question_browser(
    runs: list[AggregateRun], resolved_dicts: list[dict[str, Any]]
) -> None:
    st.header("Question Browser")

    if not runs or not resolved_dicts:
        st.warning("No data available.")
        return

    sources = sorted({d["source"] for d in resolved_dicts})
    run_names = [r.label for r in runs]

    col1, col2, col3 = st.columns(3)
    with col1:
        selected_sources = st.multiselect("Filter by source", sources, default=sources)
    with col2:
        outcome_filter = st.selectbox("Outcome (1=yes, 0=no)", ["All", "0", "1"])
    with col3:
        selected_runs = st.multiselect("Runs to show", run_names, default=run_names)

    sort_options = ["Question ID", "Worst Brier (any run)", "Most disagreement across runs", "Source"]
    sort_by = st.selectbox("Sort by", sort_options, index=0, key="qb_sort")

    with st.expander("Column guide"):
        st.markdown(
            "- **(f)** columns: The model's forecast probability "
            "(0.0 = confident 'no', 0.5 = uncertain, 1.0 = confident 'yes')\n"
            "- **(bs)** columns: Brier Score for that forecast "
            "(0.0 = perfect, 1.0 = worst possible). "
            "High values mean the model was wrong AND confident.\n"
            "- **Max Brier**: The worst Brier Score across all shown runs for that question. "
            "Sort by this to find questions ALL models struggle with.\n"
            "- **Disagreement**: Difference between the highest and lowest forecast across runs. "
            "High disagreement = models see this question very differently — "
            "interesting for analysis.\n"
            "- **Outcome (1=yes, 0=no)**: What actually happened. "
            "1 = the event occurred, 0 = it did not."
        )

    filtered = [
        d
        for d in resolved_dicts
        if d["source"] in selected_sources
        and (outcome_filter == "All" or d["outcome"] == int(outcome_filter))
    ]

    st.caption(f"N = {len(filtered):,} questions")

    rows: list[dict[str, Any]] = []
    for d in filtered:
        row: dict[str, Any] = {
            "ID": d["id"],
            "Question": d["question"][:100],
            "Source": d["source"],
            "Outcome (1=yes, 0=no)": d["outcome"],
        }
        briers: list[float] = []
        forecasts_list: list[float] = []
        for run_name in selected_runs:
            run = next(r for r in runs if r.label == run_name)
            f = _lookup_forecast(run.combined_forecasts, d["id"])
            bs = brier_score(f, d["outcome"])
            row[f"{run_name} (f)"] = round(f, 3)
            row[f"{run_name} (bs)"] = round(bs, 4)
            briers.append(bs)
            forecasts_list.append(f)
        row["Max Brier"] = round(max(briers), 4) if briers else 0
        row["Disagreement"] = round(max(forecasts_list) - min(forecasts_list), 3) if len(forecasts_list) > 1 else 0
        rows.append(row)

    if rows:
        df = pd.DataFrame(rows)
        if sort_by == "Question ID":
            df = df.sort_values("ID")
        elif sort_by == "Worst Brier (any run)":
            df = df.sort_values("Max Brier", ascending=False)
        elif sort_by == "Most disagreement across runs":
            df = df.sort_values("Disagreement", ascending=False)
        elif sort_by == "Source":
            df = df.sort_values("Source")
        st.dataframe(df, use_container_width=True, height=600)
    else:
        st.info("No questions match the current filters.")


def view_compare(runs: list[AggregateRun], resolved_dicts: list[dict[str, Any]]) -> None:
    st.header("Compare Runs")

    if len(runs) < 2:
        st.warning("Need at least 2 runs for comparison.")
        return

    run_names = [r.label for r in runs]
    col1, col2 = st.columns(2)
    with col1:
        run_a_name = st.selectbox("Run A", run_names, index=0, key="compare_a")
    with col2:
        default_b = 1 if len(run_names) > 1 else 0
        run_b_name = st.selectbox("Run B", run_names, index=default_b, key="compare_b")

    if run_a_name == run_b_name:
        st.info("Select two different runs to compare.")
        return

    run_a = next(r for r in runs if r.label == run_a_name)
    run_b = next(r for r in runs if r.label == run_b_name)

    forecasts_a = run_a.combined_forecasts
    forecasts_b = run_b.combined_forecasts
    outcomes = run_a.combined_outcomes or run_b.combined_outcomes

    shared_ids = set(forecasts_a.keys()) & set(forecasts_b.keys()) & set(outcomes.keys())
    if not shared_ids:
        st.warning("No shared questions between these runs.")
        return

    st.subheader("Aggregate Comparison")

    diffs: list[float] = []
    a_wins = 0
    b_wins = 0
    for qid in shared_ids:
        outcome = outcomes[qid]
        bs_a = (forecasts_a[qid] - outcome) ** 2
        bs_b = (forecasts_b[qid] - outcome) ** 2
        diffs.append(bs_a - bs_b)
        if bs_a < bs_b:
            a_wins += 1
        elif bs_b < bs_a:
            b_wins += 1

    n = len(diffs)
    mean_diff = sum(diffs) / n
    var_diff = sum((d - mean_diff) ** 2 for d in diffs) / (n - 1) if n > 1 else 0.0
    se = math.sqrt(var_diff / n) if n > 1 else 0.0
    t_stat = mean_diff / se if se > 0 else 0.0

    sr_a = run_a.scoring_result
    sr_b = run_b.scoring_result

    brier_a = sr_a.get("overall_brier", 0)
    brier_b = sr_b.get("overall_brier", 0)
    index_a = sr_a.get("overall_index", 0)
    index_b = sr_b.get("overall_index", 0)

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Shared Questions", f"{n:,}")
    m2.metric("Mean Brier A", f"{brier_a:.4f}", f"Index: {index_a:.1f}")
    m3.metric("Mean Brier B", f"{brier_b:.4f}", f"Index: {index_b:.1f}")
    m4.metric("t-statistic", f"{t_stat:+.3f}")
    st.caption(
        "t-statistic: measures statistical significance of the Brier score difference. "
        "|t| > 2.0 suggests a meaningful difference (not just noise). "
        "Positive = A is worse, negative = A is better."
    )

    w1, w2, w3 = st.columns(3)
    w1.metric(f"{run_a_name} wins", a_wins)
    w2.metric(f"{run_b_name} wins", b_wins)
    w3.metric("Ties", n - a_wins - b_wins)
    st.caption(
        'A "win" means that run had a lower Brier score '
        "(more accurate forecast) on that specific question."
    )

    resolved = _resolved_to_objects(resolved_dicts)
    by_source_a = analyze_by_source(forecasts_a, resolved)
    by_source_b = analyze_by_source(forecasts_b, resolved)
    all_sources = sorted(set(by_source_a.keys()) | set(by_source_b.keys()))

    source_labels: list[str] = []
    brier_diffs: list[float] = []
    colors: list[str] = []
    for source in all_sources:
        ba_val: Any = by_source_a.get(source, {}).get("brier", 0)
        bb_val: Any = by_source_b.get(source, {}).get("brier", 0)
        ba = float(ba_val)
        bb = float(bb_val)
        diff = ba - bb
        source_labels.append(source)
        brier_diffs.append(diff)
        colors.append("#2ca02c" if diff < 0 else "#d62728")

    fig = go.Figure(
        go.Bar(
            x=source_labels,
            y=brier_diffs,
            marker_color=colors,
            hovertemplate="Source: %{x}<br>Diff (A-B): %{y:.4f}<extra></extra>",
        )
    )
    fig.update_layout(
        title=f"Brier Difference by Source ({run_a_name} - {run_b_name})",
        xaxis_title="Source",
        yaxis_title="Brier Diff (negative = A better)",
        yaxis_zeroline=True,
    )
    st.plotly_chart(fig, use_container_width=True)

    st.divider()
    st.subheader("Head-to-Head Disagreements")

    threshold = st.slider("Minimum disagreement (|diff|)", 0.0, 1.0, 0.2, 0.05)
    st.caption(
        "Shows questions where the two runs disagreed by more than the threshold. "
        "|Diff| is the absolute difference in forecast probabilities. "
        '"Closer" indicates which run had the lower Brier score (more accurate).'
    )

    h2h_rows: list[dict[str, Any]] = []

    for d in resolved_dicts:
        qid = d["id"]
        fa = _lookup_forecast(run_a.combined_forecasts, qid)
        fb = _lookup_forecast(run_b.combined_forecasts, qid)
        diff = abs(fa - fb)
        if diff < threshold:
            continue

        outcome = d["outcome"]
        bs_a = brier_score(fa, outcome)
        bs_b = brier_score(fb, outcome)
        closer = run_a_name if bs_a < bs_b else (run_b_name if bs_b < bs_a else "Tie")

        h2h_rows.append(
            {
                "Question": d["question"][:100],
                "Source": d["source"],
                f"{run_a_name}": round(fa, 3),
                f"{run_b_name}": round(fb, 3),
                "Outcome": outcome,
                f"{run_a_name} BS": round(bs_a, 4),
                f"{run_b_name} BS": round(bs_b, 4),
                "Closer": closer,
                "|Diff|": round(diff, 3),
            }
        )

    h2h_rows.sort(key=lambda r: r["|Diff|"], reverse=True)

    st.caption(f"{len(h2h_rows):,} questions with |diff| >= {threshold:.2f}")

    if h2h_rows:
        a_closer = sum(1 for r in h2h_rows if r["Closer"] == run_a_name)
        b_closer = sum(1 for r in h2h_rows if r["Closer"] == run_b_name)
        ties = sum(1 for r in h2h_rows if r["Closer"] == "Tie")
        m1, m2, m3 = st.columns(3)
        m1.metric(f"{run_a_name} closer", a_closer)
        m2.metric(f"{run_b_name} closer", b_closer)
        m3.metric("Ties", ties)

        df = pd.DataFrame(h2h_rows)
        st.dataframe(df, use_container_width=True, height=600)
    else:
        st.info("No questions exceed the disagreement threshold.")


def main() -> None:
    st.set_page_config(page_title="ForecastBench Dashboard", layout="wide")
    st.title("ForecastBench Experiment Dashboard")

    all_results = load_all_results()
    if not all_results:
        st.error(
            "No result files found in results/ directory. "
            "Run `uv run python eval.py` first to generate results."
        )
        return

    view_mode = st.sidebar.radio(
        "View mode",
        ["Aggregate", "Individual rounds"],
        index=0,
        key="view_mode",
    )

    if view_mode == "Individual rounds":
        runs = _group_results_into_runs(all_results)
        individual_runs: list[AggregateRun] = []
        for r in all_results:
            slug = str(r["model_slug"])
            rnd = _round_name_from_result(r)
            individual_label = f"{slug}/{rnd}"
            sr = r.get("scoring_result", {})
            individual_runs.append(AggregateRun(
                slug=individual_label,
                n_rounds=1,
                rounds=[rnd],
                combined_forecasts=r.get("forecasts", {}),
                combined_outcomes=r.get("outcomes", {}),
                scoring_result=sr,
                per_round_results=[r],
                combined_sources=r.get("sources", {}),
            ))
        runs = individual_runs
    else:
        run_dicts = group_results(all_results)
        runs = [_dict_to_aggregate(d) for d in run_dicts]

    run_names = [r.label for r in runs]
    selected_runs = st.sidebar.multiselect(
        "Select runs", run_names, default=run_names, key="global_runs"
    )
    runs = [r for r in runs if r.label in selected_runs]
    st.sidebar.caption(f"{len(runs)} of {len(run_names)} runs selected")

    st.sidebar.markdown(f"**{len(runs)} runs loaded**")
    for run in sorted(runs, key=lambda x: x.scoring_result.get("overall_brier", 1)):
        sr = run.scoring_result
        bs = sr.get("overall_brier", 0)
        idx = sr.get("overall_index", 0)
        if view_mode == "Aggregate":
            st.sidebar.text(f"{run.label} ({run.n_rounds} rounds): Index {idx:.1f}")
        else:
            st.sidebar.text(f"{run.label}: Index {idx:.1f} (Brier {bs:.4f})")
    st.sidebar.markdown(
        "---\n"
        "Backtest of [ForecastBench](https://www.forecastbench.org/) "
        "tournament rounds. See the **About** tab for methodology."
    )

    tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs(
        [
            "Overview",
            "Leaderboard",
            "Failures",
            "Heatmap",
            "Compare",
            "Calibration",
            "Questions",
            "About",
        ]
    )

    with st.spinner("Loading question data..."):
        resolved_dicts = load_resolved_questions()

    with tab1:
        view_overview(runs, resolved_dicts)
    with tab2:
        view_leaderboard(runs)
    with tab3:
        view_failures(runs, resolved_dicts)
    with tab4:
        view_heatmap(runs, resolved_dicts)
    with tab5:
        view_compare(runs, resolved_dicts)
    with tab6:
        view_calibration(runs, resolved_dicts)
    with tab7:
        view_question_browser(runs, resolved_dicts)
    with tab8:
        view_about()


def view_about() -> None:
    st.header("About ForecastBench")

    st.markdown("""
### What is ForecastBench?

[**ForecastBench**](https://www.forecastbench.org/) is a dynamic benchmark for evaluating
AI forecasting models, published at **ICLR 2025** and maintained by the
[Forecasting Research Institute](https://forecastingresearch.org/). Unlike static
benchmarks, ForecastBench uses questions about *future events* — eliminating data
contamination by design.

---

### How It Works

- **~1,000 binary forecasting questions** are drawn from 9 sources and updated on a
  **biweekly schedule** (new "rounds").
- Models receive a question and must output a **probability ∈ [0, 1]** that the event
  will occur.
- Because questions concern events that haven't happened yet, no model can have seen the
  answer during training.

---

### Question Sources

Questions fall into two tracks:

**Dataset sources** (time series) — numeric data questions
(*"Will GDP exceed X?" / "Will the stock price be above Y?"*)

| Source | Domain |
|--------|--------|
| `acled` | Armed conflict & protest events |
| `wikipedia` | Wikipedia page view counts |
| `yfinance` | Stock prices & financial data |
| `dbnomics` | Macroeconomic indicators |
| `fred` | Federal Reserve economic data |

**Market sources** (prediction markets / forecasting platforms) — event-based questions
from human forecasting communities

| Source | Platform |
|--------|----------|
| `metaculus` | Metaculus |
| `polymarket` | Polymarket |
| `manifold` | Manifold Markets |
| `infer` | INFER (formerly Good Judgment Open) |

---

### Scoring

**Brier Score** (per question):

$$BS = (forecast - outcome)^2$$

- Range **[0, 1]** — *lower is better*
- Measures both **calibration** (are your 70% forecasts right 70% of the time?)
  and **accuracy** (are you confident on the right questions?)

**Brier Index** (aggregate):

$$BI = (1 - \\sqrt{\\overline{BS}}) \\times 100$$

- Range **(-∞, 100]** — *higher is better*, percentage scale
- The square-root transform is applied **after** averaging across questions, not
  per-question

A Brier Index of **0** corresponds to always predicting 0.5 (no skill). Human
superforecasters typically score in the **60–70** range.

---

### Our Setup

This dashboard visualizes results from a **backtest harness** that replays historical
ForecastBench rounds. Key details:

- We use the **same prompts, scoring, and question sets** as the official tournament
- Each "run" in this dashboard represents a **model + configuration**
  (e.g., different prompts, temperature settings, thinking mode, ensemble size)
- Results files store **per-question forecasts** so we can compare models at any
  granularity — by source, by question, by round
- Missing forecasts **default to 0.5** per ForecastBench rules
- **Binary outcomes only**: each question resolves to 0 or 1

---

### Key Conventions

| Convention | Detail |
|------------|--------|
| Missing forecasts | Default to **0.5** (no-skill baseline) |
| Outcomes | Binary only: **{0, 1}** |
| Difficulty adjustment | Available but dashboard shows **raw scores** by default |
| Brier Index transform | Applied **after** averaging, not per-question |
""")


if __name__ == "__main__":
    main()
