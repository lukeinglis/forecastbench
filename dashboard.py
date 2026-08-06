"""Interactive experiment results dashboard for ForecastBench.

Launch: uv run --extra dashboard streamlit run dashboard.py
"""

from __future__ import annotations

import json
import math
import sys
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
from fetch_data import MARKET_SOURCES, ResolvedQuestion, load_data
from score import brier_index, brier_score


ResultData = dict[str, Any]

RESULTS_DIR = Path("results")

LEADERBOARD_REFERENCE: dict[str, dict[str, float]] = {
    "human_superforecaster": {"overall_index": 70.0, "overall_brier": 0.081},
    "sonnet_4_official": {
        "overall_index": 60.3,
        "dataset_index": 59.1,
        "market_index": 61.5,
        "overall_brier": 0.141,
    },
}


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


def _run_label(result: ResultData) -> str:
    return str(result["model_slug"])


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
    results: list[ResultData],
    resolved_dicts: list[dict[str, Any]],
) -> tuple[list[str], list[str], list[list[float | None]], list[list[int]]]:
    resolved = _resolved_to_objects(resolved_dicts)
    all_sources: set[str] = set()
    run_source_scores: dict[str, dict[str, float]] = {}
    run_source_counts: dict[str, dict[str, int]] = {}

    for result in results:
        label = _run_label(result)
        forecasts = result["forecasts"]
        by_source = analyze_by_source(forecasts, resolved)
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
    for result in results:
        sr = result.get("scoring_result", {})
        overall_brier[_run_label(result)] = sr.get("overall_brier", 1.0)
    run_labels = sorted(run_source_scores.keys(), key=lambda r: overall_brier.get(r, 1.0))

    matrix: list[list[float | None]] = []
    counts: list[list[int]] = []
    for run in run_labels:
        row: list[float | None] = []
        count_row: list[int] = []
        for source in sources:
            row.append(run_source_scores[run].get(source))
            count_row.append(run_source_counts[run].get(source, 0))
        matrix.append(row)
        counts.append(count_row)

    return run_labels, sources, matrix, counts


def _sort_sources_by_track(sources: list[str]) -> list[str]:
    dataset_sources = [s for s in sources if s not in MARKET_SOURCES]
    market_sources = [s for s in sources if s in MARKET_SOURCES]
    return sorted(dataset_sources) + sorted(market_sources)


def view_overview(results: list[ResultData], resolved_dicts: list[dict[str, Any]]) -> None:
    st.header("Overview")

    if not results:
        st.warning("No result files found in results/ directory.")
        return

    best_result = min(results, key=lambda r: r.get("scoring_result", {}).get("overall_brier", 1.0))
    best_sr = best_result.get("scoring_result", {})
    best_label = _run_label(best_result)
    best_index = best_sr.get("overall_index", 0.0)

    total_questions = best_sr.get("n_dataset", 0) + best_sr.get("n_market", 0)

    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Best Run", f"{best_label}", f"Index: {best_index:.1f}")
    m2.metric("Total Runs", len(results))
    m3.metric("Total Questions", f"{total_questions:,}")
    m4.metric("Dataset Index", f"{best_sr.get('dataset_index', 0):.1f}")
    m5.metric("Market Index", f"{best_sr.get('market_index', 0):.1f}")

    st.subheader("All Runs")
    rows: list[dict[str, Any]] = []
    for result in results:
        sr = result.get("scoring_result", {})
        n_total = sr.get("n_dataset", 0) + sr.get("n_market", 0)
        rows.append({
            "Run": _run_label(result),
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

    st.subheader("Leaderboard Reference")
    st.caption("Approximate reference points from the official ForecastBench leaderboard.")
    ref_rows: list[dict[str, Any]] = []
    for name, ref in LEADERBOARD_REFERENCE.items():
        ref_rows.append({
            "Reference": name.replace("_", " ").title(),
            "Overall Index": ref.get("overall_index", 0),
            "Overall Brier": ref.get("overall_brier", 0),
            "Dataset Index": ref.get("dataset_index"),
            "Market Index": ref.get("market_index"),
        })
    ref_df = pd.DataFrame(ref_rows)
    st.dataframe(ref_df, hide_index=True, use_container_width=True)


def view_heatmap(results: list[ResultData], resolved_dicts: list[dict[str, Any]]) -> None:
    st.header("Run × Source Heatmap")

    if not results:
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
        _view_heatmap_by_track(results, resolved_dicts, show_index)
    else:
        _view_heatmap_by_source(results, resolved_dicts, show_index)


def _view_heatmap_by_source(
    results: list[ResultData],
    resolved_dicts: list[dict[str, Any]],
    show_index: bool,
) -> None:
    run_labels, sources, matrix, counts = _build_source_brier_matrix(results, resolved_dicts)

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

    st.subheader("Marginal Averages")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**By Run**")
        for result in sorted(
            results, key=lambda r: r.get("scoring_result", {}).get("overall_brier", 1.0)
        ):
            sr = result.get("scoring_result", {})
            bs = sr.get("overall_brier", 0)
            idx = sr.get("overall_index", 0)
            st.text(f"{_run_label(result)}: Brier {bs:.4f} | Index {idx:.1f}")
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
    results: list[ResultData],
    resolved_dicts: list[dict[str, Any]],
    show_index: bool,
) -> None:
    overall_brier_map: dict[str, float] = {}
    for result in results:
        sr = result.get("scoring_result", {})
        overall_brier_map[_run_label(result)] = sr.get("overall_brier", 1.0)
    run_labels = sorted(overall_brier_map.keys(), key=lambda r: overall_brier_map.get(r, 1.0))

    tracks = ["dataset", "market"]
    track_matrix: list[list[float | None]] = []
    track_counts: list[list[int]] = []

    for run_label in run_labels:
        result = next(r for r in results if _run_label(r) == run_label)
        sr = result.get("scoring_result", {})
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


def view_pairwise(results: list[ResultData], resolved_dicts: list[dict[str, Any]]) -> None:
    st.header("Pairwise Run Comparison")

    if len(results) < 2:
        st.warning("Need at least 2 result files for pairwise comparison.")
        return

    run_names = [_run_label(r) for r in results]
    col1, col2 = st.columns(2)
    with col1:
        run_a_name = st.selectbox("Run A", run_names, index=0)
    with col2:
        default_b = 1 if len(run_names) > 1 else 0
        run_b_name = st.selectbox("Run B", run_names, index=default_b)

    if run_a_name == run_b_name:
        st.info("Select two different runs to compare.")
        return

    result_a = next(r for r in results if _run_label(r) == run_a_name)
    result_b = next(r for r in results if _run_label(r) == run_b_name)

    forecasts_a: dict[str, float] = result_a["forecasts"]
    forecasts_b: dict[str, float] = result_b["forecasts"]
    outcomes_a: dict[str, int] = result_a.get("outcomes", {})
    outcomes_b: dict[str, int] = result_b.get("outcomes", {})
    outcomes = outcomes_a or outcomes_b

    shared_ids = set(forecasts_a.keys()) & set(forecasts_b.keys()) & set(outcomes.keys())
    if not shared_ids:
        st.warning("No shared questions between these runs.")
        return

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

    sr_a = result_a.get("scoring_result", {})
    sr_b = result_b.get("scoring_result", {})

    brier_a = sr_a.get("overall_brier", 0)
    brier_b = sr_b.get("overall_brier", 0)
    index_a = sr_a.get("overall_index", 0)
    index_b = sr_b.get("overall_index", 0)

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Shared Questions", f"{n:,}")
    m2.metric("Mean Brier A", f"{brier_a:.4f}", f"Index: {index_a:.1f}")
    m3.metric("Mean Brier B", f"{brier_b:.4f}", f"Index: {index_b:.1f}")
    m4.metric("t-statistic", f"{t_stat:+.3f}")

    w1, w2, w3 = st.columns(3)
    w1.metric(f"{run_a_name} wins", a_wins)
    w2.metric(f"{run_b_name} wins", b_wins)
    w3.metric("Ties", n - a_wins - b_wins)

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


def view_failures(results: list[ResultData], resolved_dicts: list[dict[str, Any]]) -> None:
    st.header("Failure Explorer")

    if not results:
        st.warning("No result files found.")
        return

    run_names = [_run_label(r) for r in results]
    selected_run = st.selectbox("Select run", run_names, index=0, key="failures_run")
    result = next(r for r in results if _run_label(r) == selected_run)
    forecasts: dict[str, float] = result["forecasts"]

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

    st.subheader("Error Type Summary")
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

    st.subheader("Worst Questions")
    top_n = st.slider("Show top N worst", 10, 200, 50, key="worst_n")
    question_data.sort(key=lambda q: -q["_brier_raw"])
    worst_rows = [
        {k: v for k, v in q.items() if k != "_brier_raw"}
        for q in question_data[:top_n]
    ]
    worst_df = pd.DataFrame(worst_rows)
    st.dataframe(worst_df, hide_index=True, use_container_width=True, height=600)


def view_calibration(results: list[ResultData], resolved_dicts: list[dict[str, Any]]) -> None:
    st.header("Calibration Curves")

    if not results:
        st.warning("No result files found.")
        return

    run_names = [_run_label(r) for r in results]
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
        result = next(r for r in results if _run_label(r) == run_name)
        forecasts = result["forecasts"]
        cal_bins = analyze_calibration(forecasts, resolved, n_bins=n_bins)
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

        pairs = [(_lookup_forecast(forecasts, q.id), q.outcome) for q in resolved]
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

    if metrics_rows:
        st.subheader("Calibration Metrics")
        df = pd.DataFrame(metrics_rows)
        df["ECE"] = df["ECE"].map("{:.4f}".format)
        df["MCE"] = df["MCE"].map("{:.4f}".format)
        df["Sharpness"] = df["Sharpness"].map("{:.6f}".format)
        st.dataframe(df, hide_index=True, use_container_width=True)


def view_question_browser(
    results: list[ResultData], resolved_dicts: list[dict[str, Any]]
) -> None:
    st.header("Question Browser")

    if not results or not resolved_dicts:
        st.warning("No data available.")
        return

    sources = sorted({d["source"] for d in resolved_dicts})
    run_names = [_run_label(r) for r in results]

    col1, col2, col3 = st.columns(3)
    with col1:
        selected_sources = st.multiselect("Filter by source", sources, default=sources)
    with col2:
        outcome_filter = st.selectbox("Outcome (1=yes, 0=no)", ["All", "0", "1"])
    with col3:
        selected_runs = st.multiselect("Runs to show", run_names, default=run_names)

    sort_options = ["Question ID", "Worst Brier (any run)", "Most disagreement across runs", "Source"]
    sort_by = st.selectbox("Sort by", sort_options, index=0, key="qb_sort")

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
            result = next(r for r in results if _run_label(r) == run_name)
            f = _lookup_forecast(result["forecasts"], d["id"])
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


def view_head_to_head(results: list[ResultData], resolved_dicts: list[dict[str, Any]]) -> None:
    st.header("Head-to-Head Comparison")

    if len(results) < 2:
        st.warning("Need at least 2 result files.")
        return

    run_names = [_run_label(r) for r in results]
    col1, col2 = st.columns(2)
    with col1:
        run_a_name = st.selectbox("Run A", run_names, index=0, key="h2h_a")
    with col2:
        default_b = 1 if len(run_names) > 1 else 0
        run_b_name = st.selectbox("Run B", run_names, index=default_b, key="h2h_b")

    if run_a_name == run_b_name:
        st.info("Select two different runs.")
        return

    threshold = st.slider("Minimum disagreement (|diff|)", 0.0, 1.0, 0.2, 0.05)

    result_a = next(r for r in results if _run_label(r) == run_a_name)
    result_b = next(r for r in results if _run_label(r) == run_b_name)

    rows: list[dict[str, Any]] = []

    for d in resolved_dicts:
        qid = d["id"]
        fa = _lookup_forecast(result_a["forecasts"], qid)
        fb = _lookup_forecast(result_b["forecasts"], qid)
        diff = abs(fa - fb)
        if diff < threshold:
            continue

        outcome = d["outcome"]
        bs_a = brier_score(fa, outcome)
        bs_b = brier_score(fb, outcome)
        closer = run_a_name if bs_a < bs_b else (run_b_name if bs_b < bs_a else "Tie")

        rows.append(
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

    rows.sort(key=lambda r: r["|Diff|"], reverse=True)

    st.caption(f"{len(rows):,} questions with |diff| >= {threshold:.2f}")

    if rows:
        a_closer = sum(1 for r in rows if r["Closer"] == run_a_name)
        b_closer = sum(1 for r in rows if r["Closer"] == run_b_name)
        ties = sum(1 for r in rows if r["Closer"] == "Tie")
        m1, m2, m3 = st.columns(3)
        m1.metric(f"{run_a_name} closer", a_closer)
        m2.metric(f"{run_b_name} closer", b_closer)
        m3.metric("Ties", ties)

        df = pd.DataFrame(rows)
        st.dataframe(df, use_container_width=True, height=600)
    else:
        st.info("No questions exceed the disagreement threshold.")


def main() -> None:
    st.set_page_config(page_title="ForecastBench Dashboard", layout="wide")
    st.title("ForecastBench Experiment Dashboard")

    results = load_all_results()
    if not results:
        st.error(
            "No result files found in results/ directory. "
            "Run `uv run python eval.py` first to generate results."
        )
        return

    st.sidebar.markdown(f"**{len(results)} runs loaded**")
    for r in sorted(results, key=lambda x: x.get("scoring_result", {}).get("overall_brier", 1)):
        sr = r.get("scoring_result", {})
        bs = sr.get("overall_brier", 0)
        idx = sr.get("overall_index", 0)
        st.sidebar.text(f"{_run_label(r)}: Index {idx:.1f} (Brier {bs:.4f})")
    st.sidebar.markdown(
        "---\n"
        "Backtest of [ForecastBench](https://www.forecastbench.org/) "
        "tournament rounds. See the **About** tab for methodology."
    )

    tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs(
        [
            "Overview",
            "Heatmap",
            "Failures",
            "Pairwise",
            "Calibration",
            "Questions",
            "Head-to-Head",
            "About",
        ]
    )

    with st.spinner("Loading question data..."):
        resolved_dicts = load_resolved_questions()

    with tab1:
        view_overview(results, resolved_dicts)
    with tab2:
        view_heatmap(results, resolved_dicts)
    with tab3:
        view_failures(results, resolved_dicts)
    with tab4:
        view_pairwise(results, resolved_dicts)
    with tab5:
        view_calibration(results, resolved_dicts)
    with tab6:
        view_question_browser(results, resolved_dicts)
    with tab7:
        view_head_to_head(results, resolved_dicts)
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
