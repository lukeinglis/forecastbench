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
from fetch_data import ResolvedQuestion, load_data
from score import brier_score


ResultData = dict[str, Any]

RESULTS_DIR = Path("results")


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


def view_heatmap(results: list[ResultData], resolved_dicts: list[dict[str, Any]]) -> None:
    st.header("Run × Source Heatmap")
    st.caption("Brier score by run and source. Darker = better (lower score).")

    if not results:
        st.warning("No result files found in results/ directory.")
        return

    run_labels, sources, matrix, counts = _build_source_brier_matrix(results, resolved_dicts)

    hover_text: list[list[str]] = []
    for i, run in enumerate(run_labels):
        row: list[str] = []
        for j, source in enumerate(sources):
            val = matrix[i][j]
            n = counts[i][j]
            if val is not None:
                row.append(f"Run: {run}<br>Source: {source}<br>Brier: {val:.3f}<br>N: {n}")
            else:
                row.append(f"Run: {run}<br>Source: {source}<br>No data")
        hover_text.append(row)

    annotations: list[dict[str, Any]] = []
    for i, run in enumerate(run_labels):
        for j, _ in enumerate(sources):
            val = matrix[i][j]
            if val is not None:
                annotations.append(
                    dict(
                        x=j,
                        y=i,
                        text=f"{val:.3f}",
                        showarrow=False,
                        font=dict(size=10, color="white" if val > 0.15 else "black"),
                    )
                )

    fig = go.Figure(
        data=go.Heatmap(
            z=matrix,
            x=sources,
            y=run_labels,
            colorscale="Viridis_r",
            hovertext=hover_text,
            hoverinfo="text",
            colorbar=dict(title="Brier"),
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
        st.markdown("**By Run (overall Brier)**")
        for result in sorted(results, key=lambda r: r.get("scoring_result", {}).get("overall_brier", 1.0)):
            sr = result.get("scoring_result", {})
            st.text(f"{_run_label(result)}: {sr.get('overall_brier', 0):.4f}")
    with col2:
        st.markdown("**By Source (avg across runs)**")
        source_avgs: dict[str, list[float]] = {}
        for i, run in enumerate(run_labels):
            for j, source in enumerate(sources):
                val = matrix[i][j]
                if val is not None:
                    source_avgs.setdefault(source, []).append(val)
        for source in sources:
            vals = source_avgs.get(source, [])
            avg = sum(vals) / len(vals) if vals else 0
            st.text(f"{source}: {avg:.4f} (n={len(vals)} runs)")


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

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Shared Questions", f"{n:,}")
    m2.metric("Mean Brier A", f"{sr_a.get('overall_brier', 0):.4f}")
    m3.metric("Mean Brier B", f"{sr_b.get('overall_brier', 0):.4f}")
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
        import pandas as pd

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

    import pandas as pd

    sources = sorted({d["source"] for d in resolved_dicts})
    run_names = [_run_label(r) for r in results]

    col1, col2, col3 = st.columns(3)
    with col1:
        selected_sources = st.multiselect("Filter by source", sources, default=sources)
    with col2:
        outcome_filter = st.selectbox("Outcome", ["All", "0", "1"])
    with col3:
        selected_runs = st.multiselect("Runs to show", run_names, default=run_names)

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
            "Outcome": d["outcome"],
        }
        for run_name in selected_runs:
            result = next(r for r in results if _run_label(r) == run_name)
            f = _lookup_forecast(result["forecasts"], d["id"])
            bs = brier_score(f, d["outcome"])
            row[f"{run_name} (f)"] = round(f, 3)
            row[f"{run_name} (bs)"] = round(bs, 4)
        rows.append(row)

    if rows:
        df = pd.DataFrame(rows)
        st.dataframe(df, use_container_width=True, height=600)
    else:
        st.info("No questions match the current filters.")


def view_head_to_head(results: list[ResultData], resolved_dicts: list[dict[str, Any]]) -> None:
    st.header("Head-to-Head Comparison")

    if len(results) < 2:
        st.warning("Need at least 2 result files.")
        return

    import pandas as pd

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
        st.sidebar.text(f"{_run_label(r)}: {sr.get('overall_brier', 0):.4f}")

    tab1, tab2, tab3, tab4, tab5 = st.tabs(
        [
            "Heatmap",
            "Pairwise",
            "Calibration",
            "Questions",
            "Head-to-Head",
        ]
    )

    with st.spinner("Loading question data..."):
        resolved_dicts = load_resolved_questions()

    with tab1:
        view_heatmap(results, resolved_dicts)
    with tab2:
        view_pairwise(results, resolved_dicts)
    with tab3:
        view_calibration(results, resolved_dicts)
    with tab4:
        view_question_browser(results, resolved_dicts)
    with tab5:
        view_head_to_head(results, resolved_dicts)


if __name__ == "__main__":
    main()
