"""
Experiments View — History dan perbandingan experiment runs.
"""

from __future__ import annotations

import pandas as pd  # type: ignore # pyright: ignore[reportMissingImports]
import streamlit as st  # type: ignore # pyright: ignore[reportMissingImports]

from src.layouts.header import render_header  # type: ignore # pyright: ignore[reportMissingImports]
from src.components.widgets import empty_state  # type: ignore # pyright: ignore[reportMissingImports]
from core.experiment import ExperimentTracker  # type: ignore # pyright: ignore[reportMissingImports]


def show() -> None:
    """Render halaman Experiments."""
    render_header(
        "Experiment History",
        "Riwayat semua training runs dan perbandingan metrik",
        emoji="📋",
    )

    tracker = ExperimentTracker()
    runs = tracker.list_runs()

    if not runs:
        empty_state(
            "📋",
            "Belum ada experiment. "
            "Train model terlebih dahulu untuk mulai logging.",
        )
        return

    # ─── Summary ───
    completed = [r for r in runs if r.get("status") == "completed"]
    failed = [r for r in runs if r.get("status") == "failed"]

    cols = st.columns(3)
    with cols[0]:
        st.metric("Total Runs", len(runs))
    with cols[1]:
        st.metric("Completed", len(completed))
    with cols[2]:
        st.metric("Failed", len(failed))

    st.markdown("---")

    # ─── Experiment Table ───
    st.markdown("### 📊 Semua Runs")

    table_data = []
    for run in runs:
        row = {
            "Run ID": run.get("run_id", ""),
            "Name": run.get("name", ""),
            "Model": run.get("model_name", ""),
            "Status": run.get("status", ""),
            "Duration (s)": round(run.get("execution_duration") or 0.0, 2),
            "Start": run.get("start_time", "")[:19],
        }

        # Tambahkan metrik utama
        metrics = run.get("metrics", {})
        for k, v in metrics.items():
            row[k.upper()] = round(v, 4) if isinstance(v, float) else v

        table_data.append(row)

    df = pd.DataFrame(table_data)
    st.dataframe(df, use_container_width=True, hide_index=True)

    # ─── Best Run ───
    st.markdown("---")
    st.markdown("### 🏆 Best Run")

    # Detect available metrics
    all_metric_keys = set()
    for run in completed:
        all_metric_keys.update(run.get("metrics", {}).keys())

    if all_metric_keys:
        metric_for_best = st.selectbox(
            "Best berdasarkan metrik:",
            sorted(all_metric_keys),
            key="exp_best_metric",
        )

        best = tracker.get_best_run(metric_for_best)
        if best:
            duration = best.get('execution_duration') or 0.0
            st.success(
                f"🏆 **{best['name']}** (Run: {best['run_id']}) — "
                f"{metric_for_best}: "
                f"{best['metrics'].get(metric_for_best, 0):.4f} | Waktu Eksekusi: {duration:.2f}s"
            )

            with st.expander("Detail Run"):
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("**Metrics:**")
                    st.json(best.get("metrics", {}))
                    st.markdown("**Tags:**")
                    st.json(best.get("tags", {}))
                with col2:
                    st.markdown("**Parameters:**")
                    st.json(best.get("params", {}))

    # ─── Clear Experiments ───
    st.markdown("---")
    with st.expander("🗑️ Danger Zone"):
        if st.button("Hapus Semua Experiment", type="secondary"):
            count = tracker.clear_all()
            st.success(f"✅ {count} experiment dihapus.")
            st.rerun()

