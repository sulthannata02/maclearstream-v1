"""
Evaluation View — Metrik model dan model comparison dashboard.
"""

from __future__ import annotations

from pathlib import Path

import streamlit as st

from config.settings import get_config, get_paths_config
from src.layouts.header import render_header
from src.components.widgets import empty_state
from src.components.charts import (
    plot_confusion_matrix,
    plot_roc_curve,
    plot_feature_importance,
    plot_model_comparison,
    plot_residuals,
    plot_prediction_vs_actual,
)
from services.preprocessing import preprocess
from services.evaluation import evaluate_model, compare_models
from services.prediction import get_available_models
from services.export import export_comparison_csv
from utils.file_handler import load_artifact
from utils.constants import CLASSIFICATION


def show() -> None:
    """Render halaman Evaluasi Model."""
    render_header(
        "Evaluasi Model",
        "Metrik performa, confusion matrix, ROC curve, dan perbandingan model",
        emoji="📈",
    )

    config = get_config()
    paths = get_paths_config()
    artifacts_dir = Path(paths.get("artifacts_dir", "artifacts"))
    task_type = config.get("project", {}).get("task_type", CLASSIFICATION)

    available = get_available_models()

    if not available:
        empty_state(
            "🤖",
            "Belum ada model yang di-train. "
            "Buka halaman Training untuk melatih model.",
        )
        return

    # ─── Evaluate All Models ───
    try:
        with st.spinner("Preprocessing & evaluating..."):
            data = preprocess(config)

            all_results: dict[str, dict] = {}
            all_models: dict[str, object] = {}

            for model_name in available:
                model_path = artifacts_dir / f"{model_name}.pkl"
                model = load_artifact(model_path)
                metrics = evaluate_model(
                    model, data["X_test"], data["y_test"], task_type,
                )
                all_results[model_name] = metrics
                all_models[model_name] = model

    except Exception as e:
        st.error(f"❌ Evaluasi gagal: {e}")
        return

    # ─── Model Comparison Table ───
    st.markdown("### 🏆 Perbandingan Model")

    comparison = compare_models(all_results, task_type)
    st.dataframe(comparison, use_container_width=True, hide_index=True)

    # Download button
    csv_bytes = export_comparison_csv(comparison)
    st.download_button(
        "📥 Download Perbandingan (CSV)",
        data=csv_bytes,
        file_name="model_comparison.csv",
        mime="text/csv",
    )

    # Chart
    plot_model_comparison(comparison)

    st.markdown("---")

    # ─── Per-Model Detail ───
    st.markdown("### 🔍 Detail Per Model")

    selected_model = st.selectbox(
        "Pilih model untuk detail",
        available,
        key="eval_model_select",
    )

    metrics = all_results[selected_model]

    # Scalar metrics
    if task_type == CLASSIFICATION:
        metric_keys = ["accuracy", "precision", "recall", "f1_score", "roc_auc"]
    else:
        metric_keys = ["mae", "mse", "rmse", "r2"]

    cols = st.columns(len(metric_keys))
    for col, key in zip(cols, metric_keys):
        val = metrics.get(key)
        with col:
            if val is not None:
                st.metric(key.upper(), f"{val:.4f}")
            else:
                st.metric(key.upper(), "-")

    # Visualizations
    if task_type == CLASSIFICATION:
        col1, col2 = st.columns(2)

        with col1:
            cm = metrics.get("confusion_matrix")
            if cm is not None:
                plot_confusion_matrix(cm)

        with col2:
            y_proba = metrics.get("y_proba")
            if y_proba is not None and len(y_proba.shape) == 1:
                plot_roc_curve(
                    data["y_test"],
                    y_proba,
                    auc_score=metrics.get("roc_auc"),
                )
    else:
        # Regression plots
        y_pred = metrics.get("y_pred")
        if y_pred is not None:
            col1, col2 = st.columns(2)
            with col1:
                plot_residuals(data["y_test"].values, y_pred)
            with col2:
                plot_prediction_vs_actual(data["y_test"].values, y_pred)

    # Feature importance
    model = all_models[selected_model]
    fi = None
    if hasattr(model, "feature_importances_"):
        fi = model.feature_importances_
    elif hasattr(model, "coef_"):
        import numpy as np
        fi = np.abs(model.coef_).flatten()

    if fi is not None:
        plot_feature_importance(fi, data["feature_names"])
