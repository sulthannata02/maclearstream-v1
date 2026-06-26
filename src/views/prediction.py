"""
Prediction View — Single dan batch prediction.
"""

from __future__ import annotations

import streamlit as st

from config.settings import get_config
from src.layouts.header import render_header
from src.components.widgets import empty_state
from src.components.forms import build_prediction_form, build_upload_form
from services.prediction import predict_single, predict_batch, get_available_models
from services.export import export_predictions_csv
from utils.helpers import format_percentage


def show() -> None:
    """Render halaman Prediksi."""
    render_header(
        "Prediksi",
        "Input manual atau upload CSV untuk batch prediction",
        emoji="🔮",
    )

    config = get_config()
    task_type = config.get("project", {}).get("task_type", "classification")

    available = get_available_models()

    if not available:
        empty_state(
            "🤖",
            "Belum ada model yang di-train. "
            "Buka halaman Training terlebih dahulu.",
        )
        return

    # ─── Model Selection ───
    selected_model = st.selectbox(
        "Pilih Model",
        available,
        key="predict_model_select",
    )

    st.markdown("---")

    # ─── Tabs ───
    tab_single, tab_batch = st.tabs([
        "✏️ Input Manual",
        "📄 Upload CSV",
    ])

    # ─── Tab: Single Prediction ───
    with tab_single:
        st.markdown("### Masukkan Nilai Fitur")

        values = build_prediction_form()

        if values is not None:
            if st.button(
                "🔮 Prediksi",
                type="primary",
                use_container_width=True,
                key="predict_single_btn",
            ):
                try:
                    result = predict_single(selected_model, values)
                    prediction = result.get("prediction")
                    probability = result.get("probability")

                    st.markdown("---")
                    st.markdown("### 📋 Hasil Prediksi")

                    if task_type == "classification":
                        # Classification result
                        col1, col2 = st.columns(2)

                        with col1:
                            if prediction == 1:
                                st.error(f"⚠️ Prediksi: **Positif (1)**")
                            else:
                                st.success(f"✅ Prediksi: **Negatif (0)**")

                        with col2:
                            if probability:
                                st.metric(
                                    "Confidence",
                                    format_percentage(max(probability)),
                                )

                        if probability:
                            st.markdown("#### Probabilitas per Kelas")
                            prob_cols = st.columns(len(probability))
                            for i, (col, p) in enumerate(
                                zip(prob_cols, probability)
                            ):
                                with col:
                                    st.metric(f"Class {i}", f"{p:.4f}")

                    else:
                        # Regression result
                        st.metric(
                            "Predicted Value",
                            f"{prediction:.4f}",
                        )

                except Exception as e:
                    st.error(f"❌ Prediksi gagal: {e}")
                    if hasattr(e, "hint"):
                        st.info(f"💡 {e.hint}")

    # ─── Tab: Batch Prediction ───
    with tab_batch:
        st.markdown("### Upload CSV untuk Batch Prediction")

        df = build_upload_form()

        if df is not None:
            if st.button(
                "🔮 Prediksi Batch",
                type="primary",
                use_container_width=True,
                key="predict_batch_btn",
            ):
                try:
                    with st.spinner("Processing..."):
                        result_df = predict_batch(selected_model, df)

                    st.success(
                        f"✅ Prediksi selesai: {len(result_df)} baris"
                    )

                    st.markdown("#### Hasil Prediksi")
                    st.dataframe(result_df, use_container_width=True)

                    # Download button
                    csv_bytes = export_predictions_csv(result_df)
                    st.download_button(
                        "📥 Download Hasil (CSV)",
                        data=csv_bytes,
                        file_name="predictions.csv",
                        mime="text/csv",
                    )

                    # Summary
                    if "Prediction" in result_df.columns:
                        st.markdown("#### Summary")
                        pred_counts = result_df["Prediction"].value_counts()
                        summary_cols = st.columns(len(pred_counts))
                        for col, (label, count) in zip(
                            summary_cols, pred_counts.items()
                        ):
                            with col:
                                st.metric(f"Class {label}", count)

                except Exception as e:
                    st.error(f"❌ Batch prediction gagal: {e}")
                    if hasattr(e, "hint"):
                        st.info(f"💡 {e.hint}")
