"""
Training View — Train model via UI.

Fitur: pilih model, set hyperparameters, K-Fold CV, progress, save artifacts.
"""

from __future__ import annotations

from pathlib import Path

import streamlit as st

from config.settings import get_config, get_paths_config
from src.layouts.header import render_header
from src.components.widgets import empty_state, status_badge
from core.experiment import ExperimentTracker
from models.registry import ModelRegistry
from services.preprocessing import preprocess
from services.training import train_model, cross_validate_model
from services.evaluation import evaluate_model
from utils.file_handler import save_artifact


def show() -> None:
    """Render halaman Training."""
    render_header(
        "Training Model",
        "Pilih model, atur parameter, dan train langsung dari UI",
        emoji="🧠",
    )

    config = get_config()
    paths = get_paths_config()
    artifacts_dir = Path(paths.get("artifacts_dir", "artifacts"))
    task_type = config.get("project", {}).get("task_type", "classification")

    # ─── Model Selection ───
    try:
        registry = ModelRegistry.from_config(config)
    except Exception as e:
        st.error(f"Gagal load model registry: {e}")
        return

    available_models = registry.list_models()

    if not available_models:
        empty_state("🤖", "Belum ada model di config.yaml.")
        return

    model_names = [m["display_name"] for m in available_models]
    model_ids = [m["name"] for m in available_models]

    col1, col2 = st.columns([2, 1])

    with col1:
        selected_display = st.selectbox(
            "Pilih Model",
            model_names,
            key="train_model_select",
        )
        selected_id = model_ids[model_names.index(selected_display)]

    with col2:
        cv_folds = st.selectbox(
            "Cross-Validation Folds",
            [0, 3, 5, 10],
            index=0,
            help="0 = tanpa CV, langsung train",
            key="train_cv_folds",
        )

    # Show current params
    model_wrapper = registry.get(selected_id)
    params = model_wrapper.get_params()

    with st.expander("⚙️ Hyperparameters", expanded=False):
        st.json(params)

    st.markdown("---")

    # ─── Train Button ───
    col_train, col_train_all = st.columns(2)

    with col_train:
        train_single = st.button(
            f"🚀 Train {selected_display}",
            use_container_width=True,
            type="primary",
        )

    with col_train_all:
        train_all = st.button(
            "🚀 Train Semua Model",
            use_container_width=True,
        )

    # ─── Training Logic ───
    if train_single or train_all:
        try:
            with st.spinner("📦 Preprocessing dataset..."):
                data = preprocess(config)

            models_to_train = (
                model_ids if train_all else [selected_id]
            )

            progress_bar = st.progress(0)
            tracker = ExperimentTracker()

            for i, model_id in enumerate(models_to_train):
                display_name = registry.get_display_name(model_id)
                st.markdown(f"#### Training: {display_name}")

                # Cross-validation (opsional)
                if cv_folds > 0:
                    with st.spinner(f"🔄 Cross-validation ({cv_folds}-fold)..."):
                        cv_result = cross_validate_model(
                            model_id,
                            data["X_train"],
                            data["y_train"],
                            cv=cv_folds,
                            config=config,
                        )
                        st.info(
                            f"CV Score: {cv_result['mean']:.4f} "
                            f"± {cv_result['std']:.4f}"
                        )

                # Train
                with st.spinner(f"🧠 Training {display_name}..."):
                    trained_model = train_model(
                        model_id,
                        data["X_train"],
                        data["y_train"],
                        config=config,
                    )

                # Evaluate
                metrics = evaluate_model(
                    trained_model,
                    data["X_test"],
                    data["y_test"],
                    task_type,
                )

                # Save artifacts
                model_path = artifacts_dir / f"{model_id}.pkl"
                save_artifact(trained_model.model, model_path)

                if data.get("scaler"):
                    save_artifact(data["scaler"], artifacts_dir / "scaler.pkl")
                save_artifact(
                    data["feature_names"],
                    artifacts_dir / "feature_names.pkl",
                )

                # Log experiment
                scalar_metrics = {
                    k: v for k, v in metrics.items()
                    if isinstance(v, (int, float)) and v is not None
                }

                with tracker.start_run(model_id) as run:
                    run.log_model(model_id)
                    run.log_params(trained_model.get_params())
                    run.log_metrics(scalar_metrics)

                # Display metrics
                metric_cols = st.columns(len(scalar_metrics))
                for col, (k, v) in zip(metric_cols, scalar_metrics.items()):
                    with col:
                        st.metric(k.upper(), f"{v:.4f}")

                st.success(f"✅ {display_name} berhasil di-train dan disimpan!")

                progress_bar.progress((i + 1) / len(models_to_train))

            st.balloons()

        except Exception as e:
            st.error(f"❌ Training gagal: {e}")
