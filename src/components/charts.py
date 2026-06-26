"""
Charts — Plotly chart wrappers untuk visualisasi ML.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff
import streamlit as st


# ──────────────────────────────────────────────
# Classification Charts
# ──────────────────────────────────────────────

def plot_confusion_matrix(
    cm: np.ndarray,
    labels: list[str] | None = None,
) -> None:
    """Render confusion matrix sebagai heatmap."""
    if labels is None:
        labels = [f"Class {i}" for i in range(cm.shape[0])]

    fig = ff.create_annotated_heatmap(
        z=cm,
        x=labels,
        y=labels,
        colorscale="Purples",
        showscale=True,
    )
    fig.update_layout(
        title="Confusion Matrix",
        xaxis_title="Predicted",
        yaxis_title="Actual",
        template="plotly_dark",
        height=400,
        margin=dict(l=60, r=20, t=60, b=60),
    )
    fig.update_yaxes(autorange="reversed")
    st.plotly_chart(fig, use_container_width=True)


def plot_roc_curve(
    y_test: np.ndarray,
    y_proba: np.ndarray,
    auc_score: float | None = None,
) -> None:
    """Render ROC curve."""
    from sklearn.metrics import roc_curve

    fpr, tpr, _ = roc_curve(y_test, y_proba)

    label = "ROC Curve"
    if auc_score is not None:
        label = f"ROC Curve (AUC = {auc_score:.4f})"

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=fpr, y=tpr,
        mode="lines",
        name=label,
        line=dict(color="#6C63FF", width=2),
    ))
    fig.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1],
        mode="lines",
        name="Random",
        line=dict(color="gray", width=1, dash="dash"),
    ))
    fig.update_layout(
        title="ROC Curve",
        xaxis_title="False Positive Rate",
        yaxis_title="True Positive Rate",
        template="plotly_dark",
        height=400,
        margin=dict(l=60, r=20, t=60, b=60),
        showlegend=True,
    )
    st.plotly_chart(fig, use_container_width=True)


def plot_feature_importance(
    importances: np.ndarray,
    feature_names: list[str],
    top_n: int = 15,
) -> None:
    """Render feature importance sebagai horizontal bar chart."""
    # Sort by importance
    indices = np.argsort(importances)[-top_n:]
    sorted_names = [feature_names[i] for i in indices]
    sorted_values = importances[indices]

    fig = go.Figure(go.Bar(
        x=sorted_values,
        y=sorted_names,
        orientation="h",
        marker_color="#6C63FF",
    ))
    fig.update_layout(
        title=f"Feature Importance (Top {min(top_n, len(feature_names))})",
        xaxis_title="Importance",
        template="plotly_dark",
        height=max(300, top_n * 30),
        margin=dict(l=120, r=20, t=60, b=40),
    )
    st.plotly_chart(fig, use_container_width=True)


# ──────────────────────────────────────────────
# EDA Charts
# ──────────────────────────────────────────────

def plot_correlation_heatmap(df: pd.DataFrame) -> None:
    """Render correlation matrix heatmap."""
    corr = df.corr(numeric_only=True)

    fig = go.Figure(data=go.Heatmap(
        z=corr.values,
        x=corr.columns.tolist(),
        y=corr.columns.tolist(),
        colorscale="RdBu_r",
        zmin=-1,
        zmax=1,
        text=corr.round(2).values,
        texttemplate="%{text}",
        textfont=dict(size=10),
    ))
    fig.update_layout(
        title="Correlation Matrix",
        template="plotly_dark",
        height=500,
        margin=dict(l=80, r=20, t=60, b=80),
    )
    st.plotly_chart(fig, use_container_width=True)


def plot_distribution(
    df: pd.DataFrame,
    column: str,
    target: str | None = None,
) -> None:
    """Render histogram + KDE untuk satu kolom."""
    if target and target in df.columns:
        fig = px.histogram(
            df, x=column, color=target,
            marginal="box",
            barmode="overlay",
            opacity=0.7,
            template="plotly_dark",
        )
    else:
        fig = px.histogram(
            df, x=column,
            marginal="box",
            opacity=0.7,
            template="plotly_dark",
        )

    fig.update_layout(
        title=f"Distribusi: {column}",
        height=400,
        margin=dict(l=60, r=20, t=60, b=40),
    )
    st.plotly_chart(fig, use_container_width=True)


def plot_target_distribution(
    df: pd.DataFrame,
    target: str,
) -> None:
    """Render pie chart distribusi target."""
    counts = df[target].value_counts()

    fig = go.Figure(data=[go.Pie(
        labels=counts.index.astype(str),
        values=counts.values,
        hole=0.4,
        marker_colors=["#6C63FF", "#FF6584", "#17B890", "#F59E0B"],
    )])
    fig.update_layout(
        title=f"Distribusi Target: {target}",
        template="plotly_dark",
        height=400,
        margin=dict(l=20, r=20, t=60, b=20),
    )
    st.plotly_chart(fig, use_container_width=True)


# ──────────────────────────────────────────────
# Model Comparison
# ──────────────────────────────────────────────

def plot_model_comparison(
    comparison_df: pd.DataFrame,
    metric_columns: list[str] | None = None,
) -> None:
    """Render grouped bar chart perbandingan model."""
    if metric_columns is None:
        metric_columns = [
            c for c in comparison_df.columns
            if c != "Model"
        ]

    fig = go.Figure()
    colors = ["#6C63FF", "#FF6584", "#17B890", "#F59E0B", "#3B82F6"]

    for i, metric in enumerate(metric_columns):
        values = comparison_df[metric].tolist()
        # Convert string '-' to 0
        numeric_values = [
            float(v) if v != "-" else 0
            for v in values
        ]
        fig.add_trace(go.Bar(
            name=metric,
            x=comparison_df["Model"].tolist(),
            y=numeric_values,
            marker_color=colors[i % len(colors)],
        ))

    fig.update_layout(
        title="Model Comparison",
        barmode="group",
        template="plotly_dark",
        height=450,
        margin=dict(l=60, r=20, t=60, b=60),
        legend=dict(orientation="h", y=-0.15),
    )
    st.plotly_chart(fig, use_container_width=True)


# ──────────────────────────────────────────────
# Regression Charts
# ──────────────────────────────────────────────

def plot_residuals(
    y_test: np.ndarray,
    y_pred: np.ndarray,
) -> None:
    """Render residual plot (regression)."""
    residuals = y_test - y_pred

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=y_pred, y=residuals,
        mode="markers",
        marker=dict(color="#6C63FF", opacity=0.6),
    ))
    fig.add_hline(y=0, line_dash="dash", line_color="gray")
    fig.update_layout(
        title="Residual Plot",
        xaxis_title="Predicted",
        yaxis_title="Residuals",
        template="plotly_dark",
        height=400,
    )
    st.plotly_chart(fig, use_container_width=True)


def plot_prediction_vs_actual(
    y_test: np.ndarray,
    y_pred: np.ndarray,
) -> None:
    """Render scatter plot prediction vs actual (regression)."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=y_test, y=y_pred,
        mode="markers",
        marker=dict(color="#6C63FF", opacity=0.6),
        name="Predictions",
    ))

    # Diagonal line
    min_val = min(y_test.min(), y_pred.min())
    max_val = max(y_test.max(), y_pred.max())
    fig.add_trace(go.Scatter(
        x=[min_val, max_val], y=[min_val, max_val],
        mode="lines",
        name="Ideal",
        line=dict(color="gray", dash="dash"),
    ))

    fig.update_layout(
        title="Prediction vs Actual",
        xaxis_title="Actual",
        yaxis_title="Predicted",
        template="plotly_dark",
        height=400,
    )
    st.plotly_chart(fig, use_container_width=True)
