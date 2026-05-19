"""Plotly charts for SHAP-based explainability."""

from __future__ import annotations

import plotly.graph_objects as go

from trading_copilot.explain.shap_explain import ExplanationResult


def shap_waterfall_chart(
    explanation: ExplanationResult,
    title: str = "SHAP Feature Contributions",
) -> go.Figure:
    """Horizontal bar chart of SHAP values, colored green (bullish) / red (bearish)."""
    features = explanation.top_features or sorted(
        explanation.shap_values.items(), key=lambda x: abs(x[1]), reverse=True,
    )[:15]

    if not features:
        fig = go.Figure()
        fig.add_annotation(text="No explanation data available", showarrow=False)
        fig.update_layout(title=title)
        return fig

    # Reverse so largest bar is at top
    names = [f[0] for f in reversed(features)]
    values = [f[1] for f in reversed(features)]
    colors = ["#26a69a" if v >= 0 else "#ef5350" for v in values]

    fig = go.Figure(go.Bar(
        x=values,
        y=names,
        orientation="h",
        marker_color=colors,
        text=[f"{v:+.4f}" for v in values],
        textposition="outside",
    ))

    fig.update_layout(
        title=title,
        xaxis_title="SHAP Value",
        yaxis_title="Feature",
        template="plotly_white",
        height=max(300, len(names) * 30 + 100),
    )
    return fig


def feature_importance_chart(
    explanation: ExplanationResult,
    top_n: int = 15,
) -> go.Figure:
    """Bar chart of absolute SHAP values (feature importance)."""
    if not explanation.shap_values:
        fig = go.Figure()
        fig.add_annotation(text="No explanation data available", showarrow=False)
        return fig

    ranked = sorted(
        explanation.shap_values.items(), key=lambda x: abs(x[1]), reverse=True,
    )[:top_n]

    # Reverse for top-down display
    names = [f[0] for f in reversed(ranked)]
    abs_values = [abs(f[1]) for f in reversed(ranked)]

    fig = go.Figure(go.Bar(
        x=abs_values,
        y=names,
        orientation="h",
        marker_color="#42a5f5",
        text=[f"{v:.4f}" for v in abs_values],
        textposition="outside",
    ))

    fig.update_layout(
        title="Feature Importance (|SHAP|)",
        xaxis_title="Absolute SHAP Value",
        yaxis_title="Feature",
        template="plotly_white",
        height=max(300, len(names) * 30 + 100),
    )
    return fig
