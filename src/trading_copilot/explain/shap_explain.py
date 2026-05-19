"""SHAP-based model explanations for tree-based classifiers."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import shap

from trading_copilot.logging_setup import get_logger

log = get_logger("explain.shap")


@dataclass
class ExplanationResult:
    """Container for SHAP explanation output."""

    shap_values: dict[str, float] = field(default_factory=dict)
    top_features: list[tuple[str, float]] = field(default_factory=list)
    summary_text: str = ""


def explain_prediction(
    model,
    features_row: pd.Series,
    feature_names: list[str],
    top_n: int = 10,
) -> ExplanationResult:
    """Explain a single prediction using SHAP TreeExplainer.

    Works with RandomForest, XGBoost, and LightGBM models.
    Returns an ExplanationResult with SHAP values for each feature.
    """
    try:
        explainer = shap.TreeExplainer(model)
        row_array = np.array(features_row.values).reshape(1, -1)
        sv = explainer.shap_values(row_array)

        # SHAP returns different shapes depending on version/model:
        #   list of arrays: [neg_class(n,feat), pos_class(n,feat)]
        #   3D array: (n, feat, classes)
        #   2D array: (n, feat)
        if isinstance(sv, list):
            values = sv[1][0]
        elif isinstance(sv, np.ndarray) and sv.ndim == 3:
            values = sv[0, :, 1]
        elif isinstance(sv, np.ndarray) and sv.ndim == 2:
            values = sv[0]
        else:
            values = np.asarray(sv).flatten()

        shap_dict = {name: float(val) for name, val in zip(feature_names, values, strict=False)}
        sorted_features = sorted(shap_dict.items(), key=lambda x: abs(x[1]), reverse=True)
        top = sorted_features[:top_n]

        bullish = [(n, v) for n, v in top if v > 0]
        bearish = [(n, v) for n, v in top if v < 0]

        parts: list[str] = []
        if bullish:
            items = ", ".join(f"{n} ({v:+.4f})" for n, v in bullish)
            parts.append(f"Bullish factors: {items}")
        if bearish:
            items = ", ".join(f"{n} ({v:+.4f})" for n, v in bearish)
            parts.append(f"Bearish factors: {items}")

        summary = ". ".join(parts) + "." if parts else "No dominant factors identified."

        log.info("shap_explanation_computed", n_features=len(feature_names), top_n=top_n)
        return ExplanationResult(shap_values=shap_dict, top_features=top, summary_text=summary)

    except Exception:
        log.exception("shap_explanation_failed")
        return ExplanationResult(summary_text="Explanation unavailable.")


def explain_ensemble(
    models: list,
    weights: list[float],
    features_row: pd.Series,
    feature_names: list[str],
    top_n: int = 10,
) -> ExplanationResult:
    """Compute weighted SHAP values across ensemble members."""
    try:
        combined: dict[str, float] = {name: 0.0 for name in feature_names}

        for model, weight in zip(models, weights, strict=False):
            result = explain_prediction(model, features_row, feature_names, top_n=len(feature_names))
            for name, val in result.shap_values.items():
                combined[name] += val * weight

        sorted_features = sorted(combined.items(), key=lambda x: abs(x[1]), reverse=True)
        top = sorted_features[:top_n]

        bullish = [(n, v) for n, v in top if v > 0]
        bearish = [(n, v) for n, v in top if v < 0]

        parts: list[str] = []
        if bullish:
            items = ", ".join(f"{n} ({v:+.4f})" for n, v in bullish)
            parts.append(f"Bullish factors: {items}")
        if bearish:
            items = ", ".join(f"{n} ({v:+.4f})" for n, v in bearish)
            parts.append(f"Bearish factors: {items}")

        summary = ". ".join(parts) + "." if parts else "No dominant factors identified."

        log.info("ensemble_explanation_computed", n_models=len(models))
        return ExplanationResult(shap_values=combined, top_features=top, summary_text=summary)

    except Exception:
        log.exception("ensemble_explanation_failed")
        return ExplanationResult(summary_text="Ensemble explanation unavailable.")
