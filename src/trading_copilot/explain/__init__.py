"""AI explainability layer — SHAP-based explanations and plain-English narratives."""

from trading_copilot.explain.narrative import build_narrative, feature_importance_summary
from trading_copilot.explain.plots import feature_importance_chart, shap_waterfall_chart
from trading_copilot.explain.shap_explain import (
    ExplanationResult,
    explain_ensemble,
    explain_prediction,
)

__all__ = [
    "ExplanationResult",
    "build_narrative",
    "explain_ensemble",
    "explain_prediction",
    "feature_importance_chart",
    "feature_importance_summary",
    "shap_waterfall_chart",
]
