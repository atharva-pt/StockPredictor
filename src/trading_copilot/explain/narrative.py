"""Plain-English explanation generator for ML predictions."""

from __future__ import annotations

from trading_copilot.explain.shap_explain import ExplanationResult
from trading_copilot.models.engine import Prediction
from trading_copilot.signals.generator import Signal


def build_narrative(
    explanation: ExplanationResult,
    prediction: Prediction,
    signal: Signal,
) -> str:
    """Convert SHAP values + signal factors into a readable explanation.

    Example output:
        "The model predicts UP with 67% confidence. The strongest bullish factors
        are: RSI momentum (+0.12), positive sentiment shift (+0.08). Bearish
        pressure from: elevated volatility (-0.04)."
    """
    pct = f"{prediction.confidence:.0%}"
    parts: list[str] = [f"The model predicts {prediction.direction} with {pct} confidence."]

    bullish = [(n, v) for n, v in explanation.top_features if v > 0]
    bearish = [(n, v) for n, v in explanation.top_features if v < 0]

    if bullish:
        items = ", ".join(f"{n} ({v:+.2f})" for n, v in bullish[:5])
        parts.append(f"The strongest bullish factors are: {items}.")

    if bearish:
        items = ", ".join(f"{n} ({v:+.2f})" for n, v in bearish[:5])
        parts.append(f"Bearish pressure from: {items}.")

    if signal.factors:
        parts.append(f"Signal factors: {', '.join(signal.factors[:5])}.")

    if signal.action != prediction.direction:
        parts.append(
            f"Final signal adjusted to {signal.action} after confirmation filters."
        )

    return " ".join(parts)


def feature_importance_summary(
    explanation: ExplanationResult,
    top_n: int = 10,
) -> str:
    """Ranked list of most important features by absolute SHAP value."""
    if not explanation.shap_values:
        return "No feature importance data available."

    ranked = sorted(
        explanation.shap_values.items(),
        key=lambda x: abs(x[1]),
        reverse=True,
    )[:top_n]

    lines = [f"Top {len(ranked)} features by importance:"]
    for i, (name, val) in enumerate(ranked, 1):
        direction = "bullish" if val > 0 else "bearish"
        lines.append(f"  {i}. {name}: {val:+.4f} ({direction})")

    return "\n".join(lines)
