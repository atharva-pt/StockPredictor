"""Signal generator — combines ML prediction with confirmation filters."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime

import pandas as pd

from trading_copilot.models.engine import Prediction
from trading_copilot.signals.rules import ConfirmationResult, compute_confirmation


@dataclass
class Signal:
    ticker: str
    timestamp: datetime
    action: str  # BUY | SELL | HOLD
    confidence: float
    risk_level: str  # LOW | MEDIUM | HIGH
    up_prob: float
    down_prob: float
    reasoning: str
    factors: list[str] = field(default_factory=list)
    model_name: str = ""


def generate_signal(
    ticker: str,
    prediction: Prediction,
    feature_row: pd.Series,
    min_confidence: float = 0.55,
    require_technical_alignment: bool = True,
    max_atr_pct: float = 0.05,
) -> Signal:
    """Generate a trading signal from ML prediction + confirmation rules.

    Signal logic:
    1. ML model provides directional probability
    2. Technical indicators confirm or contradict
    3. Sentiment aligns or conflicts
    4. Volatility filter adjusts risk level
    5. Final action = consensus of all factors
    """
    confirmation = compute_confirmation(feature_row, max_atr_pct)
    timestamp = datetime.now(UTC)

    # Start with ML direction
    ml_direction = prediction.direction
    ml_conf = prediction.confidence

    # Adjust confidence based on confirmation alignment
    adjusted_conf = ml_conf
    alignment_bonus = 0.0

    if ml_direction == "UP":
        if confirmation.technical_score > 0:
            alignment_bonus += 0.05
        elif confirmation.technical_score < -0.2:
            alignment_bonus -= 0.1
        if confirmation.sentiment_score > 0:
            alignment_bonus += 0.03
        elif confirmation.sentiment_score < -0.2:
            alignment_bonus -= 0.05

    elif ml_direction == "DOWN":
        if confirmation.technical_score < 0:
            alignment_bonus += 0.05
        elif confirmation.technical_score > 0.2:
            alignment_bonus -= 0.1
        if confirmation.sentiment_score < 0:
            alignment_bonus += 0.03
        elif confirmation.sentiment_score > 0.2:
            alignment_bonus -= 0.05

    adjusted_conf = max(0.0, min(1.0, adjusted_conf + alignment_bonus))

    # Volatility penalty
    if not confirmation.volatility_ok:
        adjusted_conf *= 0.85

    # Determine action
    action = _determine_action(
        ml_direction, adjusted_conf, confirmation,
        min_confidence, require_technical_alignment,
    )

    risk_level = _assess_risk(adjusted_conf, confirmation)
    reasoning = _build_reasoning(ml_direction, prediction, confirmation, action, adjusted_conf)

    return Signal(
        ticker=ticker,
        timestamp=timestamp,
        action=action,
        confidence=round(adjusted_conf, 4),
        risk_level=risk_level,
        up_prob=prediction.up_prob,
        down_prob=prediction.down_prob,
        reasoning=reasoning,
        factors=confirmation.factors,
        model_name=prediction.model_name,
    )


def generate_signals(
    ticker: str,
    predictions: list[Prediction],
    features: pd.DataFrame,
    **kwargs,
) -> list[Signal]:
    """Generate signals for multiple predictions aligned with feature rows."""
    signals: list[Signal] = []
    for pred, (_, row) in zip(predictions, features.iterrows(), strict=False):
        signals.append(generate_signal(ticker, pred, row, **kwargs))
    return signals


def _determine_action(
    ml_direction: str,
    confidence: float,
    confirmation: ConfirmationResult,
    min_confidence: float,
    require_technical: bool,
) -> str:
    if confidence < min_confidence:
        return "HOLD"

    if ml_direction == "HOLD":
        return "HOLD"

    if require_technical:
        if ml_direction == "UP" and confirmation.technical_score < -0.3:
            return "HOLD"
        if ml_direction == "DOWN" and confirmation.technical_score > 0.3:
            return "HOLD"

    if ml_direction == "UP":
        return "BUY"
    return "SELL"


def _assess_risk(confidence: float, confirmation: ConfirmationResult) -> str:
    if confidence > 0.7 and confirmation.volatility_ok:
        return "LOW"
    if confidence < 0.55 or not confirmation.volatility_ok:
        return "HIGH"
    return "MEDIUM"


def _build_reasoning(
    ml_direction: str,
    prediction: Prediction,
    confirmation: ConfirmationResult,
    action: str,
    adjusted_conf: float,
) -> str:
    parts: list[str] = []

    parts.append(
        f"ML model ({prediction.model_name}) predicts {ml_direction} "
        f"with {prediction.confidence:.0%} confidence (UP={prediction.up_prob:.1%}, DOWN={prediction.down_prob:.1%})."
    )

    tech = confirmation.technical_score
    if abs(tech) > 0.1:
        direction = "bullish" if tech > 0 else "bearish"
        parts.append(f"Technical indicators are {direction} (score: {tech:+.2f}).")

    sent = confirmation.sentiment_score
    if abs(sent) > 0.1:
        direction = "bullish" if sent > 0 else "bearish"
        parts.append(f"News sentiment is {direction} (score: {sent:+.2f}).")

    if not confirmation.volatility_ok:
        parts.append("Elevated volatility reduces signal confidence.")

    if action != ml_direction:
        parts.append(f"Signal adjusted to {action} due to confirmation filters.")

    parts.append(f"Final confidence: {adjusted_conf:.0%}.")
    return " ".join(parts)
