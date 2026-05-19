"""VADER sentiment analyzer — first fallback.

Rule-based, no model download needed. Fast but less accurate on financial text.
"""

from __future__ import annotations

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

from trading_copilot.nlp.models import SentimentResult

_analyzer: SentimentIntensityAnalyzer | None = None


def _get_analyzer() -> SentimentIntensityAnalyzer:
    global _analyzer
    if _analyzer is None:
        _analyzer = SentimentIntensityAnalyzer()
    return _analyzer


def analyze(text: str) -> SentimentResult:
    """Analyze text with VADER. Always succeeds (rule-based)."""
    scores = _get_analyzer().polarity_scores(text)
    compound = scores["compound"]

    if compound >= 0.05:
        sentiment = "bullish"
    elif compound <= -0.05:
        sentiment = "bearish"
    else:
        sentiment = "neutral"

    confidence = abs(compound)

    return SentimentResult(
        text=text[:200],
        sentiment=sentiment,
        score=compound,
        confidence=min(confidence, 1.0),
        method="vader",
    )
