"""TextBlob sentiment analyzer — second fallback.

Pattern-based (no NLTK downloads needed). Different algorithm from VADER,
so it provides an independent second opinion when FinBERT is unavailable.
"""

from __future__ import annotations

from textblob import TextBlob

from trading_copilot.nlp.models import SentimentResult


def analyze(text: str) -> SentimentResult:
    """Analyze text with TextBlob. Always succeeds (pattern-based)."""
    blob = TextBlob(text)
    polarity = blob.sentiment.polarity  # -1 to +1
    subjectivity = blob.sentiment.subjectivity  # 0 to 1

    if polarity > 0.1:
        sentiment = "bullish"
    elif polarity < -0.1:
        sentiment = "bearish"
    else:
        sentiment = "neutral"

    # Confidence: blend of polarity strength and subjectivity
    # High subjectivity + strong polarity = more opinionated = higher confidence
    confidence = min(abs(polarity) * (0.5 + 0.5 * subjectivity), 1.0)

    return SentimentResult(
        text=text[:200],
        sentiment=sentiment,
        score=polarity,
        confidence=confidence,
        method="textblob",
    )
