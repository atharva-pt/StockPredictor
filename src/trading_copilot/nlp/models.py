"""Sentiment analysis result schema."""

from __future__ import annotations

from pydantic import BaseModel, Field


class SentimentResult(BaseModel):
    text: str
    sentiment: str  # bullish | bearish | neutral
    score: float  # -1.0 (most bearish) to +1.0 (most bullish)
    confidence: float  # 0.0 to 1.0
    method: str  # finbert | vader | textblob
    events: list[str] = Field(default_factory=list)
    tickers: list[str] = Field(default_factory=list)


class AggregatedSentiment(BaseModel):
    """Aggregated sentiment across multiple articles for a ticker/period."""

    ticker: str
    avg_score: float
    bullish_count: int
    bearish_count: int
    neutral_count: int
    total_articles: int
    dominant_sentiment: str
    confidence: float
    news_impact_score: float  # weighted by recency and confidence
