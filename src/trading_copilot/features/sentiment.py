"""Sentiment features for ML — aggregated from news articles aligned to market bars."""

from __future__ import annotations

from datetime import timedelta

import numpy as np
import pandas as pd

from trading_copilot.news.models import Article
from trading_copilot.nlp.engine import analyze_articles
from trading_copilot.nlp.models import SentimentResult


def build_sentiment_features(
    market_index: pd.DatetimeIndex,
    articles: list[Article],
    ticker: str,
    use_finbert: bool = False,
) -> pd.DataFrame:
    """Build sentiment features aligned to market bars.

    For each bar at time t, we aggregate sentiment from articles published
    BEFORE t (specifically, in the 24h window ending at t). This prevents
    future leakage — we never use articles published after the bar.
    """
    out = pd.DataFrame(index=market_index)
    out["sent_score"] = 0.0
    out["sent_confidence"] = 0.0
    out["sent_bullish_ratio"] = 0.5
    out["sent_article_count"] = 0
    out["sent_impact"] = 0.0

    if not articles:
        return out

    results = analyze_articles(articles, use_finbert=use_finbert)

    # Build (published_utc, result) pairs, filter to ticker
    pairs: list[tuple[pd.Timestamp, SentimentResult]] = []
    for article, result in zip(articles, results, strict=False):
        if result.tickers and ticker not in result.tickers:
            continue
        pairs.append((pd.Timestamp(article.published_utc), result))

    if not pairs:
        return out

    pairs.sort(key=lambda p: p[0])

    for i, t in enumerate(market_index):
        # Window: articles from (t - 24h) to t (exclusive of t to be safe)
        window_start = t - timedelta(hours=24)
        window = [r for pub, r in pairs if window_start <= pub < t]

        if not window:
            continue

        scores = [r.score for r in window]
        confs = [r.confidence for r in window]
        bullish = sum(1 for r in window if r.sentiment == "bullish")

        out.iloc[i, out.columns.get_loc("sent_score")] = float(np.mean(scores))
        out.iloc[i, out.columns.get_loc("sent_confidence")] = float(np.mean(confs))
        out.iloc[i, out.columns.get_loc("sent_bullish_ratio")] = bullish / len(window)
        out.iloc[i, out.columns.get_loc("sent_article_count")] = len(window)
        out.iloc[i, out.columns.get_loc("sent_impact")] = float(
            np.mean([abs(s) * c for s, c in zip(scores, confs, strict=False)])
        )

    # Rolling aggregates for trend detection
    out["sent_score_ma3"] = out["sent_score"].rolling(3, min_periods=1).mean()
    out["sent_score_ma7"] = out["sent_score"].rolling(7, min_periods=1).mean()
    out["sent_shift"] = out["sent_score"] - out["sent_score_ma7"]

    return out
