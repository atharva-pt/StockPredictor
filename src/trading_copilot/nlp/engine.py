"""Sentiment engine — orchestrates FinBERT → VADER → TextBlob fallback chain.

Also handles ticker extraction, event classification, and aggregation.
"""

from __future__ import annotations

import math
from datetime import UTC, datetime

from trading_copilot.logging_setup import get_logger
from trading_copilot.news.models import Article
from trading_copilot.nlp import events as evt
from trading_copilot.nlp import ticker_extract as te
from trading_copilot.nlp.models import AggregatedSentiment, SentimentResult

log = get_logger("nlp.engine")


def analyze_text(text: str, use_finbert: bool = True) -> SentimentResult:
    """Analyze a single text through the fallback chain.

    Chain: FinBERT (best quality) → VADER (fast, rule-based) → TextBlob (pattern-based).
    Each uses a fundamentally different algorithm for independence.
    """
    result: SentimentResult | None = None

    if use_finbert:
        from trading_copilot.nlp import finbert
        result = finbert.analyze(text)

    if result is None:
        from trading_copilot.nlp import vader
        result = vader.analyze(text)
        if result.confidence < 0.1:
            from trading_copilot.nlp import textblob_analyzer
            tb_result = textblob_analyzer.analyze(text)
            if tb_result.confidence > result.confidence:
                result = tb_result

    result.tickers = te.extract_tickers(text)
    result.events = evt.classify_events(text)
    return result


def analyze_article(article: Article, use_finbert: bool = True) -> SentimentResult:
    """Analyze a news Article — combines title + summary for better signal."""
    combined = f"{article.title}. {article.summary}".strip()
    result = analyze_text(combined, use_finbert=use_finbert)
    if not result.tickers:
        result.tickers = article.tickers
    return result


def analyze_articles(articles: list[Article], use_finbert: bool = True) -> list[SentimentResult]:
    """Batch analyze articles. Uses FinBERT batch mode when available."""
    if not articles:
        return []

    if use_finbert:
        try:
            from trading_copilot.nlp import finbert
            texts = [f"{a.title}. {a.summary}".strip() for a in articles]
            results = finbert.analyze_batch(texts)

            output: list[SentimentResult] = []
            for article, text, fb_result in zip(articles, texts, results, strict=False):
                if fb_result is not None:
                    fb_result.tickers = te.extract_tickers(text) or article.tickers
                    fb_result.events = evt.classify_events(text)
                    output.append(fb_result)
                else:
                    output.append(analyze_article(article, use_finbert=False))
            return output
        except Exception as exc:
            log.warning("finbert_batch_unavailable", error=str(exc))

    return [analyze_article(a, use_finbert=False) for a in articles]


def aggregate_sentiment(
    results: list[SentimentResult],
    ticker: str,
    article_times: list[datetime] | None = None,
) -> AggregatedSentiment:
    """Aggregate sentiment results for a ticker with recency weighting."""
    relevant = [r for r in results if not r.tickers or ticker in r.tickers]

    if not relevant:
        return AggregatedSentiment(
            ticker=ticker, avg_score=0.0, bullish_count=0, bearish_count=0,
            neutral_count=0, total_articles=0, dominant_sentiment="neutral",
            confidence=0.0, news_impact_score=0.0,
        )

    bullish = sum(1 for r in relevant if r.sentiment == "bullish")
    bearish = sum(1 for r in relevant if r.sentiment == "bearish")
    neutral = sum(1 for r in relevant if r.sentiment == "neutral")

    avg_score = sum(r.score for r in relevant) / len(relevant)
    avg_conf = sum(r.confidence for r in relevant) / len(relevant)

    if bullish > bearish and bullish > neutral:
        dominant = "bullish"
    elif bearish > bullish and bearish > neutral:
        dominant = "bearish"
    else:
        dominant = "neutral"

    # News impact: confidence-weighted, with exponential recency decay
    now = datetime.now(UTC)
    impact = 0.0
    for i, r in enumerate(relevant):
        weight = r.confidence
        if article_times and i < len(article_times):
            hours_ago = (now - article_times[i]).total_seconds() / 3600
            weight *= math.exp(-0.05 * hours_ago)  # half-life ~14 hours
        impact += abs(r.score) * weight
    impact = min(impact / max(len(relevant), 1), 1.0)

    return AggregatedSentiment(
        ticker=ticker,
        avg_score=round(avg_score, 4),
        bullish_count=bullish,
        bearish_count=bearish,
        neutral_count=neutral,
        total_articles=len(relevant),
        dominant_sentiment=dominant,
        confidence=round(avg_conf, 4),
        news_impact_score=round(impact, 4),
    )
