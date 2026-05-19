"""Tests for NLP sentiment engine — VADER/TextBlob always available, FinBERT mocked."""

from __future__ import annotations

from datetime import UTC, datetime
from unittest.mock import patch

from trading_copilot.news.models import Article
from trading_copilot.nlp.engine import aggregate_sentiment, analyze_article, analyze_text
from trading_copilot.nlp.events import classify_events
from trading_copilot.nlp.models import SentimentResult
from trading_copilot.nlp.textblob_analyzer import analyze as tb_analyze
from trading_copilot.nlp.ticker_extract import extract_tickers
from trading_copilot.nlp.vader import analyze as vader_analyze

# --- VADER tests ---

class TestVader:
    def test_bullish_text(self):
        r = vader_analyze("Stock surges 10% on great earnings, investors celebrate")
        assert r.sentiment == "bullish"
        assert r.score > 0
        assert r.method == "vader"

    def test_bearish_text(self):
        r = vader_analyze("Market crashes, massive losses, investors panic")
        assert r.sentiment == "bearish"
        assert r.score < 0

    def test_neutral_text(self):
        r = vader_analyze("The company held its annual meeting today")
        assert r.sentiment == "neutral"


# --- TextBlob tests ---

class TestTextBlob:
    def test_bullish_text(self):
        r = tb_analyze("Excellent results, strong growth, wonderful performance")
        assert r.sentiment == "bullish"
        assert r.method == "textblob"

    def test_bearish_text(self):
        r = tb_analyze("Terrible losses, horrible decline, worst quarter ever")
        assert r.sentiment == "bearish"

    def test_returns_confidence(self):
        r = tb_analyze("Amazing incredible wonderful fantastic")
        assert r.confidence > 0


# --- Ticker extraction ---

class TestTickerExtract:
    def test_company_name_lookup(self):
        tickers = extract_tickers("Reliance Industries reported strong Q3 results")
        assert "RELIANCE.NS" in tickers

    def test_explicit_ticker_pattern(self):
        tickers = extract_tickers("Check out $AAPL and INFY.NS today")
        assert "AAPL" in tickers
        assert "INFY.NS" in tickers

    def test_nse_prefix(self):
        tickers = extract_tickers("NSE:HDFCBANK hit 52-week high")
        assert "HDFCBANK.NS" in tickers

    def test_multiple_companies(self):
        tickers = extract_tickers("TCS and Infosys both reported earnings today")
        assert "TCS.NS" in tickers
        assert "INFY.NS" in tickers

    def test_no_tickers(self):
        assert extract_tickers("The weather is nice today") == []


# --- Event classification ---

class TestEvents:
    def test_earnings(self):
        events = classify_events("Q3 results show 20% profit growth")
        assert "earnings" in events

    def test_merger(self):
        events = classify_events("Company announces acquisition of rival firm")
        assert "merger_acquisition" in events

    def test_regulatory(self):
        events = classify_events("SEBI issues new regulation for mutual funds")
        assert "regulatory" in events

    def test_macro(self):
        events = classify_events("RBI holds repo rate steady amid inflation concerns")
        assert "macro" in events

    def test_multiple_events(self):
        events = classify_events("RBI regulation impacts banking sector earnings")
        assert len(events) >= 2


# --- Engine fallback chain ---

class TestEngine:
    def test_analyze_text_without_finbert(self):
        r = analyze_text("Stock surges on strong earnings", use_finbert=False)
        assert r.sentiment in ("bullish", "bearish", "neutral")
        assert r.method in ("vader", "textblob")

    def test_analyze_text_extracts_tickers(self):
        r = analyze_text("Reliance shares jump 5% on Q3 results", use_finbert=False)
        assert "RELIANCE.NS" in r.tickers

    def test_analyze_text_classifies_events(self):
        r = analyze_text("SEBI penalty on company for insider trading", use_finbert=False)
        assert "regulatory" in r.events

    def test_finbert_fallback_to_vader(self):
        with patch("trading_copilot.nlp.finbert.analyze", return_value=None):
            r = analyze_text("Great earnings report", use_finbert=True)
            assert r.method in ("vader", "textblob")

    def test_analyze_article(self):
        article = Article(
            title="TCS Q3 profit rises 15%",
            source="test",
            url="https://x.com/1",
            published_utc=datetime.now(UTC),
            summary="Tata Consultancy Services reported strong quarterly results.",
        )
        r = analyze_article(article, use_finbert=False)
        assert "TCS.NS" in r.tickers
        assert r.sentiment in ("bullish", "bearish", "neutral")


# --- Aggregation ---

class TestAggregation:
    def test_aggregate_basic(self):
        results = [
            SentimentResult(text="a", sentiment="bullish", score=0.8, confidence=0.9, method="vader"),
            SentimentResult(text="b", sentiment="bullish", score=0.6, confidence=0.7, method="vader"),
            SentimentResult(text="c", sentiment="bearish", score=-0.5, confidence=0.6, method="vader"),
        ]
        agg = aggregate_sentiment(results, ticker="TEST")
        assert agg.bullish_count == 2
        assert agg.bearish_count == 1
        assert agg.dominant_sentiment == "bullish"
        assert agg.total_articles == 3
        assert agg.avg_score > 0

    def test_aggregate_empty(self):
        agg = aggregate_sentiment([], ticker="EMPTY")
        assert agg.total_articles == 0
        assert agg.dominant_sentiment == "neutral"

    def test_aggregate_filters_by_ticker(self):
        results = [
            SentimentResult(text="a", sentiment="bullish", score=0.8, confidence=0.9, method="vader", tickers=["AAPL"]),
            SentimentResult(text="b", sentiment="bearish", score=-0.5, confidence=0.6, method="vader", tickers=["MSFT"]),
        ]
        agg = aggregate_sentiment(results, ticker="AAPL")
        assert agg.total_articles == 1
        assert agg.dominant_sentiment == "bullish"
