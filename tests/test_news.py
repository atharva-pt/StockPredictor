"""Tests for news module — mocked HTTP, no network calls."""

from __future__ import annotations

from datetime import UTC, datetime
from unittest.mock import MagicMock, patch

from trading_copilot.config import PathsConfig, Settings
from trading_copilot.news.models import Article
from trading_copilot.news.scraper import _deduplicate, _extract_summary, _parse_pub_date, fetch_feed
from trading_copilot.news.store import NewsStore

SAMPLE_RSS = """<?xml version="1.0" encoding="UTF-8"?>
<rss version="2.0">
<channel>
  <title>Test Feed</title>
  <item>
    <title>Stock XYZ surges 5%</title>
    <link>https://example.com/article1</link>
    <pubDate>Mon, 19 May 2025 10:00:00 GMT</pubDate>
    <description>&lt;p&gt;Big move in markets today.&lt;/p&gt;</description>
  </item>
  <item>
    <title>Market outlook for Q3</title>
    <link>https://example.com/article2</link>
    <pubDate>Mon, 19 May 2025 09:00:00 GMT</pubDate>
    <description>Analysts predict moderate growth.</description>
  </item>
</channel>
</rss>"""


def _make_settings(tmp_path) -> Settings:
    return Settings(
        paths=PathsConfig(
            data_dir=tmp_path / "data",
            cache_dir=tmp_path / "data" / "cache",
            log_dir=tmp_path / "data" / "logs",
            db_path=tmp_path / "data" / "db" / "test.sqlite",
        ),
    )


# --- Scraper tests ---

@patch("trading_copilot.news.scraper.requests.get")
def test_fetch_feed_parses_rss(mock_get):
    mock_resp = MagicMock()
    mock_resp.text = SAMPLE_RSS
    mock_resp.raise_for_status = MagicMock()
    mock_get.return_value = mock_resp

    articles = fetch_feed("https://example.com/rss", "test_source", "Test")
    assert len(articles) == 2
    assert articles[0].title == "Stock XYZ surges 5%"
    assert articles[0].source == "test_source"
    assert articles[0].published_utc.tzinfo is not None


@patch("trading_copilot.news.scraper.requests.get")
def test_fetch_feed_handles_failure(mock_get):
    mock_get.side_effect = Exception("network error")
    articles = fetch_feed("https://bad.url/rss", "bad", "Bad")
    assert articles == []


def test_parse_pub_date_rfc2822():
    entry = {"published": "Mon, 19 May 2025 10:00:00 +0530"}
    dt = _parse_pub_date(entry)
    assert dt is not None
    assert dt.tzinfo == UTC
    assert dt.hour == 4  # 10:00 IST = 04:30 UTC... actually 4:30


def test_parse_pub_date_returns_none_on_garbage():
    assert _parse_pub_date({}) is None
    assert _parse_pub_date({"published": ""}) is None


def test_extract_summary_strips_html():
    entry = {"summary": "<p>Hello <b>world</b></p>"}
    assert _extract_summary(entry) == "Hello world"


def test_deduplicate_by_url():
    a1 = Article(title="A", source="s", url="https://x.com/1", published_utc=datetime.now(UTC))
    a2 = Article(title="B", source="s", url="https://x.com/1", published_utc=datetime.now(UTC))
    a3 = Article(title="C", source="s", url="https://x.com/2", published_utc=datetime.now(UTC))
    result = _deduplicate([a1, a2, a3])
    assert len(result) == 2


# --- Store tests ---

def test_store_save_and_query(tmp_path):
    store = NewsStore(_make_settings(tmp_path))
    now = datetime.now(UTC)
    articles = [
        Article(title="A1", source="s1", url="https://x.com/1", published_utc=now, summary="sum1"),
        Article(title="A2", source="s2", url="https://x.com/2", published_utc=now, summary="sum2"),
    ]
    store.save_articles(articles)
    assert store.count() >= 2

    results = store.query(limit=10)
    assert len(results) >= 2
    assert results[0].title in ("A1", "A2")


def test_store_dedup_on_url(tmp_path):
    store = NewsStore(_make_settings(tmp_path))
    now = datetime.now(UTC)
    a = Article(title="Dup", source="s", url="https://x.com/dup", published_utc=now)
    store.save_articles([a])
    store.save_articles([a])
    assert store.count() == 1


def test_store_query_by_source(tmp_path):
    store = NewsStore(_make_settings(tmp_path))
    now = datetime.now(UTC)
    articles = [
        Article(title="A", source="alpha", url="https://x.com/a", published_utc=now),
        Article(title="B", source="beta", url="https://x.com/b", published_utc=now),
    ]
    store.save_articles(articles)
    results = store.query(source="alpha", limit=10)
    assert all(r.source == "alpha" for r in results)
