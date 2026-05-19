"""RSS feed scraper — fetches articles, normalizes timestamps to UTC, deduplicates."""

from __future__ import annotations

import time
from datetime import UTC, datetime
from email.utils import parsedate_to_datetime

import feedparser
import requests
from bs4 import BeautifulSoup

from trading_copilot.logging_setup import get_logger
from trading_copilot.news.models import Article
from trading_copilot.news.sources import get_all_feed_urls

log = get_logger("news.scraper")

REQUEST_TIMEOUT = 15
RETRY_WAIT = 2.0


def fetch_feed(url: str, source: str, label: str) -> list[Article]:
    """Parse a single RSS feed URL into Article objects."""
    try:
        resp = requests.get(url, timeout=REQUEST_TIMEOUT, headers={"User-Agent": "TradingCopilot/0.1"})
        resp.raise_for_status()
    except Exception as exc:
        log.warning("feed_fetch_failed", url=url, error=str(exc))
        return []

    feed = feedparser.parse(resp.text)
    articles: list[Article] = []
    now = datetime.now(UTC)

    for entry in feed.entries:
        pub = _parse_pub_date(entry)
        if pub is None:
            continue

        summary = _extract_summary(entry)
        articles.append(
            Article(
                title=entry.get("title", "").strip(),
                source=source,
                url=entry.get("link", ""),
                published_utc=pub,
                summary=summary,
                fetched_utc=now,
            )
        )

    log.info("feed_parsed", source=source, label=label, articles=len(articles))
    return articles


def fetch_all_feeds() -> list[Article]:
    """Fetch every configured feed. Returns deduplicated articles sorted by publish time."""
    all_articles: list[Article] = []
    feeds = get_all_feed_urls()

    for f in feeds:
        arts = fetch_feed(f["url"], f["source"], f["label"])
        all_articles.extend(arts)
        time.sleep(0.5)

    deduped = _deduplicate(all_articles)
    deduped.sort(key=lambda a: a.published_utc)
    log.info("all_feeds_done", total=len(deduped))
    return deduped


def _parse_pub_date(entry: dict) -> datetime | None:
    """Best-effort extraction of published datetime in UTC."""
    for field in ("published", "updated", "created"):
        raw = entry.get(field)
        if not raw:
            continue
        try:
            dt = parsedate_to_datetime(raw)
            dt = dt.replace(tzinfo=UTC) if dt.tzinfo is None else dt.astimezone(UTC)
            return dt
        except Exception:
            pass

    parsed = entry.get("published_parsed") or entry.get("updated_parsed")
    if parsed:
        try:
            ts = time.mktime(parsed)
            return datetime.fromtimestamp(ts, tz=UTC)
        except Exception:
            pass

    return None


def _extract_summary(entry: dict) -> str:
    raw = entry.get("summary", "") or entry.get("description", "")
    if not raw:
        return ""
    text = BeautifulSoup(raw, "html.parser").get_text(separator=" ", strip=True)
    return text[:500]


def _deduplicate(articles: list[Article]) -> list[Article]:
    seen: set[str] = set()
    result: list[Article] = []
    for a in articles:
        key = a.url or a.title
        if key not in seen:
            seen.add(key)
            result.append(a)
    return result
