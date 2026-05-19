#!/usr/bin/env python3
"""Fetch latest news from all RSS feeds and store in SQLite.

Usage:
    python scripts/fetch_news.py
"""

from __future__ import annotations

from trading_copilot.config import get_settings
from trading_copilot.logging_setup import configure_logging
from trading_copilot.news.scraper import fetch_all_feeds
from trading_copilot.news.store import NewsStore


def main() -> None:
    settings = get_settings()
    log = configure_logging(settings)
    store = NewsStore(settings)

    articles = fetch_all_feeds()
    inserted = store.save_articles(articles)
    total = store.count()
    log.info("news_fetch_complete", fetched=len(articles), new=inserted, total_in_db=total)


if __name__ == "__main__":
    main()
