"""RSS feed definitions for each news source."""

from __future__ import annotations

FEEDS: dict[str, list[dict[str, str]]] = {
    "yahoo_finance": [
        {"url": "https://finance.yahoo.com/news/rssindex", "label": "Yahoo Finance Top"},
    ],
    "economic_times": [
        {"url": "https://economictimes.indiatimes.com/markets/rssfeeds/1977021501.cms", "label": "ET Markets"},
        {"url": "https://economictimes.indiatimes.com/markets/stocks/rssfeeds/2146842.cms", "label": "ET Stocks"},
    ],
    "moneycontrol": [
        {"url": "https://www.moneycontrol.com/rss/marketreports.xml", "label": "MC Market Reports"},
        {"url": "https://www.moneycontrol.com/rss/stocksinnews.xml", "label": "MC Stocks in News"},
    ],
    "reuters": [
        {"url": "https://www.reutersagency.com/feed/?taxonomy=best-sectors&post_type=best", "label": "Reuters Business"},
    ],
}


def get_all_feed_urls() -> list[dict[str, str]]:
    """Flat list of {url, label, source} for every configured feed."""
    result = []
    for source, feeds in FEEDS.items():
        for f in feeds:
            result.append({"url": f["url"], "label": f["label"], "source": source})
    return result
