#!/usr/bin/env python3
"""Backfill OHLCV history for one or all watchlist tickers.

Usage:
    python scripts/fetch_history.py                       # all watchlist tickers
    python scripts/fetch_history.py --ticker RELIANCE.NS  # single ticker
"""

from __future__ import annotations

import argparse

from trading_copilot.config import get_settings
from trading_copilot.data.cache import OHLCVCache
from trading_copilot.data.fetcher import fetch_ohlcv
from trading_copilot.logging_setup import configure_logging


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Fetch & cache OHLCV history")
    parser.add_argument("--ticker", type=str, default=None, help="Single ticker to fetch")
    args = parser.parse_args(argv)

    settings = get_settings()
    log = configure_logging(settings)
    cache = OHLCVCache(settings)

    if args.ticker:
        tickers = [args.ticker]
    else:
        tickers = [t for lst in settings.markets.watchlist.values() for t in lst]

    if not tickers:
        log.warning("no_tickers", msg="Watchlist is empty and no --ticker provided")
        return

    log.info("fetch_start", tickers=tickers, history_days=settings.data.history_days)

    for ticker in tickers:
        missing = cache.get_missing_range(ticker, history_days=settings.data.history_days)
        if missing is None:
            log.info("up_to_date", ticker=ticker)
            continue

        start, end = missing
        log.info("fetching", ticker=ticker, start=str(start.date()), end=str(end.date()))
        df = fetch_ohlcv(ticker, start=start, end=end, interval=settings.data.default_interval)

        if df is not None and not df.empty:
            cache.save(ticker, df)
        else:
            log.warning("skip_empty", ticker=ticker)

    log.info("fetch_complete", total=len(tickers), cached=len(cache.list_tickers()))


if __name__ == "__main__":
    main()
