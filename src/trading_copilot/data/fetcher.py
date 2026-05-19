"""yfinance wrapper with retry, validation, and UTC normalization."""

from __future__ import annotations

import time
from datetime import datetime, timedelta, timezone

import pandas as pd
import yfinance as yf

from trading_copilot.logging_setup import get_logger

log = get_logger("data.fetcher")

MAX_RETRIES = 3
BACKOFF_BASE = 2.0


def fetch_ohlcv(
    ticker: str,
    start: datetime | str | None = None,
    end: datetime | str | None = None,
    interval: str = "1d",
    history_days: int = 730,
) -> pd.DataFrame | None:
    """Fetch OHLCV from yfinance. Returns DataFrame with UTC DatetimeIndex or None on failure.

    Columns: open, high, low, close, volume (lowercase).
    """
    if start is None:
        start = datetime.now(timezone.utc) - timedelta(days=history_days)
    if end is None:
        end = datetime.now(timezone.utc)

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            t = yf.Ticker(ticker)
            df = t.history(start=start, end=end, interval=interval, auto_adjust=True)

            if df is None or df.empty:
                log.warning("empty_response", ticker=ticker, attempt=attempt)
                return None

            df = _normalize(df)
            log.info("fetch_ok", ticker=ticker, rows=len(df), interval=interval)
            return df

        except Exception as exc:
            wait = BACKOFF_BASE**attempt
            log.warning(
                "fetch_retry",
                ticker=ticker,
                attempt=attempt,
                error=str(exc),
                wait_s=wait,
            )
            if attempt < MAX_RETRIES:
                time.sleep(wait)

    log.error("fetch_failed", ticker=ticker, retries=MAX_RETRIES)
    return None


def validate_ticker(ticker: str) -> bool:
    """Quick check: does yfinance return any info for this ticker?"""
    try:
        info = yf.Ticker(ticker).info
        return bool(info and info.get("regularMarketPrice"))
    except Exception:
        return False


def _normalize(df: pd.DataFrame) -> pd.DataFrame:
    """Lowercase columns, ensure UTC index, drop yfinance extras."""
    df = df.copy()
    df.columns = [c.lower().replace(" ", "_") for c in df.columns]

    keep = [c for c in ["open", "high", "low", "close", "volume"] if c in df.columns]
    df = df[keep]

    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    else:
        df.index = df.index.tz_convert("UTC")

    df.index.name = "datetime"
    return df
