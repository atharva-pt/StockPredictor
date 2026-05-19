"""Typed schemas for market data."""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel

OHLCV_COLUMNS = ["open", "high", "low", "close", "volume"]


class TickerMeta(BaseModel):
    """Row in the SQLite metadata table — tracks cache freshness per ticker."""

    ticker: str
    last_date: datetime
    row_count: int
    interval: str = "1d"
    updated_at: datetime
