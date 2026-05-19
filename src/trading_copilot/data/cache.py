"""Parquet file cache for OHLCV + SQLite metadata index.

Storage layout:
    data/cache/{TICKER}.parquet   — one file per ticker, append-friendly
    data/db/copilot.sqlite        — metadata: last_date, row_count per ticker
"""

from __future__ import annotations

import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

from trading_copilot.config import Settings, get_settings
from trading_copilot.logging_setup import get_logger

log = get_logger("data.cache")

_CREATE_META = """
CREATE TABLE IF NOT EXISTS ticker_meta (
    ticker     TEXT PRIMARY KEY,
    last_date  TEXT NOT NULL,
    row_count  INTEGER NOT NULL,
    interval   TEXT NOT NULL DEFAULT '1d',
    updated_at TEXT NOT NULL
)
"""


class OHLCVCache:
    """Read/write OHLCV parquet files with a SQLite metadata sidecar."""

    def __init__(self, settings: Settings | None = None):
        self._settings = settings or get_settings()
        self._cache_dir = self._settings.paths.cache_dir
        self._db_path = self._settings.paths.db_path
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _init_db(self) -> None:
        with self._conn() as conn:
            conn.execute(_CREATE_META)

    def _conn(self) -> sqlite3.Connection:
        return sqlite3.connect(str(self._db_path))

    def _parquet_path(self, ticker: str) -> Path:
        safe = ticker.replace("/", "_").replace(".", "_")
        return self._cache_dir / f"{safe}.parquet"

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def load(self, ticker: str) -> pd.DataFrame | None:
        path = self._parquet_path(ticker)
        if not path.exists():
            return None
        df = pd.read_parquet(path)
        if df.index.tz is None:
            df.index = df.index.tz_localize("UTC")
        log.debug("cache_hit", ticker=ticker, rows=len(df))
        return df

    def save(self, ticker: str, df: pd.DataFrame) -> None:
        if df.empty:
            return

        existing = self.load(ticker)
        if existing is not None and not existing.empty:
            df = pd.concat([existing, df])
            df = df[~df.index.duplicated(keep="last")]
            df = df.sort_index()

        path = self._parquet_path(ticker)
        df.to_parquet(path, engine="pyarrow")

        last_date = df.index.max()
        now = datetime.now(timezone.utc)
        with self._conn() as conn:
            conn.execute(
                """
                INSERT INTO ticker_meta (ticker, last_date, row_count, interval, updated_at)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(ticker) DO UPDATE SET
                    last_date=excluded.last_date,
                    row_count=excluded.row_count,
                    updated_at=excluded.updated_at
                """,
                (ticker, last_date.isoformat(), len(df), "1d", now.isoformat()),
            )
        log.info("cache_saved", ticker=ticker, rows=len(df), path=str(path))

    def get_missing_range(
        self, ticker: str, history_days: int = 730
    ) -> tuple[datetime, datetime] | None:
        """Return (start, end) for the date range not yet cached, or None if up-to-date."""
        now = datetime.now(timezone.utc)
        today = now.replace(hour=0, minute=0, second=0, microsecond=0)

        with self._conn() as conn:
            row = conn.execute(
                "SELECT last_date FROM ticker_meta WHERE ticker = ?", (ticker,)
            ).fetchone()

        if row is None:
            start = now - timedelta(days=history_days)
            return (start, now)

        last = datetime.fromisoformat(row[0])
        if last.tzinfo is None:
            last = last.replace(tzinfo=timezone.utc)

        if last.date() >= today.date():
            return None

        return (last + timedelta(days=1), now)

    def list_tickers(self) -> list[str]:
        with self._conn() as conn:
            rows = conn.execute("SELECT ticker FROM ticker_meta ORDER BY ticker").fetchall()
        return [r[0] for r in rows]
