"""SQLite storage for news articles. Handles dedup on insert, time-range queries."""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime

from trading_copilot.config import Settings, get_settings
from trading_copilot.logging_setup import get_logger
from trading_copilot.news.models import Article

log = get_logger("news.store")

_CREATE_TABLE = """
CREATE TABLE IF NOT EXISTS articles (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    title       TEXT NOT NULL,
    source      TEXT NOT NULL,
    url         TEXT UNIQUE NOT NULL,
    published_utc TEXT NOT NULL,
    summary     TEXT DEFAULT '',
    tickers     TEXT DEFAULT '[]',
    fetched_utc TEXT
)
"""

_CREATE_INDEX = """
CREATE INDEX IF NOT EXISTS idx_articles_published ON articles (published_utc)
"""


class NewsStore:
    def __init__(self, settings: Settings | None = None):
        self._settings = settings or get_settings()
        self._db_path = self._settings.paths.db_path
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _conn(self) -> sqlite3.Connection:
        return sqlite3.connect(str(self._db_path))

    def _init_db(self) -> None:
        with self._conn() as conn:
            conn.execute(_CREATE_TABLE)
            conn.execute(_CREATE_INDEX)

    def save_articles(self, articles: list[Article]) -> int:
        inserted = 0
        with self._conn() as conn:
            for a in articles:
                try:
                    conn.execute(
                        """INSERT OR IGNORE INTO articles
                           (title, source, url, published_utc, summary, tickers, fetched_utc)
                           VALUES (?, ?, ?, ?, ?, ?, ?)""",
                        (
                            a.title,
                            a.source,
                            a.url,
                            a.published_utc.isoformat(),
                            a.summary,
                            json.dumps(a.tickers),
                            a.fetched_utc.isoformat() if a.fetched_utc else None,
                        ),
                    )
                    if conn.total_changes:
                        inserted += 1
                except sqlite3.IntegrityError:
                    pass
        log.info("articles_saved", total=len(articles), inserted=inserted)
        return inserted

    def query(
        self,
        since: datetime | None = None,
        until: datetime | None = None,
        source: str | None = None,
        limit: int = 100,
    ) -> list[Article]:
        clauses: list[str] = []
        params: list[str | int] = []

        if since:
            clauses.append("published_utc >= ?")
            params.append(since.isoformat())
        if until:
            clauses.append("published_utc <= ?")
            params.append(until.isoformat())
        if source:
            clauses.append("source = ?")
            params.append(source)

        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        sql = f"SELECT title, source, url, published_utc, summary, tickers, fetched_utc FROM articles {where} ORDER BY published_utc DESC LIMIT ?"
        params.append(limit)

        with self._conn() as conn:
            rows = conn.execute(sql, params).fetchall()

        return [
            Article(
                title=r[0],
                source=r[1],
                url=r[2],
                published_utc=r[3],
                summary=r[4],
                tickers=json.loads(r[5]) if r[5] else [],
                fetched_utc=r[6],
            )
            for r in rows
        ]

    def count(self) -> int:
        with self._conn() as conn:
            return conn.execute("SELECT COUNT(*) FROM articles").fetchone()[0]
