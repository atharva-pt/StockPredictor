"""News article schema."""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, Field


class Article(BaseModel):
    title: str
    source: str
    url: str
    published_utc: datetime
    summary: str = ""
    tickers: list[str] = Field(default_factory=list)
    fetched_utc: datetime | None = None
