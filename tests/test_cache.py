"""Tests for data.cache — uses tmp_path, no real market data."""

from __future__ import annotations

from datetime import UTC, datetime

import pandas as pd

from trading_copilot.config import DataConfig, PathsConfig, Settings
from trading_copilot.data.cache import OHLCVCache


def _make_settings(tmp_path) -> Settings:
    return Settings(
        paths=PathsConfig(
            data_dir=tmp_path / "data",
            cache_dir=tmp_path / "data" / "cache",
            log_dir=tmp_path / "data" / "logs",
            db_path=tmp_path / "data" / "db" / "test.sqlite",
        ),
        data=DataConfig(history_days=30),
    )


def _sample_df(n: int = 5, start_date: str = "2024-01-02") -> pd.DataFrame:
    idx = pd.date_range(start_date, periods=n, freq="B", tz="UTC")
    return pd.DataFrame(
        {
            "open": range(100, 100 + n),
            "high": range(105, 105 + n),
            "low": range(99, 99 + n),
            "close": range(104, 104 + n),
            "volume": [1000] * n,
        },
        index=idx,
    )


def test_save_and_load_roundtrip(tmp_path):
    cache = OHLCVCache(_make_settings(tmp_path))
    df = _sample_df()
    cache.save("TEST.NS", df)

    loaded = cache.load("TEST.NS")
    assert loaded is not None
    assert len(loaded) == len(df)
    assert list(loaded.columns) == list(df.columns)
    assert loaded.index.tz is not None


def test_load_returns_none_when_missing(tmp_path):
    cache = OHLCVCache(_make_settings(tmp_path))
    assert cache.load("NOPE") is None


def test_incremental_save_deduplicates(tmp_path):
    cache = OHLCVCache(_make_settings(tmp_path))
    df1 = _sample_df(3, "2024-01-02")
    df2 = _sample_df(3, "2024-01-04")  # overlaps by 2 days
    cache.save("OVERLAP.NS", df1)
    cache.save("OVERLAP.NS", df2)

    loaded = cache.load("OVERLAP.NS")
    assert loaded is not None
    assert loaded.index.is_unique


def test_get_missing_range_full_when_no_cache(tmp_path):
    cache = OHLCVCache(_make_settings(tmp_path))
    result = cache.get_missing_range("NEW.NS", history_days=30)
    assert result is not None
    start, end = result
    assert (end - start).days >= 29


def test_get_missing_range_none_when_current(tmp_path):
    cache = OHLCVCache(_make_settings(tmp_path))
    today = datetime.now(UTC).replace(hour=0, minute=0, second=0, microsecond=0)
    idx = pd.DatetimeIndex([today], tz="UTC")
    df = pd.DataFrame(
        {"open": [1], "high": [2], "low": [0], "close": [1], "volume": [100]}, index=idx
    )
    cache.save("CURRENT.NS", df)
    assert cache.get_missing_range("CURRENT.NS") is None


def test_list_tickers(tmp_path):
    cache = OHLCVCache(_make_settings(tmp_path))
    cache.save("AAA", _sample_df(2))
    cache.save("BBB", _sample_df(2))
    assert cache.list_tickers() == ["AAA", "BBB"]
