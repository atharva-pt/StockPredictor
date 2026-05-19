"""Tests for data.fetcher — mocked yfinance, no network."""

from __future__ import annotations

from datetime import UTC, datetime
from unittest.mock import MagicMock, patch

import pandas as pd

from trading_copilot.data.fetcher import _normalize, fetch_ohlcv


def _fake_df() -> pd.DataFrame:
    idx = pd.DatetimeIndex(
        [datetime(2024, 1, 2, tzinfo=UTC), datetime(2024, 1, 3, tzinfo=UTC)],
        name="Date",
    )
    return pd.DataFrame(
        {"Open": [100, 101], "High": [105, 106], "Low": [99, 100], "Close": [104, 105], "Volume": [1000, 1100]},
        index=idx,
    )


@patch("trading_copilot.data.fetcher.yf.Ticker")
def test_fetch_ohlcv_returns_normalized_df(mock_ticker_cls):
    mock_inst = MagicMock()
    mock_inst.history.return_value = _fake_df()
    mock_ticker_cls.return_value = mock_inst

    df = fetch_ohlcv("RELIANCE.NS", history_days=30)
    assert df is not None
    assert list(df.columns) == ["open", "high", "low", "close", "volume"]
    assert df.index.tz is not None
    assert len(df) == 2


@patch("trading_copilot.data.fetcher.yf.Ticker")
def test_fetch_ohlcv_returns_none_on_empty(mock_ticker_cls):
    mock_inst = MagicMock()
    mock_inst.history.return_value = pd.DataFrame()
    mock_ticker_cls.return_value = mock_inst

    result = fetch_ohlcv("INVALID", history_days=30)
    assert result is None


@patch("trading_copilot.data.fetcher.yf.Ticker")
def test_fetch_ohlcv_retries_on_exception(mock_ticker_cls):
    mock_inst = MagicMock()
    mock_inst.history.side_effect = [Exception("network"), Exception("timeout"), _fake_df()]
    mock_ticker_cls.return_value = mock_inst

    df = fetch_ohlcv("TCS.NS", history_days=30)
    assert df is not None
    assert len(df) == 2
    assert mock_inst.history.call_count == 3


def test_normalize_lowercases_and_utc():
    raw = _fake_df()
    raw.index = raw.index.tz_localize(None)  # simulate naive index
    df = _normalize(raw)
    assert "open" in df.columns
    assert df.index.tz is not None
