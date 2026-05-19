"""Tests for dashboard chart builders."""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from trading_copilot.dashboard.charts import (
    candlestick_chart,
    equity_curve_chart,
    prediction_gauge,
    sentiment_timeline,
)


def _ohlcv(n: int = 50) -> pd.DataFrame:
    np.random.seed(42)
    close = 100 * np.exp(np.cumsum(np.random.randn(n) * 0.01))
    idx = pd.date_range("2024-01-01", periods=n, freq="B", tz="UTC")
    return pd.DataFrame({
        "open": close * (1 + np.random.randn(n) * 0.002),
        "high": close * (1 + np.abs(np.random.randn(n)) * 0.005),
        "low": close * (1 - np.abs(np.random.randn(n)) * 0.005),
        "close": close,
        "volume": np.random.randint(100_000, 1_000_000, size=n).astype(float),
    }, index=idx)


class TestCandlestickChart:
    def test_returns_figure(self):
        fig = candlestick_chart(_ohlcv())
        assert isinstance(fig, go.Figure)

    def test_with_indicators(self):
        ohlcv = _ohlcv()
        indicators = pd.DataFrame({
            "ema_20": ohlcv["close"].ewm(span=20).mean(),
            "rsi": np.random.uniform(30, 70, len(ohlcv)),
            "bb_upper": ohlcv["close"] * 1.02,
            "bb_lower": ohlcv["close"] * 0.98,
        }, index=ohlcv.index)
        fig = candlestick_chart(ohlcv, indicators)
        assert isinstance(fig, go.Figure)
        assert len(fig.data) > 1


class TestEquityCurve:
    def test_returns_figure(self):
        equity = list(np.cumsum(np.random.randn(50)) + 100_000)
        dates = list(pd.date_range("2024-01-01", periods=50, freq="B"))
        fig = equity_curve_chart(equity, dates)
        assert isinstance(fig, go.Figure)


class TestSentimentTimeline:
    def test_with_data(self):
        data = [
            {"published_utc": "2024-01-01T10:00", "score": 0.5, "title": "Good news"},
            {"published_utc": "2024-01-02T10:00", "score": -0.3, "title": "Bad news"},
        ]
        fig = sentiment_timeline(data)
        assert isinstance(fig, go.Figure)
        assert len(fig.data) == 1

    def test_empty(self):
        fig = sentiment_timeline([])
        assert isinstance(fig, go.Figure)


class TestPredictionGauge:
    def test_buy_signal(self):
        fig = prediction_gauge(0.72, 0.28, "BUY", 0.65)
        assert isinstance(fig, go.Figure)

    def test_sell_signal(self):
        fig = prediction_gauge(0.35, 0.65, "SELL", 0.60)
        assert isinstance(fig, go.Figure)
