"""Tests for indicators — uses synthetic OHLCV, no market data needed."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from trading_copilot.indicators.core import (
    atr,
    bollinger_bands,
    ema,
    macd,
    rsi,
    sma,
    stochastic_rsi,
    volume_ratio,
    vwap,
)
from trading_copilot.indicators.engine import compute_all
from trading_copilot.indicators.signals import (
    crossover,
    crossunder,
    ema_crossovers,
    overbought_oversold,
    support_resistance,
    trend_strength,
    volume_spike,
)


def _synthetic_ohlcv(n: int = 300) -> pd.DataFrame:
    np.random.seed(42)
    close = 100 + np.cumsum(np.random.randn(n) * 0.5)
    high = close + np.abs(np.random.randn(n)) * 0.5
    low = close - np.abs(np.random.randn(n)) * 0.5
    opn = close + np.random.randn(n) * 0.2
    vol = np.random.randint(100_000, 1_000_000, size=n).astype(float)
    idx = pd.date_range("2023-01-01", periods=n, freq="B", tz="UTC")
    return pd.DataFrame(
        {"open": opn, "high": high, "low": low, "close": close, "volume": vol},
        index=idx,
    )


@pytest.fixture
def ohlcv():
    return _synthetic_ohlcv()


# --- Core indicators ---

class TestCore:
    def test_sma_length(self, ohlcv):
        s = sma(ohlcv["close"], 20)
        assert len(s) == len(ohlcv)
        assert s.iloc[:19].isna().all()
        assert s.iloc[19:].notna().all()

    def test_ema_length(self, ohlcv):
        e = ema(ohlcv["close"], 20)
        assert e.iloc[19:].notna().all()

    def test_rsi_range(self, ohlcv):
        r = rsi(ohlcv["close"])
        valid = r.dropna()
        assert (valid >= 0).all() and (valid <= 100).all()

    def test_macd_columns(self, ohlcv):
        m = macd(ohlcv["close"])
        assert list(m.columns) == ["macd", "macd_signal", "macd_hist"]

    def test_bollinger_bands_ordering(self, ohlcv):
        bb = bollinger_bands(ohlcv["close"])
        valid = bb.dropna()
        assert (valid["bb_upper"] >= valid["bb_mid"]).all()
        assert (valid["bb_mid"] >= valid["bb_lower"]).all()

    def test_atr_positive(self, ohlcv):
        a = atr(ohlcv)
        assert (a.dropna() > 0).all()

    def test_stochastic_rsi_columns(self, ohlcv):
        sr = stochastic_rsi(ohlcv["close"])
        assert list(sr.columns) == ["stoch_rsi_k", "stoch_rsi_d"]

    def test_vwap_reasonable(self, ohlcv):
        v = vwap(ohlcv)
        valid = v.dropna()
        assert (valid > 0).all()

    def test_volume_ratio(self, ohlcv):
        vr = volume_ratio(ohlcv["volume"])
        valid = vr.dropna()
        assert (valid > 0).all()


# --- Signal detection ---

class TestSignals:
    def test_crossover_basic(self):
        a = pd.Series([1, 2, 3, 4, 5])
        b = pd.Series([3, 3, 3, 3, 3])
        x = crossover(a, b)
        assert x.iloc[3] is np.True_

    def test_crossunder_basic(self):
        a = pd.Series([5, 4, 3, 2, 1])
        b = pd.Series([3, 3, 3, 3, 3])
        x = crossunder(a, b)
        assert x.iloc[3] is np.True_

    def test_ema_crossovers_columns(self, ohlcv):
        ec = ema_crossovers(ohlcv["close"])
        assert "golden_cross" in ec.columns
        assert "death_cross" in ec.columns

    def test_overbought_oversold_columns(self, ohlcv):
        ob = overbought_oversold(ohlcv["close"])
        assert "overbought" in ob.columns and "oversold" in ob.columns

    def test_support_resistance_columns(self, ohlcv):
        sr = support_resistance(ohlcv)
        assert "pivot" in sr.columns
        assert "resistance_1" in sr.columns
        assert "support_1" in sr.columns

    def test_trend_strength_range(self, ohlcv):
        adx = trend_strength(ohlcv)
        valid = adx.dropna()
        assert (valid >= 0).all()

    def test_volume_spike_boolean(self, ohlcv):
        vs = volume_spike(ohlcv["volume"])
        assert vs.dtype == bool


# --- Engine facade ---

class TestEngine:
    def test_compute_all_adds_columns(self, ohlcv):
        result = compute_all(ohlcv)
        assert len(result) == len(ohlcv)
        expected = [
            "sma_20", "ema_200", "rsi", "macd", "bb_upper", "atr",
            "vwap", "adx", "golden_cross", "overbought", "volume_spike",
        ]
        for col in expected:
            assert col in result.columns, f"Missing: {col}"

    def test_compute_all_does_not_mutate_input(self, ohlcv):
        original_cols = list(ohlcv.columns)
        compute_all(ohlcv)
        assert list(ohlcv.columns) == original_cols
