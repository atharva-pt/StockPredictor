"""Core technical indicators computed on OHLCV DataFrames.

Every function takes a DataFrame with columns [open, high, low, close, volume]
and a UTC DatetimeIndex. Returns a Series or DataFrame — never mutates the input.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Moving averages
# ---------------------------------------------------------------------------

def sma(close: pd.Series, period: int = 20) -> pd.Series:
    return close.rolling(window=period, min_periods=period).mean()


def ema(close: pd.Series, period: int = 20) -> pd.Series:
    return close.ewm(span=period, adjust=False, min_periods=period).mean()


# ---------------------------------------------------------------------------
# RSI
# ---------------------------------------------------------------------------

def rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


# ---------------------------------------------------------------------------
# Stochastic RSI
# ---------------------------------------------------------------------------

def stochastic_rsi(
    close: pd.Series, rsi_period: int = 14, stoch_period: int = 14, k_smooth: int = 3, d_smooth: int = 3,
) -> pd.DataFrame:
    rsi_vals = rsi(close, rsi_period)
    stoch_rsi = (rsi_vals - rsi_vals.rolling(stoch_period).min()) / (
        rsi_vals.rolling(stoch_period).max() - rsi_vals.rolling(stoch_period).min()
    )
    k = stoch_rsi.rolling(k_smooth).mean() * 100
    d = k.rolling(d_smooth).mean()
    return pd.DataFrame({"stoch_rsi_k": k, "stoch_rsi_d": d}, index=close.index)


# ---------------------------------------------------------------------------
# MACD
# ---------------------------------------------------------------------------

def macd(
    close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9,
) -> pd.DataFrame:
    ema_fast = ema(close, fast)
    ema_slow = ema(close, slow)
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False, min_periods=signal).mean()
    histogram = macd_line - signal_line
    return pd.DataFrame(
        {"macd": macd_line, "macd_signal": signal_line, "macd_hist": histogram},
        index=close.index,
    )


# ---------------------------------------------------------------------------
# Bollinger Bands
# ---------------------------------------------------------------------------

def bollinger_bands(close: pd.Series, period: int = 20, std_dev: float = 2.0) -> pd.DataFrame:
    mid = sma(close, period)
    std = close.rolling(window=period, min_periods=period).std()
    return pd.DataFrame(
        {"bb_upper": mid + std_dev * std, "bb_mid": mid, "bb_lower": mid - std_dev * std},
        index=close.index,
    )


# ---------------------------------------------------------------------------
# ATR (Average True Range)
# ---------------------------------------------------------------------------

def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high, low, prev_close = df["high"], df["low"], df["close"].shift(1)
    tr = pd.concat(
        [high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1,
    ).max(axis=1)
    return tr.ewm(alpha=1 / period, min_periods=period, adjust=False).mean().rename("atr")


# ---------------------------------------------------------------------------
# VWAP (session-based; for daily bars uses cumulative intraday proxy)
# ---------------------------------------------------------------------------

def vwap(df: pd.DataFrame) -> pd.Series:
    typical = (df["high"] + df["low"] + df["close"]) / 3
    cum_tp_vol = (typical * df["volume"]).cumsum()
    cum_vol = df["volume"].cumsum().replace(0, np.nan)
    return (cum_tp_vol / cum_vol).rename("vwap")


# ---------------------------------------------------------------------------
# Volume indicators
# ---------------------------------------------------------------------------

def volume_sma(volume: pd.Series, period: int = 20) -> pd.Series:
    return volume.rolling(window=period, min_periods=period).mean().rename("vol_sma")


def volume_ratio(volume: pd.Series, period: int = 20) -> pd.Series:
    avg = volume_sma(volume, period)
    return (volume / avg.replace(0, np.nan)).rename("vol_ratio")
