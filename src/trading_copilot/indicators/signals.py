"""Pattern detection: crossovers, overbought/oversold, breakouts, support/resistance, trend strength."""

from __future__ import annotations

import numpy as np
import pandas as pd

from trading_copilot.indicators.core import (
    atr,
    bollinger_bands,
    ema,
    macd,
    rsi,
    volume_ratio,
)

# ---------------------------------------------------------------------------
# Crossover helpers
# ---------------------------------------------------------------------------

def crossover(series_a: pd.Series, series_b: pd.Series) -> pd.Series:
    """True on bars where series_a crosses above series_b."""
    prev_a, prev_b = series_a.shift(1), series_b.shift(1)
    return ((series_a > series_b) & (prev_a <= prev_b)).rename("crossover")


def crossunder(series_a: pd.Series, series_b: pd.Series) -> pd.Series:
    prev_a, prev_b = series_a.shift(1), series_b.shift(1)
    return ((series_a < series_b) & (prev_a >= prev_b)).rename("crossunder")


# ---------------------------------------------------------------------------
# EMA crossovers
# ---------------------------------------------------------------------------

def ema_crossovers(close: pd.Series) -> pd.DataFrame:
    e20, e50, e200 = ema(close, 20), ema(close, 50), ema(close, 200)
    return pd.DataFrame({
        "ema_20": e20,
        "ema_50": e50,
        "ema_200": e200,
        "golden_cross": crossover(e50, e200),
        "death_cross": crossunder(e50, e200),
        "ema20_x_ema50_up": crossover(e20, e50),
        "ema20_x_ema50_dn": crossunder(e20, e50),
    }, index=close.index)


# ---------------------------------------------------------------------------
# MACD crossovers
# ---------------------------------------------------------------------------

def macd_crossovers(close: pd.Series) -> pd.DataFrame:
    m = macd(close)
    m["macd_cross_up"] = crossover(m["macd"], m["macd_signal"])
    m["macd_cross_dn"] = crossunder(m["macd"], m["macd_signal"])
    return m


# ---------------------------------------------------------------------------
# Overbought / Oversold
# ---------------------------------------------------------------------------

def overbought_oversold(close: pd.Series, period: int = 14, ob: float = 70, os_: float = 30) -> pd.DataFrame:
    r = rsi(close, period)
    return pd.DataFrame({
        "rsi": r,
        "overbought": r >= ob,
        "oversold": r <= os_,
    }, index=close.index)


# ---------------------------------------------------------------------------
# Bollinger breakout
# ---------------------------------------------------------------------------

def bollinger_breakout(close: pd.Series, period: int = 20, std_dev: float = 2.0) -> pd.DataFrame:
    bb = bollinger_bands(close, period, std_dev)
    bb["bb_breakout_up"] = close > bb["bb_upper"]
    bb["bb_breakout_dn"] = close < bb["bb_lower"]
    bb["bb_squeeze"] = (bb["bb_upper"] - bb["bb_lower"]) / bb["bb_mid"]
    return bb


# ---------------------------------------------------------------------------
# Support / Resistance estimation (pivot-based)
# ---------------------------------------------------------------------------

def support_resistance(df: pd.DataFrame, lookback: int = 20) -> pd.DataFrame:
    """Rolling pivot-point support/resistance using high/low/close."""
    pivot = (df["high"].rolling(lookback).max() + df["low"].rolling(lookback).min() + df["close"]) / 3
    r1 = 2 * pivot - df["low"].rolling(lookback).min()
    s1 = 2 * pivot - df["high"].rolling(lookback).max()
    return pd.DataFrame({"pivot": pivot, "resistance_1": r1, "support_1": s1}, index=df.index)


# ---------------------------------------------------------------------------
# Trend strength (ADX-like simplified)
# ---------------------------------------------------------------------------

def trend_strength(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Simplified trend strength: ATR-normalized directional movement."""
    up_move = df["high"] - df["high"].shift(1)
    dn_move = df["low"].shift(1) - df["low"]
    plus_dm = np.where((up_move > dn_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((dn_move > up_move) & (dn_move > 0), dn_move, 0)
    atr_vals = atr(df, period)
    plus_di = pd.Series(plus_dm, index=df.index).ewm(alpha=1 / period, min_periods=period, adjust=False).mean() / atr_vals.replace(0, np.nan) * 100
    minus_di = pd.Series(minus_dm, index=df.index).ewm(alpha=1 / period, min_periods=period, adjust=False).mean() / atr_vals.replace(0, np.nan) * 100
    dx = ((plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)) * 100
    adx = dx.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    return adx.rename("adx")


# ---------------------------------------------------------------------------
# Volume spike detection
# ---------------------------------------------------------------------------

def volume_spike(volume: pd.Series, threshold: float = 2.0, period: int = 20) -> pd.Series:
    ratio = volume_ratio(volume, period)
    return (ratio >= threshold).rename("volume_spike")
