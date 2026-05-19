"""Facade: compute all indicators for a given OHLCV DataFrame in one call."""

from __future__ import annotations

import pandas as pd

from trading_copilot.indicators.core import (
    atr,
    bollinger_bands,
    ema,
    macd,
    rsi,
    sma,
    stochastic_rsi,
    volume_ratio,
    volume_sma,
    vwap,
)
from trading_copilot.indicators.signals import (
    bollinger_breakout,
    ema_crossovers,
    macd_crossovers,
    overbought_oversold,
    support_resistance,
    trend_strength,
    volume_spike,
)


def compute_all(df: pd.DataFrame) -> pd.DataFrame:
    """Attach all indicators as new columns to a copy of the OHLCV DataFrame.

    Input must have columns: open, high, low, close, volume with a DatetimeIndex.
    """
    out = df.copy()
    c = df["close"]

    # Moving averages
    out["sma_20"] = sma(c, 20)
    out["sma_50"] = sma(c, 50)
    out["ema_20"] = ema(c, 20)
    out["ema_50"] = ema(c, 50)
    out["ema_200"] = ema(c, 200)

    # RSI
    out["rsi"] = rsi(c)

    # Stochastic RSI
    stoch = stochastic_rsi(c)
    out = pd.concat([out, stoch], axis=1)

    # MACD
    m = macd(c)
    out = pd.concat([out, m], axis=1)

    # Bollinger Bands
    bb = bollinger_bands(c)
    out = pd.concat([out, bb], axis=1)

    # ATR
    out["atr"] = atr(df)

    # VWAP
    out["vwap"] = vwap(df)

    # Volume
    out["vol_sma"] = volume_sma(df["volume"])
    out["vol_ratio"] = volume_ratio(df["volume"])

    # Crossovers
    ema_x = ema_crossovers(c)
    for col in ["golden_cross", "death_cross", "ema20_x_ema50_up", "ema20_x_ema50_dn"]:
        out[col] = ema_x[col]

    # MACD crossovers
    macd_x = macd_crossovers(c)
    out["macd_cross_up"] = macd_x["macd_cross_up"]
    out["macd_cross_dn"] = macd_x["macd_cross_dn"]

    # Overbought / oversold
    ob = overbought_oversold(c)
    out["overbought"] = ob["overbought"]
    out["oversold"] = ob["oversold"]

    # Bollinger breakout
    bb_brk = bollinger_breakout(c)
    out["bb_breakout_up"] = bb_brk["bb_breakout_up"]
    out["bb_breakout_dn"] = bb_brk["bb_breakout_dn"]
    out["bb_squeeze"] = bb_brk["bb_squeeze"]

    # Support / resistance
    sr = support_resistance(df)
    out = pd.concat([out, sr], axis=1)

    # Trend strength (ADX)
    out["adx"] = trend_strength(df)

    # Volume spike
    out["volume_spike"] = volume_spike(df["volume"])

    return out
