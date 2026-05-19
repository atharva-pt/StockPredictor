"""Technical indicator features for ML — all derived from t-1 and earlier only."""

from __future__ import annotations

import numpy as np
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
    vwap,
)
from trading_copilot.indicators.signals import trend_strength


def build_technical_features(df: pd.DataFrame) -> pd.DataFrame:
    """Build technical features from OHLCV. All values use data up to t-1 (shifted)."""
    c = df["close"]
    out = pd.DataFrame(index=df.index)

    # Price-relative features (normalize for cross-stock comparability)
    out["return_1d"] = c.pct_change(1)
    out["return_5d"] = c.pct_change(5)
    out["return_10d"] = c.pct_change(10)
    out["return_20d"] = c.pct_change(20)

    # Volatility
    out["volatility_5d"] = c.pct_change().rolling(5).std()
    out["volatility_20d"] = c.pct_change().rolling(20).std()

    # RSI
    out["rsi_14"] = rsi(c, 14)

    # Stochastic RSI
    sr = stochastic_rsi(c)
    out["stoch_rsi_k"] = sr["stoch_rsi_k"]
    out["stoch_rsi_d"] = sr["stoch_rsi_d"]

    # MACD
    m = macd(c)
    out["macd_line"] = m["macd"]
    out["macd_signal"] = m["macd_signal"]
    out["macd_hist"] = m["macd_hist"]
    out["macd_hist_diff"] = m["macd_hist"].diff()

    # Bollinger
    bb = bollinger_bands(c)
    bb_width = bb["bb_upper"] - bb["bb_lower"]
    out["bb_position"] = (c - bb["bb_lower"]) / bb_width.replace(0, np.nan)
    out["bb_width_pct"] = bb_width / bb["bb_mid"].replace(0, np.nan)

    # Moving average distances (normalized)
    for period in (20, 50, 200):
        e = ema(c, period)
        out[f"dist_ema_{period}"] = (c - e) / e.replace(0, np.nan)

    s20 = sma(c, 20)
    s50 = sma(c, 50)
    out["sma20_above_sma50"] = (s20 > s50).astype(float)

    # ATR normalized
    atr_val = atr(df, 14)
    out["atr_pct"] = atr_val / c.replace(0, np.nan)

    # VWAP distance
    v = vwap(df)
    out["dist_vwap"] = (c - v) / v.replace(0, np.nan)

    # Volume features
    out["vol_ratio_20"] = volume_ratio(df["volume"], 20)
    out["vol_change_1d"] = df["volume"].pct_change(1)

    # ADX
    out["adx"] = trend_strength(df, 14)

    # Candle body/shadow ratios
    body = (df["close"] - df["open"]).abs()
    full_range = (df["high"] - df["low"]).replace(0, np.nan)
    out["body_ratio"] = body / full_range
    out["upper_shadow"] = (df["high"] - df[["close", "open"]].max(axis=1)) / full_range
    out["lower_shadow"] = (df[["close", "open"]].min(axis=1) - df["low"]) / full_range

    # Gap
    out["gap_pct"] = (df["open"] - df["close"].shift(1)) / df["close"].shift(1).replace(0, np.nan)

    # SHIFT everything by 1 to prevent leakage: features at time t use data from t-1
    out = out.shift(1)

    return out
