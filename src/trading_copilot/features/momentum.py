"""Momentum and regime features for ML."""

from __future__ import annotations

import numpy as np
import pandas as pd


def build_momentum_features(df: pd.DataFrame) -> pd.DataFrame:
    """Lagged returns, streaks, regime detection. Input must have 'close' and 'volume'."""
    c = df["close"]
    out = pd.DataFrame(index=df.index)

    # Lagged returns at multiple horizons
    for lag in (1, 2, 3, 5, 10):
        out[f"lag_return_{lag}d"] = c.pct_change(lag)

    # Return momentum (acceleration)
    ret_1d = c.pct_change(1)
    out["momentum_accel"] = ret_1d.diff()

    # Win/loss streak
    direction = np.sign(ret_1d)
    streak = direction.copy()
    for i in range(1, len(streak)):
        if direction.iloc[i] == direction.iloc[i - 1] and direction.iloc[i] != 0:
            streak.iloc[i] = streak.iloc[i - 1] + direction.iloc[i]
        else:
            streak.iloc[i] = direction.iloc[i]
    out["streak"] = streak

    # High/low distance (normalized)
    rolling_high_20 = df["high"].rolling(20).max()
    rolling_low_20 = df["low"].rolling(20).min()
    out["dist_from_20d_high"] = (c - rolling_high_20) / rolling_high_20.replace(0, np.nan)
    out["dist_from_20d_low"] = (c - rolling_low_20) / rolling_low_20.replace(0, np.nan)

    rolling_high_50 = df["high"].rolling(50).max()
    rolling_low_50 = df["low"].rolling(50).min()
    out["dist_from_50d_high"] = (c - rolling_high_50) / rolling_high_50.replace(0, np.nan)
    out["dist_from_50d_low"] = (c - rolling_low_50) / rolling_low_50.replace(0, np.nan)

    # Market regime: rolling mean return vs rolling volatility
    ret_20 = c.pct_change(1).rolling(20).mean()
    vol_20 = c.pct_change(1).rolling(20).std()
    out["regime_sharpe_20d"] = ret_20 / vol_20.replace(0, np.nan)

    # Volume-price divergence
    vol_change = df["volume"].pct_change(5)
    price_change = c.pct_change(5)
    out["vol_price_divergence"] = vol_change - price_change

    # Shift to prevent leakage
    out = out.shift(1)

    return out
