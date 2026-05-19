"""Calendar/temporal features — day-of-week, month, quarter effects."""

from __future__ import annotations

import numpy as np
import pandas as pd


def build_calendar_features(index: pd.DatetimeIndex) -> pd.DataFrame:
    out = pd.DataFrame(index=index)
    out["day_of_week"] = index.dayofweek  # 0=Mon, 4=Fri
    out["month"] = index.month
    out["quarter"] = index.quarter
    out["is_month_start"] = index.is_month_start.astype(float)
    out["is_month_end"] = index.is_month_end.astype(float)
    out["is_quarter_end"] = index.is_quarter_end.astype(float)

    # Cyclical encoding for day_of_week and month
    out["dow_sin"] = np.sin(2 * np.pi * out["day_of_week"] / 5)
    out["dow_cos"] = np.cos(2 * np.pi * out["day_of_week"] / 5)
    out["month_sin"] = np.sin(2 * np.pi * out["month"] / 12)
    out["month_cos"] = np.cos(2 * np.pi * out["month"] / 12)

    return out
