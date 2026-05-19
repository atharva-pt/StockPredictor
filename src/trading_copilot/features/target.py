"""Target variable construction for ML training.

The target is FORWARD-looking (intentionally). It is NEVER included in the feature set.
It's computed here so training code can import it alongside features.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def build_targets(df: pd.DataFrame, horizons: list[int] | None = None) -> pd.DataFrame:
    """Build directional targets: 1=UP, 0=DOWN for each horizon.

    horizon_days=5 means: did the close 5 bars ahead go up vs today's close?
    The last `horizon` rows will be NaN (no future data available).
    """
    if horizons is None:
        horizons = [1, 5]

    c = df["close"]
    out = pd.DataFrame(index=df.index)

    for h in horizons:
        future_return = c.shift(-h) / c - 1
        out[f"target_{h}d_return"] = future_return
        out[f"target_{h}d_dir"] = (future_return > 0).astype(float)
        # NaN out the rows where we don't have future data
        out.loc[out.index[-h:], [f"target_{h}d_return", f"target_{h}d_dir"]] = np.nan

    return out
