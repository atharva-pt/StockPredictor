"""Feature engineering pipeline — assembles all feature groups into an ML-ready DataFrame.

LEAKAGE PREVENTION RULES (enforced here):
1. Technical features are shifted by 1 bar (done inside build_technical_features)
2. Momentum features are shifted by 1 bar (done inside build_momentum_features)
3. Sentiment uses only articles published BEFORE the bar timestamp
4. Targets are forward-looking and NEVER concatenated into the feature matrix
5. No future data touches the feature columns — verified by test_no_leakage
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from trading_copilot.features.calendar import build_calendar_features
from trading_copilot.features.momentum import build_momentum_features
from trading_copilot.features.target import build_targets
from trading_copilot.features.technical import build_technical_features
from trading_copilot.logging_setup import get_logger
from trading_copilot.news.models import Article

log = get_logger("features.pipeline")


def build_feature_matrix(
    ohlcv: pd.DataFrame,
    articles: list[Article] | None = None,
    ticker: str = "",
    use_finbert: bool = False,
    target_horizons: list[int] | None = None,
    dropna: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build features + targets from OHLCV and optional news.

    Returns:
        (features, targets) — aligned DataFrames. features has NO target columns.
        If dropna=True, rows with any NaN in features are dropped (both aligned).
    """
    tech = build_technical_features(ohlcv)
    mom = build_momentum_features(ohlcv)
    cal = build_calendar_features(ohlcv.index)

    features = pd.concat([tech, mom, cal], axis=1)

    if articles:
        from trading_copilot.features.sentiment import build_sentiment_features
        sent = build_sentiment_features(ohlcv.index, articles, ticker, use_finbert=use_finbert)
        features = pd.concat([features, sent], axis=1)
    else:
        # Placeholder sentiment columns so downstream code doesn't break
        for col in ("sent_score", "sent_confidence", "sent_bullish_ratio",
                     "sent_article_count", "sent_impact", "sent_score_ma3",
                     "sent_score_ma7", "sent_shift"):
            features[col] = 0.0

    targets = build_targets(ohlcv, horizons=target_horizons)

    # Replace inf/-inf with NaN so they get dropped with dropna
    features = features.replace([np.inf, -np.inf], np.nan)

    if dropna:
        valid_mask = features.notna().all(axis=1)
        valid_mask &= targets.notna().all(axis=1)
        features = features.loc[valid_mask]
        targets = targets.loc[valid_mask]

    log.info(
        "feature_matrix_built",
        rows=len(features),
        feature_cols=len(features.columns),
        target_cols=len(targets.columns),
    )
    return features, targets


def get_feature_names() -> list[str]:
    """Return expected feature column names (for model introspection)."""
    import numpy as np

    dummy_idx = pd.date_range("2020-01-01", periods=300, freq="B", tz="UTC")
    dummy = pd.DataFrame({
        "open": np.random.randn(300).cumsum() + 100,
        "high": np.random.randn(300).cumsum() + 101,
        "low": np.random.randn(300).cumsum() + 99,
        "close": np.random.randn(300).cumsum() + 100,
        "volume": np.abs(np.random.randn(300)) * 1e6,
    }, index=dummy_idx)
    features, _ = build_feature_matrix(dummy, dropna=True)
    return list(features.columns)
