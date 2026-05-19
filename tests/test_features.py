"""Tests for feature engineering pipeline — synthetic data, leakage checks."""

from __future__ import annotations

from datetime import timedelta

import numpy as np
import pandas as pd

from trading_copilot.features.calendar import build_calendar_features
from trading_copilot.features.momentum import build_momentum_features
from trading_copilot.features.pipeline import build_feature_matrix
from trading_copilot.features.target import build_targets
from trading_copilot.features.technical import build_technical_features
from trading_copilot.news.models import Article


def _ohlcv(n: int = 300) -> pd.DataFrame:
    np.random.seed(42)
    close = 100 + np.cumsum(np.random.randn(n) * 0.5)
    idx = pd.date_range("2023-01-01", periods=n, freq="B", tz="UTC")
    return pd.DataFrame({
        "open": close + np.random.randn(n) * 0.2,
        "high": close + np.abs(np.random.randn(n)) * 0.5,
        "low": close - np.abs(np.random.randn(n)) * 0.5,
        "close": close,
        "volume": np.random.randint(100_000, 1_000_000, size=n).astype(float),
    }, index=idx)


# --- Technical features ---

class TestTechnical:
    def test_shape_and_shift(self):
        df = _ohlcv()
        feat = build_technical_features(df)
        assert len(feat) == len(df)
        # First row should be all NaN (shifted by 1)
        assert feat.iloc[0].isna().all()

    def test_no_raw_prices_in_distance_features(self):
        """Distance/pct features should be small, not raw price magnitude."""
        feat = build_technical_features(_ohlcv())
        valid = feat.dropna()
        for col in valid.columns:
            if col.startswith("dist_") or col.endswith("_pct"):
                assert valid[col].abs().max() < 5.0, f"{col} looks like raw price"


# --- Momentum features ---

class TestMomentum:
    def test_shape(self):
        feat = build_momentum_features(_ohlcv())
        assert len(feat) == 300
        assert "lag_return_1d" in feat.columns
        assert "streak" in feat.columns

    def test_shifted(self):
        feat = build_momentum_features(_ohlcv())
        assert feat.iloc[0].isna().all()


# --- Calendar features ---

class TestCalendar:
    def test_columns(self):
        idx = pd.date_range("2023-01-01", periods=10, freq="B", tz="UTC")
        cal = build_calendar_features(idx)
        assert "day_of_week" in cal.columns
        assert "dow_sin" in cal.columns
        assert "month_sin" in cal.columns

    def test_cyclical_range(self):
        idx = pd.date_range("2023-01-01", periods=250, freq="B", tz="UTC")
        cal = build_calendar_features(idx)
        assert cal["dow_sin"].min() >= -1.0
        assert cal["dow_sin"].max() <= 1.0


# --- Targets ---

class TestTargets:
    def test_target_shape(self):
        df = _ohlcv(100)
        t = build_targets(df, horizons=[1, 5])
        assert "target_1d_dir" in t.columns
        assert "target_5d_dir" in t.columns

    def test_target_last_rows_nan(self):
        df = _ohlcv(100)
        t = build_targets(df, horizons=[5])
        assert t["target_5d_dir"].iloc[-5:].isna().all()

    def test_target_values_binary(self):
        df = _ohlcv(100)
        t = build_targets(df, horizons=[1])
        valid = t["target_1d_dir"].dropna()
        assert set(valid.unique()).issubset({0.0, 1.0})


# --- Pipeline ---

class TestPipeline:
    def test_basic_build(self):
        features, targets = build_feature_matrix(_ohlcv())
        assert len(features) > 0
        assert len(features) == len(targets)
        assert "target_1d_dir" not in features.columns
        assert "target_1d_dir" in targets.columns

    def test_no_nan_after_dropna(self):
        features, targets = build_feature_matrix(_ohlcv(), dropna=True)
        assert features.notna().all().all()
        assert targets.notna().all().all()

    def test_sentiment_placeholder_when_no_articles(self):
        features, _ = build_feature_matrix(_ohlcv())
        assert "sent_score" in features.columns
        assert (features["sent_score"] == 0.0).all()


# --- LEAKAGE DETECTION (critical) ---

class TestNoLeakage:
    def test_features_dont_contain_future_returns(self):
        """The feature at time t must not correlate perfectly with target at time t.

        If shift(1) is missing, feature[t] would use close[t] which partially
        determines target[t]. We test by checking correlation isn't suspiciously high.
        """
        df = _ohlcv(500)
        features, targets = build_feature_matrix(df, dropna=True)

        target_col = "target_1d_dir"
        for col in features.columns:
            # Skip constant columns (e.g. zero-filled sentiment placeholders) — corr is NaN
            if features[col].std() == 0:
                continue
            corr = features[col].corr(targets[target_col])
            assert abs(corr) < 0.5, (
                f"Feature '{col}' has suspiciously high correlation ({corr:.3f}) "
                f"with target — possible leakage"
            )

    def test_feature_at_t_uses_data_before_t(self):
        """Verify the shift: feature at index t should equal the indicator
        computed from data up to t-1."""
        df = _ohlcv(50)
        tech = build_technical_features(df)
        # return_1d at index 5 should be the return from close[3] to close[4]
        # (because shift(1) pushes the t=4 return to index 5)
        expected = (df["close"].iloc[4] - df["close"].iloc[3]) / df["close"].iloc[3]
        actual = tech["return_1d"].iloc[5]
        assert abs(actual - expected) < 1e-10, "Shift-1 not applied correctly"

    def test_sentiment_window_excludes_future(self):
        """Articles published after bar time must not influence that bar's features."""
        df = _ohlcv(30)
        bar_time = df.index[15]

        # Article AFTER bar_time — should NOT appear in bar 15's features
        future_article = Article(
            title="Market crashes hard",
            source="test",
            url="https://x.com/future",
            published_utc=bar_time + timedelta(hours=1),
            summary="Massive sell-off.",
        )

        from trading_copilot.features.sentiment import build_sentiment_features
        sent = build_sentiment_features(df.index, [future_article], ticker="", use_finbert=False)
        assert sent.iloc[15]["sent_article_count"] == 0, "Future article leaked into features"
