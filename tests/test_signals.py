"""Tests for signal generation — rules, confirmation, and generator."""

from __future__ import annotations

import pandas as pd

from trading_copilot.models.engine import Prediction
from trading_copilot.signals.generator import Signal, generate_signal, generate_signals
from trading_copilot.signals.rules import (
    sentiment_confirmation,
    technical_confirmation,
    volatility_filter,
)


def _feature_row(**overrides) -> pd.Series:
    defaults = {
        "rsi_14": 50, "macd_hist": 0, "macd_hist_diff": 0,
        "dist_ema_20": 0, "dist_ema_50": 0, "bb_position": 0.5,
        "adx": 20, "vol_ratio_20": 1.0, "atr_pct": 0.02,
        "sent_score": 0, "sent_confidence": 0, "sent_article_count": 0,
        "sent_shift": 0, "volume_spike": False,
    }
    defaults.update(overrides)
    return pd.Series(defaults)


def _prediction(direction="UP", up=0.65, down=0.35, conf=0.65) -> Prediction:
    return Prediction(up_prob=up, down_prob=down, direction=direction, confidence=conf, model_name="test")


# --- Technical confirmation ---

class TestTechnical:
    def test_oversold_rsi_bullish(self):
        score, factors = technical_confirmation(_feature_row(rsi_14=25))
        assert score > 0
        assert any("oversold" in f for f in factors)

    def test_overbought_rsi_bearish(self):
        score, _factors = technical_confirmation(_feature_row(rsi_14=75))
        assert score < 0

    def test_bullish_macd(self):
        score, _ = technical_confirmation(_feature_row(macd_hist=1, macd_hist_diff=0.5))
        assert score > 0

    def test_bearish_ema(self):
        score, _ = technical_confirmation(_feature_row(dist_ema_20=-0.05, dist_ema_50=-0.03))
        assert score < 0

    def test_score_bounded(self):
        score, _ = technical_confirmation(_feature_row(
            rsi_14=20, macd_hist=5, macd_hist_diff=2,
            dist_ema_20=0.1, dist_ema_50=0.1, bb_position=0.05,
        ))
        assert -1 <= score <= 1


# --- Sentiment confirmation ---

class TestSentiment:
    def test_no_articles_neutral(self):
        score, factors = sentiment_confirmation(_feature_row(sent_article_count=0))
        assert score == 0.0
        assert any("No recent news" in f for f in factors)

    def test_bullish_sentiment(self):
        score, _factors = sentiment_confirmation(_feature_row(
            sent_score=0.5, sent_confidence=0.8, sent_article_count=5,
        ))
        assert score > 0

    def test_sentiment_shift_detected(self):
        _, factors = sentiment_confirmation(_feature_row(
            sent_score=0.1, sent_confidence=0.5, sent_article_count=3, sent_shift=0.3,
        ))
        assert any("improving" in f for f in factors)


# --- Volatility filter ---

class TestVolatility:
    def test_normal_vol_passes(self):
        ok, _ = volatility_filter(_feature_row(atr_pct=0.02))
        assert ok is True

    def test_high_vol_fails(self):
        ok, factors = volatility_filter(_feature_row(atr_pct=0.08))
        assert ok is False
        assert any("High volatility" in f for f in factors)


# --- Signal generator ---

class TestGenerator:
    def test_buy_signal(self):
        sig = generate_signal(
            "TEST.NS", _prediction("UP", 0.70, 0.30, 0.70),
            _feature_row(rsi_14=35, dist_ema_20=0.01, dist_ema_50=0.01),
        )
        assert isinstance(sig, Signal)
        assert sig.action == "BUY"
        assert sig.confidence > 0.5

    def test_sell_signal(self):
        sig = generate_signal(
            "TEST.NS", _prediction("DOWN", 0.30, 0.70, 0.70),
            _feature_row(rsi_14=75, dist_ema_20=-0.05, dist_ema_50=-0.03),
        )
        assert sig.action == "SELL"

    def test_hold_on_low_confidence(self):
        sig = generate_signal(
            "TEST.NS", _prediction("UP", 0.52, 0.48, 0.52),
            _feature_row(), min_confidence=0.55,
        )
        assert sig.action == "HOLD"

    def test_hold_when_technical_contradicts(self):
        sig = generate_signal(
            "TEST.NS", _prediction("UP", 0.60, 0.40, 0.60),
            _feature_row(rsi_14=80, dist_ema_20=-0.05, dist_ema_50=-0.03, bb_position=0.95),
            require_technical_alignment=True,
        )
        assert sig.action == "HOLD"

    def test_risk_level_low_on_high_confidence(self):
        sig = generate_signal(
            "TEST.NS", _prediction("UP", 0.80, 0.20, 0.80),
            _feature_row(rsi_14=40, dist_ema_20=0.02, dist_ema_50=0.01, atr_pct=0.02),
        )
        assert sig.risk_level == "LOW"

    def test_risk_level_high_on_volatility(self):
        sig = generate_signal(
            "TEST.NS", _prediction("UP", 0.65, 0.35, 0.65),
            _feature_row(atr_pct=0.08),
        )
        assert sig.risk_level == "HIGH"

    def test_reasoning_not_empty(self):
        sig = generate_signal("TEST.NS", _prediction(), _feature_row())
        assert len(sig.reasoning) > 20

    def test_factors_populated(self):
        sig = generate_signal("TEST.NS", _prediction(), _feature_row(sent_article_count=3, sent_score=0.3, sent_confidence=0.7))
        assert len(sig.factors) > 0

    def test_generate_signals_batch(self):
        preds = [_prediction() for _ in range(3)]
        features = pd.DataFrame([_feature_row().to_dict() for _ in range(3)])
        sigs = generate_signals("TEST.NS", preds, features)
        assert len(sigs) == 3
        assert all(isinstance(s, Signal) for s in sigs)
