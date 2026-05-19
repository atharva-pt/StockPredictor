"""Tests for the AI explainability layer (Phase 12)."""

from __future__ import annotations

from datetime import UTC, datetime

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import pytest
from sklearn.ensemble import RandomForestClassifier

from trading_copilot.explain.narrative import build_narrative, feature_importance_summary
from trading_copilot.explain.plots import feature_importance_chart, shap_waterfall_chart
from trading_copilot.explain.shap_explain import (
    ExplanationResult,
    explain_ensemble,
    explain_prediction,
)
from trading_copilot.models.engine import Prediction
from trading_copilot.signals.generator import Signal

FEATURE_NAMES = ["rsi", "macd_hist", "ema_20_dist", "bb_position", "volume_ratio"]


@pytest.fixture
def trained_rf():
    """Small RandomForest trained on synthetic data."""
    rng = np.random.RandomState(42)
    X = rng.randn(100, len(FEATURE_NAMES))
    y = (X[:, 0] + X[:, 1] > 0).astype(int)
    model = RandomForestClassifier(n_estimators=10, random_state=42)
    model.fit(X, y)
    return model


@pytest.fixture
def sample_row():
    return pd.Series([0.5, -0.3, 0.1, 0.8, 1.2], index=FEATURE_NAMES)


@pytest.fixture
def sample_prediction():
    return Prediction(
        up_prob=0.67, down_prob=0.33, direction="UP", confidence=0.67, model_name="rf"
    )


@pytest.fixture
def sample_signal():
    return Signal(
        ticker="AAPL",
        timestamp=datetime.now(UTC),
        action="BUY",
        confidence=0.65,
        risk_level="MEDIUM",
        up_prob=0.67,
        down_prob=0.33,
        reasoning="ML model predicts UP.",
        factors=["RSI oversold", "Positive sentiment"],
        model_name="rf",
    )


class TestShapExplain:
    def test_explain_prediction(self, trained_rf, sample_row):
        result = explain_prediction(trained_rf, sample_row, FEATURE_NAMES)

        assert isinstance(result, ExplanationResult)
        assert len(result.shap_values) == len(FEATURE_NAMES)
        assert len(result.top_features) <= 10
        assert result.summary_text
        # All feature names should be present
        assert set(result.shap_values.keys()) == set(FEATURE_NAMES)

    def test_explain_prediction_top_n(self, trained_rf, sample_row):
        result = explain_prediction(trained_rf, sample_row, FEATURE_NAMES, top_n=3)
        assert len(result.top_features) <= 3

    def test_explain_prediction_graceful_failure(self, sample_row):
        """Should return empty explanation if model is incompatible."""
        result = explain_prediction("not_a_model", sample_row, FEATURE_NAMES)
        assert isinstance(result, ExplanationResult)
        assert result.shap_values == {}
        assert "unavailable" in result.summary_text.lower()

    def test_explain_ensemble(self, trained_rf, sample_row):
        models = [trained_rf, trained_rf]
        weights = [0.6, 0.4]
        result = explain_ensemble(models, weights, sample_row, FEATURE_NAMES)

        assert isinstance(result, ExplanationResult)
        assert len(result.shap_values) == len(FEATURE_NAMES)
        assert result.summary_text


class TestNarrative:
    def test_build_narrative(self, sample_prediction, sample_signal):
        explanation = ExplanationResult(
            shap_values={"rsi": 0.12, "macd_hist": 0.08, "volume_ratio": -0.04},
            top_features=[("rsi", 0.12), ("macd_hist", 0.08), ("volume_ratio", -0.04)],
            summary_text="Bullish factors: rsi (+0.12), macd_hist (+0.08).",
        )
        narrative = build_narrative(explanation, sample_prediction, sample_signal)

        assert "UP" in narrative
        assert "67%" in narrative
        assert "rsi" in narrative
        assert "bearish" in narrative.lower()

    def test_feature_importance_summary(self):
        explanation = ExplanationResult(
            shap_values={"rsi": 0.12, "macd_hist": -0.08, "bb_position": 0.03},
            top_features=[("rsi", 0.12), ("macd_hist", -0.08), ("bb_position", 0.03)],
            summary_text="",
        )
        summary = feature_importance_summary(explanation, top_n=3)

        assert "rsi" in summary
        assert "macd_hist" in summary
        assert "bullish" in summary
        assert "bearish" in summary

    def test_feature_importance_summary_empty(self):
        explanation = ExplanationResult()
        summary = feature_importance_summary(explanation)
        assert "No feature importance" in summary


class TestPlots:
    def test_shap_waterfall_chart(self):
        explanation = ExplanationResult(
            shap_values={"rsi": 0.12, "macd_hist": -0.08},
            top_features=[("rsi", 0.12), ("macd_hist", -0.08)],
            summary_text="",
        )
        fig = shap_waterfall_chart(explanation, title="Test Chart")
        assert isinstance(fig, go.Figure)

    def test_feature_importance_chart(self):
        explanation = ExplanationResult(
            shap_values={"rsi": 0.12, "macd_hist": -0.08, "bb_position": 0.03},
            top_features=[],
            summary_text="",
        )
        fig = feature_importance_chart(explanation)
        assert isinstance(fig, go.Figure)

    def test_empty_explanation_charts(self):
        explanation = ExplanationResult()
        fig1 = shap_waterfall_chart(explanation)
        fig2 = feature_importance_chart(explanation)
        assert isinstance(fig1, go.Figure)
        assert isinstance(fig2, go.Figure)
