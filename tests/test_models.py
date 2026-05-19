"""Tests for ML engine — synthetic data, walk-forward validation, no overfitting claims."""

from __future__ import annotations

import numpy as np
import pandas as pd

from trading_copilot.models.engine import (
    Prediction,
    compare_models,
    predict,
    predict_ensemble,
    train_ensemble,
    train_final_model,
    walk_forward_evaluate,
)
from trading_copilot.models.trainers import ensemble_predict_proba, get_model
from trading_copilot.models.validation import rolling_window_splits, walk_forward_splits


def _synthetic_dataset(n: int = 500, n_features: int = 20):
    """Synthetic classification dataset with mild signal — not perfectly separable."""
    np.random.seed(42)
    X = pd.DataFrame(np.random.randn(n, n_features), columns=[f"f{i}" for i in range(n_features)])
    signal = 0.3 * X["f0"] + 0.2 * X["f1"] - 0.1 * X["f2"] + np.random.randn(n) * 0.8
    y = pd.Series((signal > 0).astype(float), name="target_5d_dir")
    return X, y


# --- Validation splits ---

class TestValidation:
    def test_walk_forward_chronological(self):
        splits = walk_forward_splits(500, n_splits=5, min_train_size=200)
        assert len(splits) >= 3
        for s in splits:
            assert s.train_idx.max() < s.test_idx.min(), "Train must precede test"

    def test_walk_forward_expanding(self):
        splits = walk_forward_splits(500, n_splits=5, min_train_size=200)
        train_sizes = [len(s.train_idx) for s in splits]
        for i in range(1, len(train_sizes)):
            assert train_sizes[i] > train_sizes[i - 1], "Training window must expand"

    def test_rolling_window_fixed_size(self):
        splits = rolling_window_splits(500, train_size=200, test_size=40)
        for s in splits:
            assert len(s.train_idx) == 200
            assert len(s.test_idx) == 40
            assert s.train_idx.max() < s.test_idx.min()

    def test_no_overlap_between_train_and_test(self):
        splits = walk_forward_splits(300, n_splits=3, min_train_size=100)
        for s in splits:
            assert len(set(s.train_idx) & set(s.test_idx)) == 0


# --- Model trainers ---

class TestTrainers:
    def test_get_model_rf(self):
        m = get_model("random_forest")
        assert hasattr(m, "fit")
        assert hasattr(m, "predict_proba")

    def test_get_model_xgb(self):
        m = get_model("xgboost")
        assert hasattr(m, "fit")

    def test_get_model_lgbm(self):
        m = get_model("lightgbm")
        assert hasattr(m, "fit")

    def test_get_model_unknown_raises(self):
        try:
            get_model("deepnet")
            raise AssertionError("Should have raised")
        except ValueError:
            pass

    def test_ensemble_predict_proba(self):
        X, y = _synthetic_dataset(200, 5)
        m1 = get_model("random_forest", n_estimators=10)
        m2 = get_model("lightgbm", n_estimators=10)
        m1.fit(X.values, y.values)
        m2.fit(X.values, y.values)
        probas = ensemble_predict_proba([m1, m2], X.values[:10])
        assert probas.shape == (10, 2)
        assert np.allclose(probas.sum(axis=1), 1.0, atol=0.01)


# --- Walk-forward evaluation ---

class TestWalkForward:
    def test_evaluate_returns_metrics(self):
        X, y = _synthetic_dataset(500, 10)
        result = walk_forward_evaluate(X, y, "lightgbm", n_splits=3, min_train_size=200)
        assert result.model_name == "lightgbm"
        assert len(result.fold_results) >= 2
        assert 0.0 <= result.avg_accuracy <= 1.0
        assert 0.0 <= result.avg_auc <= 1.0

    def test_accuracy_above_random(self):
        """With mild signal, walk-forward accuracy should beat 50% coin flip."""
        X, y = _synthetic_dataset(500, 10)
        result = walk_forward_evaluate(X, y, "lightgbm", n_splits=3, min_train_size=200)
        assert result.avg_accuracy > 0.50, f"Accuracy {result.avg_accuracy:.3f} is below random"

    def test_accuracy_below_suspicious(self):
        """If accuracy is too high on synthetic data, something is wrong."""
        X, y = _synthetic_dataset(500, 10)
        result = walk_forward_evaluate(X, y, "lightgbm", n_splits=3, min_train_size=200)
        assert result.avg_accuracy < 0.80, f"Accuracy {result.avg_accuracy:.3f} is suspiciously high"


# --- Compare models ---

class TestCompare:
    def test_compare_returns_sorted(self):
        X, y = _synthetic_dataset(400, 10)
        results = compare_models(X, y, n_splits=3, min_train_size=150)
        assert len(results) == 3
        aucs = [r.avg_auc for r in results]
        assert aucs == sorted(aucs, reverse=True)


# --- Predict ---

class TestPredict:
    def test_predict_output_format(self):
        X, y = _synthetic_dataset(300, 10)
        model = train_final_model(X, y, "lightgbm")
        preds = predict(model, X.tail(5))
        assert len(preds) == 5
        for p in preds:
            assert isinstance(p, Prediction)
            assert p.direction in ("UP", "DOWN", "HOLD")
            assert 0.0 <= p.up_prob <= 1.0
            assert 0.0 <= p.down_prob <= 1.0
            assert abs(p.up_prob + p.down_prob - 1.0) < 0.01

    def test_hold_when_low_confidence(self):
        X, y = _synthetic_dataset(300, 10)
        model = train_final_model(X, y, "lightgbm")
        preds = predict(model, X.tail(20), min_confidence=0.99)
        hold_count = sum(1 for p in preds if p.direction == "HOLD")
        assert hold_count > 0, "High min_confidence should produce some HOLDs"


# --- Ensemble ---

class TestEnsemble:
    def test_ensemble_train_and_predict(self):
        X, y = _synthetic_dataset(400, 10)
        models, weights = train_ensemble(X, y)
        assert len(models) == 3
        assert abs(sum(weights) - 1.0) < 0.01

        preds = predict_ensemble(models, weights, X.tail(5))
        assert len(preds) == 5
        for p in preds:
            assert p.model_name == "ensemble"
            assert p.direction in ("UP", "DOWN", "HOLD")


# --- Persistence ---

class TestPersistence:
    def test_save_and_load(self, tmp_path):
        X, y = _synthetic_dataset(200, 5)
        path = tmp_path / "model.joblib"
        model = train_final_model(X, y, "lightgbm", save_path=path)
        assert path.exists()

        from trading_copilot.models.engine import load_model
        loaded = load_model(path)
        orig_pred = model.predict_proba(X.values[:3])
        load_pred = loaded.predict_proba(X.values[:3])
        assert np.allclose(orig_pred, load_pred)
