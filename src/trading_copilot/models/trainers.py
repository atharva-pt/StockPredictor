"""Model definitions: Random Forest, XGBoost, LightGBM.

Each returns a fitted sklearn-compatible estimator with predict_proba support.
Hyperparameters are conservative defaults — tuning comes via walk-forward CV.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier


def get_model(name: str, **overrides: Any):
    """Return an unfitted model instance by name."""
    builders = {
        "random_forest": _rf,
        "xgboost": _xgb,
        "lightgbm": _lgbm,
    }
    if name not in builders:
        raise ValueError(f"Unknown model: {name}. Choose from {list(builders)}")
    return builders[name](**overrides)


def _rf(**kw) -> RandomForestClassifier:
    defaults = dict(
        n_estimators=300,
        max_depth=8,
        min_samples_leaf=20,
        max_features="sqrt",
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )
    defaults.update(kw)
    return RandomForestClassifier(**defaults)


def _xgb(**kw) -> XGBClassifier:
    defaults = dict(
        n_estimators=300,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=10,
        reg_alpha=0.1,
        reg_lambda=1.0,
        scale_pos_weight=1.0,
        use_label_encoder=False,
        eval_metric="logloss",
        random_state=42,
        n_jobs=-1,
        verbosity=0,
    )
    defaults.update(kw)
    return XGBClassifier(**defaults)


def _lgbm(**kw) -> LGBMClassifier:
    defaults = dict(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_samples=20,
        reg_alpha=0.1,
        reg_lambda=1.0,
        is_unbalance=True,
        random_state=42,
        n_jobs=-1,
        verbose=-1,
    )
    defaults.update(kw)
    return LGBMClassifier(**defaults)


def ensemble_predict_proba(
    models: list,
    X: np.ndarray,
    weights: list[float] | None = None,
) -> np.ndarray:
    """Weighted average of predict_proba across multiple models."""
    if weights is None:
        weights = [1.0 / len(models)] * len(models)

    probas = np.zeros((X.shape[0], 2))
    for model, w in zip(models, weights, strict=True):
        probas += w * model.predict_proba(X)
    return probas
