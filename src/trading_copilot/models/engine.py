"""ML prediction engine — train, evaluate, predict with walk-forward validation."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, log_loss, roc_auc_score

from trading_copilot.logging_setup import get_logger
from trading_copilot.models.trainers import ensemble_predict_proba, get_model
from trading_copilot.models.validation import walk_forward_splits

log = get_logger("models.engine")


@dataclass
class FoldResult:
    fold: int
    accuracy: float
    auc: float
    f1: float
    log_loss_val: float
    n_train: int
    n_test: int


@dataclass
class TrainResult:
    model_name: str
    target_col: str
    fold_results: list[FoldResult] = field(default_factory=list)
    avg_accuracy: float = 0.0
    avg_auc: float = 0.0
    avg_f1: float = 0.0
    avg_log_loss: float = 0.0


@dataclass
class Prediction:
    up_prob: float
    down_prob: float
    direction: str  # UP | DOWN | HOLD
    confidence: float
    model_name: str


def walk_forward_evaluate(
    features: pd.DataFrame,
    targets: pd.Series,
    model_name: str = "lightgbm",
    n_splits: int = 5,
    min_train_size: int = 200,
) -> TrainResult:
    """Run walk-forward CV on a single model. Returns aggregated metrics."""
    X = features.values
    y = targets.values

    splits = walk_forward_splits(len(X), n_splits=n_splits, min_train_size=min_train_size)
    result = TrainResult(model_name=model_name, target_col=targets.name or "target")

    for split in splits:
        X_train, y_train = X[split.train_idx], y[split.train_idx]
        X_test, y_test = X[split.test_idx], y[split.test_idx]

        model = get_model(model_name)
        model.fit(X_train, y_train)

        y_proba = model.predict_proba(X_test)[:, 1]
        y_pred = (y_proba >= 0.5).astype(int)

        fold = FoldResult(
            fold=split.fold,
            accuracy=accuracy_score(y_test, y_pred),
            auc=_safe_auc(y_test, y_proba),
            f1=f1_score(y_test, y_pred, zero_division=0),
            log_loss_val=log_loss(y_test, y_proba, labels=[0, 1]),
            n_train=len(y_train),
            n_test=len(y_test),
        )
        result.fold_results.append(fold)
        log.info(
            "fold_done",
            model=model_name,
            fold=split.fold,
            acc=f"{fold.accuracy:.3f}",
            auc=f"{fold.auc:.3f}",
            train=fold.n_train,
            test=fold.n_test,
        )

    if result.fold_results:
        result.avg_accuracy = np.mean([f.accuracy for f in result.fold_results])
        result.avg_auc = np.mean([f.auc for f in result.fold_results])
        result.avg_f1 = np.mean([f.f1 for f in result.fold_results])
        result.avg_log_loss = np.mean([f.log_loss_val for f in result.fold_results])

    log.info(
        "cv_complete",
        model=model_name,
        avg_acc=f"{result.avg_accuracy:.3f}",
        avg_auc=f"{result.avg_auc:.3f}",
        folds=len(result.fold_results),
    )
    return result


def compare_models(
    features: pd.DataFrame,
    targets: pd.Series,
    model_names: list[str] | None = None,
    **cv_kwargs,
) -> list[TrainResult]:
    """Run walk-forward CV for each model. Returns results sorted by AUC desc."""
    if model_names is None:
        model_names = ["random_forest", "xgboost", "lightgbm"]

    results = [
        walk_forward_evaluate(features, targets, name, **cv_kwargs)
        for name in model_names
    ]
    results.sort(key=lambda r: r.avg_auc, reverse=True)
    return results


def train_final_model(
    features: pd.DataFrame,
    targets: pd.Series,
    model_name: str = "lightgbm",
    save_path: Path | None = None,
):
    """Train on full dataset (for production use). Optionally persist to disk."""
    model = get_model(model_name)
    model.fit(features.values, targets.values)
    log.info("final_model_trained", model=model_name, samples=len(features))

    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(model, save_path)
        log.info("model_saved", path=str(save_path))

    return model


def load_model(path: Path):
    return joblib.load(path)


def predict(
    model,
    features: pd.DataFrame,
    model_name: str = "model",
    min_confidence: float = 0.55,
) -> list[Prediction]:
    """Generate predictions with directional classification."""
    probas = model.predict_proba(features.values)
    predictions: list[Prediction] = []

    for proba in probas:
        down_prob, up_prob = float(proba[0]), float(proba[1])
        confidence = max(up_prob, down_prob)

        if confidence < min_confidence:
            direction = "HOLD"
        elif up_prob > down_prob:
            direction = "UP"
        else:
            direction = "DOWN"

        predictions.append(Prediction(
            up_prob=round(up_prob, 4),
            down_prob=round(down_prob, 4),
            direction=direction,
            confidence=round(confidence, 4),
            model_name=model_name,
        ))

    return predictions


def train_ensemble(
    features: pd.DataFrame,
    targets: pd.Series,
    model_names: list[str] | None = None,
    save_dir: Path | None = None,
) -> tuple[list, list[float]]:
    """Train all models and return (models, weights) for ensemble prediction.

    Weights are proportional to walk-forward AUC.
    """
    if model_names is None:
        model_names = ["random_forest", "xgboost", "lightgbm"]

    cv_results = compare_models(features, targets, model_names)
    total_auc = sum(r.avg_auc for r in cv_results)

    models = []
    weights = []
    for r in cv_results:
        m = train_final_model(
            features, targets, r.model_name,
            save_path=(save_dir / f"{r.model_name}.joblib") if save_dir else None,
        )
        models.append(m)
        weights.append(r.avg_auc / total_auc if total_auc > 0 else 1.0 / len(cv_results))

    return models, weights


def predict_ensemble(
    models: list,
    weights: list[float],
    features: pd.DataFrame,
    min_confidence: float = 0.55,
) -> list[Prediction]:
    """Ensemble prediction — weighted average of probabilities."""
    probas = ensemble_predict_proba(models, features.values, weights)
    predictions: list[Prediction] = []

    for proba in probas:
        down_prob, up_prob = float(proba[0]), float(proba[1])
        confidence = max(up_prob, down_prob)

        if confidence < min_confidence:
            direction = "HOLD"
        elif up_prob > down_prob:
            direction = "UP"
        else:
            direction = "DOWN"

        predictions.append(Prediction(
            up_prob=round(up_prob, 4),
            down_prob=round(down_prob, 4),
            direction=direction,
            confidence=round(confidence, 4),
            model_name="ensemble",
        ))

    return predictions


def _safe_auc(y_true, y_proba) -> float:
    try:
        return roc_auc_score(y_true, y_proba)
    except ValueError:
        return 0.5
