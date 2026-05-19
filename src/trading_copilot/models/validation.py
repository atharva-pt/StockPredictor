"""Time-series aware validation — walk-forward and rolling window splits.

NEVER use random shuffle on time-series data. These splitters preserve
chronological order to prevent future data from leaking into training.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class Split:
    train_idx: np.ndarray
    test_idx: np.ndarray
    fold: int


def walk_forward_splits(
    n_samples: int,
    n_splits: int = 5,
    min_train_size: int = 200,
    test_size: int | None = None,
) -> list[Split]:
    """Expanding-window walk-forward validation.

    Fold 1: train[0:k], test[k:k+step]
    Fold 2: train[0:k+step], test[k+step:k+2*step]
    ...
    Training window grows each fold. Test window is fixed size.
    """
    if test_size is None:
        test_size = max((n_samples - min_train_size) // n_splits, 20)

    splits: list[Split] = []
    for i in range(n_splits):
        train_end = min_train_size + i * test_size
        test_end = train_end + test_size
        if test_end > n_samples:
            break
        splits.append(Split(
            train_idx=np.arange(0, train_end),
            test_idx=np.arange(train_end, test_end),
            fold=i,
        ))
    return splits


def rolling_window_splits(
    n_samples: int,
    train_size: int = 200,
    test_size: int = 40,
    step: int | None = None,
) -> list[Split]:
    """Fixed-size rolling window — train window slides forward each fold.

    Unlike walk-forward, training window does NOT grow. This tests model
    robustness across different market regimes.
    """
    if step is None:
        step = test_size

    splits: list[Split] = []
    fold = 0
    start = 0
    while start + train_size + test_size <= n_samples:
        train_end = start + train_size
        test_end = train_end + test_size
        splits.append(Split(
            train_idx=np.arange(start, train_end),
            test_idx=np.arange(train_end, test_end),
            fold=fold,
        ))
        start += step
        fold += 1
    return splits
