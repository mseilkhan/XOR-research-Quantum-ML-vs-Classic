from __future__ import annotations

import numpy as np


def sigmoid(z: np.ndarray) -> np.ndarray:
    z = np.clip(z, -50.0, 50.0)
    return 1.0 / (1.0 + np.exp(-z))


def binary_cross_entropy(y_true: np.ndarray, p_pred: np.ndarray, eps: float = 1e-9) -> float:
    """Binary cross-entropy for probabilities p in (0,1)."""
    y = y_true.astype(float)
    p = np.clip(p_pred.astype(float), eps, 1.0 - eps)
    return float(-np.mean(y * np.log(p) + (1.0 - y) * np.log(1.0 - p)))


def accuracy(y_true: np.ndarray, p_pred: np.ndarray, threshold: float = 0.5) -> float:
    """Accuracy using probability threshold."""
    y = y_true.astype(int)
    y_hat = (p_pred >= threshold).astype(int)
    return float(np.mean(y_hat == y))
