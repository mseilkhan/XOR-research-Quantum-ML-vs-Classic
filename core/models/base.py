from __future__ import annotations

from dataclasses import dataclass
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
import numpy as np


@dataclass
class FitResult:
    """Training history (epoch-wise)."""
    history: Dict[str, np.ndarray]  # keys: train_loss, test_loss, train_acc, test_acc
    train_seconds: float
    n_params: int


class BaseBinaryClassifier(ABC):
    """Common interface for all binary classifiers in this project."""

    def __init__(self, model_seed: int):
        self.model_seed = int(model_seed)

    @abstractmethod
    def n_params(self) -> int:
        """Number of trainable parameters."""
        raise NotImplementedError

    @abstractmethod
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return p(y=1|x) for each row in X."""
        raise NotImplementedError

    @abstractmethod
    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
        *,
        epochs: int,
        lr: float,
        optimizer: str = "gd",
        adam_beta1: float = 0.9,
        adam_beta2: float = 0.999,
        adam_eps: float = 1e-8,
    ) -> FitResult:
        """Train model and return FitResult with epoch-wise logs."""
        raise NotImplementedError

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        p = self.predict_proba(X)
        return (p >= threshold).astype(int)

    def to_config(self) -> Dict[str, Any]:
        """Minimal config snapshot (for exp_settings_models tables)."""
        return {"model": self.__class__.__name__, "model_seed": self.model_seed}
