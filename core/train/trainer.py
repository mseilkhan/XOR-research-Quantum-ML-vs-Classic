from __future__ import annotations

from dataclasses import dataclass
from typing import Dict
import numpy as np

from core.models.base import BaseBinaryClassifier, FitResult
from core.eval.metrics import binary_cross_entropy, accuracy


@dataclass(frozen=True)
class TrainConfig:
    epochs: int
    lr: float
    optimizer: str = "gd"      # "gd" or "adam"
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    adam_eps: float = 1e-8


class Trainer:
    """
    Unified training wrapper.
    Real training loops are implemented inside model.fit() to keep each model correct and simple.
    This class standardizes config passing and output extraction.
    """
    def __init__(self, cfg: TrainConfig):
        self.cfg = cfg

    def fit(self, model: BaseBinaryClassifier, X_train, y_train, X_test, y_test) -> FitResult:
        return model.fit(
            X_train, y_train, X_test, y_test,
            epochs=int(self.cfg.epochs),
            lr=float(self.cfg.lr),
            optimizer=str(self.cfg.optimizer),
            adam_beta1=float(self.cfg.adam_beta1),
            adam_beta2=float(self.cfg.adam_beta2),
            adam_eps=float(self.cfg.adam_eps),
        )

    @staticmethod
    def final_metrics(model: BaseBinaryClassifier, X_train, y_train, X_test, y_test) -> Dict[str, float]:
        """Compute final train/test acc and loss from current model state."""
        p_tr = model.predict_proba(X_train)
        p_te = model.predict_proba(X_test)
        return {
            "train_acc": accuracy(y_train, p_tr),
            "test_acc": accuracy(y_test, p_te),
            "train_loss": binary_cross_entropy(y_train, p_tr),
            "test_loss": binary_cross_entropy(y_test, p_te),
        }