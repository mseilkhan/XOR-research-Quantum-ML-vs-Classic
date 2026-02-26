from __future__ import annotations

import numpy as np

from core.models.base import BaseBinaryClassifier, FitResult
from core.eval.metrics import sigmoid, binary_cross_entropy, accuracy
from core.utils.seeds import set_global_seed
from core.utils.timer import Timer
from core.optim.adam import adam_init, adam_step

class LogisticRegression2D(BaseBinaryClassifier):
    """
    Linear classifier: logistic regression with 2 features + bias => 3 params.
    Uses full-batch SGD (per epoch) for simplicity and reproducibility.
    """

    def __init__(self, model_seed: int):
        super().__init__(model_seed=model_seed)
        set_global_seed(self.model_seed)
        # params: w1, w2, b
        self.w = np.random.normal(0.0, 0.1, size=(3,)).astype(float)

    def n_params(self) -> int:
        return 3

    def _logits(self, X: np.ndarray) -> np.ndarray:
        return X @ self.w[:2] + self.w[2]

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return sigmoid(self._logits(X))

    def fit(
            self, X_train, y_train, X_test, y_test, *,
            epochs: int,
            lr: float,
            optimizer: str = "gd",
            adam_beta1: float = 0.9,
            adam_beta2: float = 0.999,
            adam_eps: float = 1e-8,
    ) -> FitResult:
        hist_train_loss = []
        hist_test_loss = []
        hist_train_acc = []
        hist_test_acc = []

        with Timer() as t:
            adam_state = adam_init(self.w)
            for _ in range(int(epochs)):
                # forward
                p = self.predict_proba(X_train)
                # gradients (BCE wrt logits -> (p - y))
                diff = (p - y_train).astype(float)  # shape (n,)
                grad_w = (X_train.T @ diff) / X_train.shape[0]  # (2,)
                grad_b = float(np.mean(diff))
                # update
                g = np.array([grad_w[0], grad_w[1], grad_b], dtype=float)
                if optimizer == "adam":
                    self.w, adam_state = adam_step(
                        param=self.w,
                        grad=g,
                        state=adam_state,
                        lr=lr,
                        beta1=adam_beta1,
                        beta2=adam_beta2,
                        eps=adam_eps,
                    )
                else:
                    self.w[:2] -= lr * grad_w
                    self.w[2] -= lr * grad_b

                # log train/test
                p_tr = self.predict_proba(X_train)
                p_te = self.predict_proba(X_test)

                hist_train_loss.append(binary_cross_entropy(y_train, p_tr))
                hist_test_loss.append(binary_cross_entropy(y_test, p_te))
                hist_train_acc.append(accuracy(y_train, p_tr))
                hist_test_acc.append(accuracy(y_test, p_te))

        history = {
            "train_loss": np.array(hist_train_loss, dtype=float),
            "test_loss": np.array(hist_test_loss, dtype=float),
            "train_acc": np.array(hist_train_acc, dtype=float),
            "test_acc": np.array(hist_test_acc, dtype=float),
        }
        return FitResult(history=history, train_seconds=t.seconds, n_params=self.n_params())

    def to_config(self):
        cfg = super().to_config()
        cfg.update({"type": "logreg", "n_params": 3})
        return cfg
