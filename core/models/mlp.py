from __future__ import annotations

import numpy as np

from core.models.base import BaseBinaryClassifier, FitResult
from core.eval.metrics import sigmoid, binary_cross_entropy, accuracy
from core.utils.seeds import set_global_seed
from core.utils.timer import Timer
from core.optim.adam import adam_init, adam_step

class MLP1Hidden(BaseBinaryClassifier):
    """
    MLP with 1 hidden layer:
    - hidden: sigmoid
    - output: sigmoid
    Train: full-batch SGD (per epoch), fixed epochs across experiments.
    Hidden units h in {1,2,4,8} (ablation).
    """

    def __init__(self, model_seed: int, h: int):
        super().__init__(model_seed=model_seed)
        self.h = int(h)
        set_global_seed(self.model_seed)

        # Xavier-like small init
        self.W1 = np.random.normal(0.0, 0.5, size=(2, self.h)).astype(float)
        self.b1 = np.zeros((self.h,), dtype=float)
        self.W2 = np.random.normal(0.0, 0.5, size=(self.h, 1)).astype(float)
        self.b2 = np.zeros((1,), dtype=float)

    def n_params(self) -> int:
        return int(self.W1.size + self.b1.size + self.W2.size + self.b2.size)

    def _forward(self, X: np.ndarray):
        z1 = X @ self.W1 + self.b1
        a1 = sigmoid(z1)
        z2 = a1 @ self.W2 + self.b2  # (n,1)
        p = sigmoid(z2).reshape(-1)
        return a1, p

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        _, p = self._forward(X)
        return p

    def fit(
            self, X_train, y_train, X_test, y_test, *,
            epochs: int,
            lr: float,
            optimizer: str = "gd",
            adam_beta1: float = 0.9,
            adam_beta2: float = 0.999,
            adam_eps: float = 1e-8,
    ) -> FitResult:
        hist_train_loss, hist_test_loss = [], []
        hist_train_acc, hist_test_acc = [], []

        y_train = y_train.astype(float).reshape(-1)
        y_test = y_test.astype(float).reshape(-1)

        with Timer() as t:
            adam_W1 = adam_init(self.W1)
            adam_b1 = adam_init(self.b1)
            adam_W2 = adam_init(self.W2)
            adam_b2 = adam_init(self.b2)
            for _ in range(int(epochs)):
                # forward
                a1, p = self._forward(X_train)

                # gradients
                # BCE with sigmoid output: dL/dz2 = (p - y)
                dz2 = (p - y_train).reshape(-1, 1)  # (n,1)
                dW2 = (a1.T @ dz2) / X_train.shape[0]  # (h,1)
                db2 = np.mean(dz2, axis=0)  # (1,)

                da1 = dz2 @ self.W2.T  # (n,h)
                dz1 = da1 * a1 * (1.0 - a1)  # sigmoid'
                dW1 = (X_train.T @ dz1) / X_train.shape[0]  # (2,h)
                db1 = np.mean(dz1, axis=0)  # (h,)

                # update
                if optimizer == "adam":
                    self.W2, adam_W2 = adam_step(param=self.W2, grad=dW2, state=adam_W2, lr=lr, beta1=adam_beta1,
                                                 beta2=adam_beta2, eps=adam_eps)
                    self.b2, adam_b2 = adam_step(param=self.b2, grad=db2, state=adam_b2, lr=lr, beta1=adam_beta1,
                                                 beta2=adam_beta2, eps=adam_eps)
                    self.W1, adam_W1 = adam_step(param=self.W1, grad=dW1, state=adam_W1, lr=lr, beta1=adam_beta1,
                                                 beta2=adam_beta2, eps=adam_eps)
                    self.b1, adam_b1 = adam_step(param=self.b1, grad=db1, state=adam_b1, lr=lr, beta1=adam_beta1,
                                                 beta2=adam_beta2, eps=adam_eps)
                else:
                    self.W2 -= lr * dW2
                    self.b2 -= lr * db2
                    self.W1 -= lr * dW1
                    self.b1 -= lr * db1

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
        cfg.update({"type": "mlp", "h": self.h, "n_params": self.n_params()})
        return cfg
