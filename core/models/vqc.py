from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Dict, Any, Tuple
import numpy as np

from core.models.base import BaseBinaryClassifier, FitResult
from core.eval.metrics import binary_cross_entropy, accuracy
from core.utils.seeds import set_global_seed
from core.utils.timer import Timer
from core.optim.adam import adam_init, adam_step

@dataclass(frozen=True)
class VQCConfig:
    L: int                      # depth in {1,2}
    shots: Optional[int]        # None (analytic) or 128/1024
    device_name: str = "default.qubit"


class VQC2Q(BaseBinaryClassifier):
    """
    2-qubit VQC with angle encoding:
      x1 -> RX(pi*x1) on wire 0
      x2 -> RX(pi*x2) on wire 1

    Ansatz: L layers, each:
      Rot(params[0:3]) on wire 0
      Rot(params[3:6]) on wire 1
      CNOT(0->1)

    Measurement:
      m = <Z> on wire 0
      p(y=1|x) = (1 - m)/2

    Params: 6L
    Training: full-batch gradient descent via PennyLane autodiff.
    """
    def __init__(self, model_seed: int, cfg: VQCConfig):
        super().__init__(model_seed=model_seed)
        self.cfg = cfg
        self.L = int(cfg.L)
        self.shots = cfg.shots

        set_global_seed(self.model_seed)
        self.params = np.random.normal(0.0, 0.1, size=(6 * self.L,)).astype(float)

        # PennyLane is only imported here to keep classical code lightweight.
        import pennylane as qml  # noqa: WPS433
        import pennylane.numpy as pnp  # noqa: WPS433

        self.qml = qml
        self.pnp = pnp

        self.dev = qml.device(cfg.device_name, wires=2, shots=self.shots)

        @qml.qnode(self.dev, interface="autograd")
        def circuit(x, theta):
            # encoding
            qml.RX(np.pi * x[0], wires=0)
            qml.RX(np.pi * x[1], wires=1)

            # ansatz
            for l in range(self.L):
                off = 6 * l
                qml.Rot(theta[off + 0], theta[off + 1], theta[off + 2], wires=0)
                qml.Rot(theta[off + 3], theta[off + 4], theta[off + 5], wires=1)
                qml.CNOT(wires=[0, 1])

            return qml.expval(qml.PauliZ(0))

        self._circuit = circuit
        self._grad = qml.grad(self._loss_autograd)

    def n_params(self) -> int:
        return int(self.params.size)

    def _decision_f(self, X: np.ndarray) -> np.ndarray:
        """Return f(x)=<Z> in [-1,1]."""
        theta = self.pnp.array(self.params, requires_grad=False)
        out = []
        for x in X:
            m = self._circuit(self.pnp.array(x, requires_grad=False), theta)
            out.append(float(m))
        return np.array(out, dtype=float)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        m = self._decision_f(X)
        p = (1.0 - m) / 2.0
        return np.clip(p, 1e-9, 1.0 - 1e-9)

    def _loss_autograd(self, theta_pnp, X_pnp, y_pnp) -> float:
        """Autograd-compatible BCE loss over the dataset."""
        ms = []
        for i in range(X_pnp.shape[0]):
            ms.append(self._circuit(X_pnp[i], theta_pnp))
        m = self.pnp.stack(ms)
        p = (1.0 - m) / 2.0
        eps = 1e-9
        p = self.pnp.clip(p, eps, 1.0 - eps)
        y = y_pnp
        return -self.pnp.mean(y * self.pnp.log(p) + (1.0 - y) * self.pnp.log(1.0 - p))

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

        Xtr = self.pnp.array(X_train, requires_grad=False)
        ytr = self.pnp.array(y_train.astype(float), requires_grad=False)
        Xte = self.pnp.array(X_test, requires_grad=False)
        yte = self.pnp.array(y_test.astype(float), requires_grad=False)

        with Timer() as t:
            adam_state = adam_init(self.params)
            for _ in range(int(epochs)):
                theta = self.pnp.array(self.params, requires_grad=True)

                # gradient step
                g = self._grad(theta, Xtr, ytr)  # dLoss/dtheta
                g_np = np.array(g, dtype=float)

                if optimizer == "adam":
                    self.params, adam_state = adam_step(
                        param=self.params,
                        grad=g_np,
                        state=adam_state,
                        lr=lr,
                        beta1=adam_beta1,
                        beta2=adam_beta2,
                        eps=adam_eps,
                    )
                else:
                    theta = theta - lr * g
                    self.params = np.array(theta, dtype=float)

                # log train/test
                p_tr = self.predict_proba(np.array(Xtr))
                p_te = self.predict_proba(np.array(Xte))

                hist_train_loss.append(binary_cross_entropy(np.array(ytr), p_tr))
                hist_test_loss.append(binary_cross_entropy(np.array(yte), p_te))
                hist_train_acc.append(accuracy(np.array(ytr), p_tr))
                hist_test_acc.append(accuracy(np.array(yte), p_te))

        history = {
            "train_loss": np.array(hist_train_loss, dtype=float),
            "test_loss": np.array(hist_test_loss, dtype=float),
            "train_acc": np.array(hist_train_acc, dtype=float),
            "test_acc": np.array(hist_test_acc, dtype=float),
        }
        return FitResult(history=history, train_seconds=t.seconds, n_params=self.n_params())

    def to_config(self) -> Dict[str, Any]:
        cfg = super().to_config()
        cfg.update({
            "type": "vqc",
            "L": self.L,
            "shots": self.shots if self.shots is not None else "analytic",
            "encoding": "RX(pi*x1), RX(pi*x2)",
            "observable": "<Z0>",
            "n_params": self.n_params(),
            "device": self.cfg.device_name,
        })
        return cfg
