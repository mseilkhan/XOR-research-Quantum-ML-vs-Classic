from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass
class AdamState:
    t: int
    m: np.ndarray
    v: np.ndarray


def adam_init(param: np.ndarray) -> AdamState:
    p = np.array(param, dtype=float)
    return AdamState(t=0, m=np.zeros_like(p), v=np.zeros_like(p))


def adam_step(
    *,
    param: np.ndarray,
    grad: np.ndarray,
    state: AdamState,
    lr: float,
    beta1: float,
    beta2: float,
    eps: float,
) -> tuple[np.ndarray, AdamState]:
    """
    One Adam update step (bias-corrected).
    """
    g = np.array(grad, dtype=float)
    p = np.array(param, dtype=float)

    state.t += 1
    state.m = beta1 * state.m + (1.0 - beta1) * g
    state.v = beta2 * state.v + (1.0 - beta2) * (g * g)

    m_hat = state.m / (1.0 - beta1 ** state.t)
    v_hat = state.v / (1.0 - beta2 ** state.t)

    p = p - lr * m_hat / (np.sqrt(v_hat) + eps)
    return p, state