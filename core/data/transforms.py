from __future__ import annotations

import numpy as np


def add_gaussian_noise(X: np.ndarray, sigma: float, seed: int) -> np.ndarray:
    """Utility if you need noise injection outside dataset B generation."""
    rng = np.random.default_rng(seed)
    return (X + rng.normal(0.0, float(sigma), size=X.shape)).astype(float)
