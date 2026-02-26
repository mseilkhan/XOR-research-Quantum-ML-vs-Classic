from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple, Literal, Optional
import numpy as np


DatasetName = Literal["A", "B", "C"]


@dataclass(frozen=True)
class DatasetSplit:
    """Train/test split container."""
    X_train: np.ndarray
    y_train: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray

    @property
    def n_train(self) -> int:
        return int(self.X_train.shape[0])

    @property
    def n_test(self) -> int:
        return int(self.X_test.shape[0])


def xor_labels(X: np.ndarray) -> np.ndarray:
    """XOR label for canonical/clustered XOR: y = x1 XOR x2 (with x in {0,1})."""
    x1 = X[:, 0].round().astype(int)
    x2 = X[:, 1].round().astype(int)
    y = (x1 ^ x2).astype(float)
    return y


def make_dataset_A(seed: int = 42) -> Tuple[np.ndarray, np.ndarray]:
    """Dataset A: 4 canonical points of XOR."""
    _ = seed  # seed unused; deterministic
    X = np.array([[0.0, 0.0],
                  [0.0, 1.0],
                  [1.0, 0.0],
                  [1.0, 1.0]], dtype=float)
    y = xor_labels(X)
    return X, y


def make_dataset_B(
    sigma: float,
    n_per_cluster: int,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Dataset B: 4 Gaussian clusters around XOR corners with additive Gaussian noise.
    Corners: (0,0), (0,1), (1,0), (1,1)
    """
    rng = np.random.default_rng(seed)
    corners = np.array([[0.0, 0.0],
                        [0.0, 1.0],
                        [1.0, 0.0],
                        [1.0, 1.0]], dtype=float)

    X_list = []
    for c in corners:
        Xc = c[None, :] + rng.normal(loc=0.0, scale=float(sigma), size=(int(n_per_cluster), 2))
        X_list.append(Xc)

    X = np.vstack(X_list).astype(float)
    # labels by XOR of cluster identity (using corner rounding)
    y = xor_labels(X.clip(0.0, 1.0))  # clip just for stable rounding under noise
    return X, y


def make_dataset_C(
    n: int,
    t: float,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Dataset C: uniform sampling in [0,1]^2 with threshold-based XOR.
    y=1 iff exactly one coordinate > t.
    """
    rng = np.random.default_rng(seed)
    X = rng.uniform(0.0, 1.0, size=(int(n), 2)).astype(float)
    c1 = (X[:, 0] > t).astype(int)
    c2 = (X[:, 1] > t).astype(int)
    y = (c1 ^ c2).astype(float)
    return X, y


def train_test_split_fixed(
    X: np.ndarray,
    y: np.ndarray,
    train_ratio: float = 0.8,
    seed: int = 42,
) -> DatasetSplit:
    """Fixed split 80/20 with seed=42 as required."""
    assert X.ndim == 2 and X.shape[1] == 2
    assert y.ndim == 1 and y.shape[0] == X.shape[0]

    rng = np.random.default_rng(seed)
    n = X.shape[0]
    idx = np.arange(n)
    rng.shuffle(idx)

    n_train = int(round(train_ratio * n))
    train_idx = idx[:n_train]
    test_idx = idx[n_train:]

    return DatasetSplit(
        X_train=X[train_idx],
        y_train=y[train_idx],
        X_test=X[test_idx],
        y_test=y[test_idx],
    )


def make_split(
    name: DatasetName,
    *,
    sigma: Optional[float] = None,
    n_per_cluster: Optional[int] = None,
    n: Optional[int] = None,
    t: Optional[float] = None,
    data_seed: int = 42,
    split_seed: int = 42,
) -> DatasetSplit:
    """Factory for dataset splits with the required fixed seeds."""
    if name == "A":
        X, y = make_dataset_A(seed=data_seed)
    elif name == "B":
        assert sigma is not None and n_per_cluster is not None
        X, y = make_dataset_B(sigma=float(sigma), n_per_cluster=int(n_per_cluster), seed=data_seed)
    elif name == "C":
        assert n is not None and t is not None
        X, y = make_dataset_C(n=int(n), t=float(t), seed=data_seed)
    else:
        raise ValueError(f"Unknown dataset name: {name}")

    return train_test_split_fixed(X, y, train_ratio=0.8, seed=split_seed)
