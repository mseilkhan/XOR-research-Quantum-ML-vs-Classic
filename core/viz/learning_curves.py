# core/viz/learning_curves.py
from __future__ import annotations

from typing import Dict, Optional, Sequence
import numpy as np
import matplotlib.pyplot as plt

from core.viz.style import apply_plot_style, FIGSIZE_SQUARE


def plot_train_test_curves(
    *,
    history: Dict[str, np.ndarray],
    title: str,
    ylabel: str,
    key_train: str,
    key_test: str,
    out_path: Optional[str] = None,
) -> None:
    apply_plot_style()
    tr = history[key_train]
    te = history[key_test]
    xs = np.arange(1, len(tr) + 1)

    fig = plt.figure(figsize=FIGSIZE_SQUARE)
    ax = plt.gca()
    ax.plot(xs, tr, label="train")
    ax.plot(xs, te, label="test")
    ax.set_title(title)
    ax.set_xlabel("epoch")
    ax.set_ylabel(ylabel)
    ax.legend()

    if out_path is not None:
        fig.savefig(out_path, bbox_inches="tight", facecolor="white", edgecolor="white", transparent=False)
    plt.close(fig)


def plot_test_loss_across_seeds(
    *,
    histories: Sequence[Dict[str, np.ndarray]],
    title: str,
    out_path: Optional[str] = None,
    alpha: float = 0.25,
) -> None:
    apply_plot_style()

    fig = plt.figure(figsize=FIGSIZE_SQUARE)
    ax = plt.gca()

    for h in histories:
        te = h["test_loss"]
        xs = np.arange(1, len(te) + 1)
        ax.plot(xs, te, alpha=float(alpha))

    ax.set_title(title)
    ax.set_xlabel("epoch")
    ax.set_ylabel("test BCE loss")

    if out_path is not None:
        fig.savefig(out_path, bbox_inches="tight", facecolor="white", edgecolor="white", transparent=False)
    plt.close(fig)
