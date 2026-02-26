# core/viz/robustness.py
from __future__ import annotations

from typing import List, Optional
import numpy as np
import matplotlib.pyplot as plt

from core.viz.style import apply_plot_style, FIGSIZE_SQUARE


def plot_mean_std_curve(
    *,
    xs: List[float],
    ys_mean: List[float],
    ys_std: List[float],
    title: str,
    xlabel: str,
    ylabel: str,
    label: str,
    out_path: Optional[str] = None,
) -> None:
    apply_plot_style()
    x = np.array(xs, dtype=float)
    m = np.array(ys_mean, dtype=float)
    s = np.array(ys_std, dtype=float)

    fig = plt.figure(figsize=FIGSIZE_SQUARE)
    ax = plt.gca()

    ax.plot(x, m, label=label)
    ax.fill_between(x, m - s, m + s, alpha=0.2)

    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend()

    if out_path is not None:
        fig.savefig(out_path, bbox_inches="tight", facecolor="white", edgecolor="white", transparent=False)
    plt.close(fig)
