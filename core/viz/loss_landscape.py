from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt

from core.viz.style import apply_plot_style


def save_loss_landscape_figure(
    *,
    alphas: np.ndarray,
    betas: np.ndarray,
    Z: np.ndarray,
    title: str,
    out_path: str | Path,
) -> None:
    """
    Save a paper-like 2D loss landscape figure with white background.
    """
    apply_plot_style()
    fig = plt.figure(figsize=(6, 5))
    ax = plt.gca()

    cf = ax.contourf(alphas, betas, Z, levels=30)
    cbar = plt.colorbar(cf, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("loss")

    ax.set_title(title)
    ax.set_xlabel("alpha")
    ax.set_ylabel("beta")

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight", facecolor="white", edgecolor="white", transparent=False)
    plt.close(fig)