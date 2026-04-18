from __future__ import annotations

from typing import Optional, Sequence, Tuple
import numpy as np
import matplotlib.pyplot as plt

from core.viz.style import apply_plot_style
from pathlib import Path

def _is_vqc_model(model) -> bool:
    name = getattr(model, "name", "")
    return isinstance(name, str) and name.startswith("VQC")

def make_grid(xlim=(0.0, 1.0), ylim=(0.0, 1.0), n: int = 250) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    xs = np.linspace(xlim[0], xlim[1], int(n))
    ys = np.linspace(ylim[0], ylim[1], int(n))
    XX, YY = np.meshgrid(xs, ys)
    grid = np.stack([XX.ravel(), YY.ravel()], axis=1)
    return XX, YY, grid

def _expand_limits(xlim, ylim, factor: float = 0.5):
    """
    Expand limits by a fraction of range on each side.
    factor=0.5 means +50% range on both sides.
    """
    x0, x1 = xlim
    y0, y1 = ylim
    dx = (x1 - x0) * factor
    dy = (y1 - y0) * factor
    return (x0 - dx, x1 + dx), (y0 - dy, y1 + dy)

def plot_decision_function(
    *,
    model,
    X: np.ndarray,
    y: np.ndarray,
    title: str,
    out_path: Optional[str] = None,
    xlim=(0.0, 1.0),
    ylim=(0.0, 1.0),
    n_grid: int = 250,
    show_boundary: bool = True,
    vmin: float = 0.0,
    vmax: float = 1.0,
) -> None:
    """
    Decision function visualization using probability p(y=1|x).
    RULE (Fix 15):
      - Classical (Linear/MLP): show only decision boundary line(s), no heatmap.
      - VQC: show smooth heatmap + boundary, like in the paper.
    """
    apply_plot_style()
    # For classical models compute boundary on expanded area to get long lines
    if _is_vqc_model(model):
        xlim_g, ylim_g = xlim, ylim
    else:
        xlim_g, ylim_g = _expand_limits(xlim, ylim, factor=0.8)

    XX, YY, grid = make_grid(xlim=xlim_g, ylim=ylim_g, n=n_grid)
    P = model.predict_proba(grid).reshape(XX.shape)

    fig = plt.figure(figsize=(6, 6))
    ax = plt.gca()

    is_vqc = _is_vqc_model(model)

    # --- Background visualization ---
    if is_vqc:
        im = ax.imshow(
            P, origin="lower",
            extent=(xlim[0], xlim[1], ylim[0], ylim[1]),
            vmin=vmin, vmax=vmax, aspect="equal",
            alpha=0.95,
        )
    else:
        im = None  # no heatmap for classical models

    # --- Boundary at p=0.5 ---
    if show_boundary:
        if is_vqc:
            ax.contour(XX, YY, P, levels=[0.5], linewidths=2.2)
        else:
            # Classical: more stable long boundary using binary mask contour
            M = (P >= 0.5).astype(float)
            ax.contour(XX, YY, M, levels=[0.5], linewidths=2.6)

    # scatter data
    ax.scatter(X[:, 0], X[:, 1], c=y, s=22, edgecolors="k", linewidths=0.3)

    ax.set_title(title)
    ax.set_xlabel("x1")
    ax.set_ylabel("x2")
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)

    # colorbar only for VQC heatmap
    if is_vqc and im is not None:
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    if out_path is not None:
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, bbox_inches="tight", facecolor="white", edgecolor="white", transparent=False)
    plt.close(fig)

def plot_decision_boundary_seed_background(
    *,
    models: Sequence,
    X: np.ndarray,
    y: np.ndarray,
    title: str,
    out_path: Optional[str] = None,
    xlim=(0.0, 1.0),
    ylim=(0.0, 1.0),
    n_grid: int = 250,
) -> None:
    """
    "Representative background across seeds":
    RULE (Fix 15):
      - Classical (Linear/MLP): plot only multiple boundary lines (no heatmap).
      - VQC: plot heatmap from first model + background boundaries from all seeds + colorbar.
    """
    apply_plot_style()
    is_vqc = _is_vqc_model(models[0])

    # For classical models compute boundary on expanded area to get long lines
    if is_vqc:
        xlim_g, ylim_g = xlim, ylim
    else:
        xlim_g, ylim_g = _expand_limits(xlim, ylim, factor=0.8)

    XX, YY, grid = make_grid(xlim=xlim_g, ylim=ylim_g, n=n_grid)

    fig = plt.figure(figsize=(6, 6))
    ax = plt.gca()

    # Decide based on first model name
    is_vqc = _is_vqc_model(models[0])

    # background contours per seed
    # Classical (Linear/MLP): do NOT draw multi-seed spaghetti. Draw only one main boundary.
    if not is_vqc:
        P_main = models[0].predict_proba(grid).reshape(XX.shape)
        M = (P_main >= 0.5).astype(float)
        ax.contour(XX, YY, M, levels=[0.5], linewidths=2.8, alpha=0.95)
    else:
        # VQC: keep light multi-seed background
        for m in models:
            P = m.predict_proba(grid).reshape(XX.shape)
            ax.contour(XX, YY, P, levels=[0.5], linewidths=1.0, alpha=0.18)
    im = None
    if is_vqc:
        # main heatmap from first model (smooth, as in paper)
        P0 = models[0].predict_proba(grid).reshape(XX.shape)
        im = ax.imshow(
            P0, origin="lower",
            extent=(xlim[0], xlim[1], ylim[0], ylim[1]),
            vmin=0.0, vmax=1.0, aspect="equal",
            alpha=0.95,
        )
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # data points
    ax.scatter(X[:, 0], X[:, 1], c=y, s=38, edgecolors="k", linewidths=0.5)
    ax.set_title(title)
    ax.set_xlabel("x1")
    ax.set_ylabel("x2")

    # Make Dataset A corner points visible (small margin around [0,1])
    if xlim == (0.0, 1.0) and ylim == (0.0, 1.0) and X.shape[0] <= 10:
        ax.set_xlim(-0.05, 1.05)
        ax.set_ylim(-0.05, 1.05)


    if out_path is not None:
        fig.savefig(out_path, bbox_inches="tight", facecolor="white", edgecolor="white", transparent=False)
    plt.close(fig)