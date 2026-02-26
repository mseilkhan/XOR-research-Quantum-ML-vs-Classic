from __future__ import annotations

import matplotlib as mpl

# Единый формат фигур
FIGSIZE_SQUARE = (6, 6)
FIGSIZE_WIDE = (6, 4)

def apply_plot_style() -> None:
    """
    Paper-like unified style.
    IMPORTANT: enforce white background (no transparency).
    """
    mpl.rcParams.update({
        "figure.dpi": 200,
        "savefig.dpi": 200,
        "font.size": 11,
        "axes.titlesize": 12,
        "axes.labelsize": 11,
        "legend.fontsize": 10,
        "axes.grid": True,
        "grid.alpha": 0.25,

        # единый фон/экспорт (Fix 15)
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
        "savefig.edgecolor": "white",
        "savefig.transparent": False,
    })