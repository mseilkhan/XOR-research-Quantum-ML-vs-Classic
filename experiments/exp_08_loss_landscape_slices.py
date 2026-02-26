from __future__ import annotations

"""
Experiment 8 — Loss landscape slices

Build 2D loss landscape around a trained solution theta*:
  L(theta* + alpha*d1 + beta*d2)

Models covered (fast + meaningful):
  - Linear
  - MLP(h=4)
  - VQC(L=1, analytic)  (shots=None)

Outputs per model:
  - outputs/npy/loss_landscape_<tag>_grid.npy      (Z matrix)
  - outputs/csv/loss_landscape_<tag>_grid.csv      (alpha,beta,loss rows)
  - outputs/figures/loss_landscape_<tag>.png
  - outputs/logs/exp_08_loss_landscape_slices_<run_id>.log 
  - outputs/metadata/exp_08_loss_landscape_slices_<run_id>.json 
"""

import csv
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Tuple

import numpy as np

from core.data.xor_dataset import make_split
from core.eval.metrics import binary_cross_entropy
from core.train.trainer import TrainConfig, Trainer
from core.utils.io import ensure_dir
from core.viz.loss_landscape import save_loss_landscape_figure

from experiments.models_factory import make_linear, make_mlp, make_vqc
from experiments.settings import (
    DATASET_SEED,
    SPLIT_SEED,
    BENCH_SIGMA,
    BENCH_N,
    MLP_HP,
    VQC_HP,
    CSV_DIR,
    FIG_DIR,
)

# Optional integrations (Fix 13/14)
try:
    from core.utils.run_context import create_run_context
    from core.utils.logging import setup_logger
except Exception:  # pragma: no cover
    create_run_context = None
    setup_logger = None

try:
    from core.utils.determinism import set_global_determinism
except Exception:  # pragma: no cover
    set_global_determinism = None

try:
    from tqdm import tqdm
except Exception:  # pragma: no cover
    tqdm = None

try:
    from core.utils.progress import SimpleProgress
except Exception:  # pragma: no cover
    SimpleProgress = None


NPY_DIR = "outputs/npy"
LOG_DIR = "outputs/logs"
META_DIR = "outputs/metadata"


@dataclass(frozen=True)
class LandscapeConfig:
    grid_points: int = 41          # 41x41 grid
    span: float = 1.5              # alpha,beta in [-span, span]
    direction_seed: int = 123      # seed to sample d1,d2
    train_seed: int = 0            # model init seed (single run for landscape)
    dataset: str = "B"


def _get_logger(exp_name: str):
    ensure_dir(LOG_DIR)
    ensure_dir(META_DIR)

    if create_run_context is not None and setup_logger is not None:
        settings = {
            "DATASET_SEED": DATASET_SEED,
            "SPLIT_SEED": SPLIT_SEED,
            "BENCH_SIGMA": float(BENCH_SIGMA),
            "BENCH_N": int(BENCH_N),
            "DETERMINISM_SEED": int(DATASET_SEED),
        }
        run_ctx = create_run_context(exp_name=exp_name, settings=settings, metadata_dir=Path(META_DIR))
        logger = setup_logger(exp_name, run_ctx.run_id, Path(LOG_DIR))
        logger.info(f"metadata: {run_ctx.metadata_path}")
        if set_global_determinism is not None:
            applied = set_global_determinism(int(DATASET_SEED))
            logger.info(f"Determinism enforced: {applied}")
        return logger

    class _Fallback:
        def info(self, msg: str):
            print(msg)

    if set_global_determinism is not None:
        set_global_determinism(int(DATASET_SEED))
    return _Fallback()


def _iter_progress(xs, desc: str):
    if tqdm is not None:
        return tqdm(list(xs), desc=desc)
    if SimpleProgress is not None:
        xs = list(xs)
        prog = SimpleProgress(total=len(xs), prefix=f"{desc}: ")
        def gen():
            for i, x in enumerate(xs, 1):
                prog.update(i, msg=str(x))
                yield x
            prog.done()
        return gen()
    return xs


def _model_to_vector(model) -> np.ndarray:
    """
    Convert model parameters into a 1D vector.
    Supported models in this repo:
      - Linear: model.w
      - MLP: W1,b1,W2,b2
      - VQC: model.params
    """
    if hasattr(model, "w"):
        w = np.array(model.w, dtype=float).ravel()
        return w

    if hasattr(model, "W1") and hasattr(model, "b1") and hasattr(model, "W2") and hasattr(model, "b2"):
        parts = [
            np.array(model.W1, dtype=float).ravel(),
            np.array(model.b1, dtype=float).ravel(),
            np.array(model.W2, dtype=float).ravel(),
            np.array(model.b2, dtype=float).ravel(),
        ]
        return np.concatenate(parts)

    if hasattr(model, "params"):
        return np.array(model.params, dtype=float).ravel()

    raise TypeError(f"Unsupported model type for vectorization: {type(model)}")


def _vector_to_model(model, theta: np.ndarray) -> None:
    """
    In-place set model parameters from a 1D vector.
    Must mirror _model_to_vector order.
    """
    t = np.array(theta, dtype=float).ravel()

    if hasattr(model, "w"):
        model.w = t.copy()
        return

    if hasattr(model, "W1") and hasattr(model, "b1") and hasattr(model, "W2") and hasattr(model, "b2"):
        i = 0
        W1 = model.W1
        b1 = model.b1
        W2 = model.W2
        b2 = model.b2

        nW1 = int(np.prod(W1.shape))
        nb1 = int(np.prod(b1.shape))
        nW2 = int(np.prod(W2.shape))
        nb2 = int(np.prod(b2.shape))

        model.W1 = t[i:i+nW1].reshape(W1.shape); i += nW1
        model.b1 = t[i:i+nb1].reshape(b1.shape); i += nb1
        model.W2 = t[i:i+nW2].reshape(W2.shape); i += nW2
        model.b2 = t[i:i+nb2].reshape(b2.shape); i += nb2
        return

    if hasattr(model, "params"):
        model.params = t.copy()
        return

    raise TypeError(f"Unsupported model type for devectorization: {type(model)}")


def _loss_on_split(model, X, y) -> float:
    p = model.predict_proba(X)
    return float(binary_cross_entropy(y, p))


def _orthonormal_directions(dim: int, seed: int) -> Tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(int(seed))
    d1 = rng.normal(size=(dim,))
    d2 = rng.normal(size=(dim,))
    d1 = d1 / (np.linalg.norm(d1) + 1e-12)
    # Gram-Schmidt
    d2 = d2 - np.dot(d2, d1) * d1
    d2 = d2 / (np.linalg.norm(d2) + 1e-12)
    return d1, d2


def _write_grid_csv_checkpoint(path: str, rows: List[Dict[str, float]], fieldnames: List[str]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    exists = os.path.exists(path)
    with open(path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        if not exists:
            w.writeheader()
        for r in rows:
            w.writerow(r)


def _compute_landscape(
    *,
    tag: str,
    model_factory: Callable[[int], object],
    train_cfg: TrainConfig,
    split,
    cfg: LandscapeConfig,
    logger,
) -> None:
    """
    Train model once, then compute 2D landscape grid.
    Saves incremental checkpoints per beta-row.
    """
    ensure_dir(CSV_DIR)
    ensure_dir(FIG_DIR)
    ensure_dir(NPY_DIR)

    logger.info(f"[INFO] Landscape start: {tag}")
    logger.info(f"[INFO] grid_points={cfg.grid_points}, span={cfg.span}, direction_seed={cfg.direction_seed}")

    # Train base model
    trainer = Trainer(train_cfg)
    model = model_factory(int(cfg.train_seed))
    trainer.fit(model, split.X_train, split.y_train, split.X_test, split.y_test)

    theta_star = _model_to_vector(model)
    d1, d2 = _orthonormal_directions(theta_star.size, cfg.direction_seed)

    alphas = np.linspace(-cfg.span, cfg.span, cfg.grid_points)
    betas = np.linspace(-cfg.span, cfg.span, cfg.grid_points)

    # Outputs
    npy_path = os.path.join(NPY_DIR, f"loss_landscape_{tag}_grid.npy")
    csv_path = os.path.join(CSV_DIR, f"loss_landscape_{tag}_grid.csv")
    fig_path = os.path.join(FIG_DIR, f"loss_landscape_{tag}.png")

    # If csv exists, we restart from scratch for correctness (simple policy)
    if os.path.exists(csv_path):
        os.remove(csv_path)

    Z = np.zeros((cfg.grid_points, cfg.grid_points), dtype=float)

    fieldnames = ["alpha", "beta", "loss"]

    # Compute row-by-row with checkpoints
    for j, beta in enumerate(_iter_progress(betas, desc=f"{tag} betas")):
        rows = []
        for i, alpha in enumerate(alphas):
            theta = theta_star + float(alpha) * d1 + float(beta) * d2
            _vector_to_model(model, theta)
            loss = _loss_on_split(model, split.X_train, split.y_train)
            Z[j, i] = loss
            rows.append({"alpha": float(alpha), "beta": float(beta), "loss": float(loss)})
        _write_grid_csv_checkpoint(csv_path, rows, fieldnames)

    # Save npy
    np.save(npy_path, Z)

    # Save figure
    A, B = np.meshgrid(alphas, betas)
    save_loss_landscape_figure(
        alphas=A,
        betas=B,
        Z=Z,
        title=f"Loss landscape — {tag} (Dataset B, σ={BENCH_SIGMA:.2f}, n={BENCH_N})",
        out_path=fig_path,
    )

    logger.info(f"[INFO] Saved: {csv_path}")
    logger.info(f"[INFO] Saved: {npy_path}")
    logger.info(f"[INFO] Saved: {fig_path}")
    logger.info(f"[INFO] Landscape done: {tag}")


def main():
    exp_name = "exp_08_loss_landscape_slices"
    logger = _get_logger(exp_name)

    cfg = LandscapeConfig()

    # Dataset B benchmark split
    split = make_split(
        "B",
        sigma=float(BENCH_SIGMA),
        n_per_cluster=int(BENCH_N),
        data_seed=int(DATASET_SEED),
        split_seed=int(SPLIT_SEED),
    )

    # Model suite (fast)
    linear_spec, linear_factory = make_linear()
    mlp_spec, mlp_factory = make_mlp(h=4)
    vqc_spec, vqc_factory = make_vqc(L=1, shots=None)  # analytic only

    # Train configs
    linear_cfg = TrainConfig(
        epochs=int(MLP_HP.epochs),
        lr=float(MLP_HP.lr),
    )
    mlp_cfg = TrainConfig(epochs=int(MLP_HP.epochs), lr=float(MLP_HP.lr))
    vqc_cfg = TrainConfig(epochs=int(VQC_HP.epochs), lr=float(VQC_HP.lr))

    # Compute landscapes
    _compute_landscape(tag="linear", model_factory=linear_factory, train_cfg=linear_cfg, split=split, cfg=cfg, logger=logger)
    _compute_landscape(tag="mlp_h4", model_factory=mlp_factory, train_cfg=mlp_cfg, split=split, cfg=cfg, logger=logger)
    _compute_landscape(tag="vqc_L1_analytic", model_factory=vqc_factory, train_cfg=vqc_cfg, split=split, cfg=cfg, logger=logger)


if __name__ == "__main__":
    main()