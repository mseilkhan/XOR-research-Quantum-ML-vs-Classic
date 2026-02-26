from __future__ import annotations

"""
Experiment 7 — MLP width ablation 

Goal:
  Evaluate how MLP hidden width h impacts performance on Dataset B benchmark setting.

Protocol:
  - Dataset: B with sigma=BENCH_SIGMA and n_per_cluster=BENCH_N
  - Seeds: MODEL_SEEDS
  - Train config: MLP_HP (same training budget as baseline h=4)
Outputs:
  - outputs/csv/mlp_width_runs.csv        (raw per-seed)
  - outputs/csv/mlp_width_summary.csv     (mean/std per h)
  - outputs/figures/mlp_width_ablation_testacc.png
"""

import os
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import matplotlib.pyplot as plt

from core.data.xor_dataset import make_split
from core.train.trainer import TrainConfig, Trainer
from core.utils.io import append_csv_row, csv_contains_row, ensure_dir

from core.viz.style import apply_plot_style, FIGSIZE_WIDE

from experiments.models_factory import make_mlp
from experiments.settings import (
    DATASET_SEED,
    SPLIT_SEED,
    MODEL_SEEDS,
    BENCH_SIGMA,
    BENCH_N,
    MLP_HP,
    MLP_HS,
    CSV_DIR,
    FIG_DIR,
    LOG_DIR,
)

# ---- optional integrations (if present in your repo after Fix 13/14)
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


RUNS_CSV = os.path.join(CSV_DIR, "mlp_width_runs.csv")
SUM_CSV = os.path.join(CSV_DIR, "mlp_width_summary.csv")
FIG_PATH = os.path.join(FIG_DIR, "mlp_width_ablation_testacc.png")

RUN_FIELDS = [
    "experiment",
    "dataset",
    "sigma",
    "n_per_cluster",
    "h",
    "seed",
    "train_acc",
    "train_loss",
    "test_acc",
    "test_loss",
    "n_params",
]

SUM_FIELDS = [
    "experiment",
    "dataset",
    "sigma",
    "n_per_cluster",
    "h",
    "test_acc_mean",
    "test_acc_std",
    "test_loss_mean",
    "test_loss_std",
]


def _get_logger(exp_name: str):
    ensure_dir(LOG_DIR)

    if create_run_context is not None and setup_logger is not None:
        settings = {
            "DATASET_SEED": DATASET_SEED,
            "SPLIT_SEED": SPLIT_SEED,
            "MODEL_SEEDS": list(MODEL_SEEDS),
            "BENCH_SIGMA": BENCH_SIGMA,
            "BENCH_N": BENCH_N,
            "MLP_HP": {"epochs": MLP_HP.epochs, "lr": MLP_HP.lr},
            "MLP_HS": list(MLP_HS),
            "DETERMINISM_SEED": DATASET_SEED,
        }
        run_ctx = create_run_context(exp_name=exp_name, settings=settings)
        logger = setup_logger(exp_name, run_ctx.run_id, Path(LOG_DIR))
        logger.info(f"metadata: {run_ctx.metadata_path}")

        if set_global_determinism is not None:
            applied = set_global_determinism(DATASET_SEED)
            logger.info(f"Determinism enforced: {applied}")
        return logger

    # fallback logger
    class _Fallback:
        def info(self, msg: str):
            print(msg)
    if set_global_determinism is not None:
        set_global_determinism(DATASET_SEED)
    return _Fallback()


def _iter_progress(items, desc: str):
    if tqdm is not None:
        return tqdm(list(items), desc=desc)
    if SimpleProgress is not None:
        prog = SimpleProgress(total=len(list(items)), prefix=f"{desc}: ")
        def gen():
            xs = list(items)
            for i, x in enumerate(xs, 1):
                prog.update(i, msg=str(x))
                yield x
            prog.done()
        return gen()
    return items


def main():
    exp_name = "exp_07_mlp_width_ablation"
    logger = _get_logger(exp_name)

    ensure_dir(CSV_DIR)
    ensure_dir(FIG_DIR)

    logger.info(f"[INFO] MLP width ablation start: hs={MLP_HS}, seeds={MODEL_SEEDS}")
    logger.info(f"[INFO] Dataset B benchmark: sigma={BENCH_SIGMA}, n={BENCH_N}")

    # Dataset
    split = make_split(
        "B",
        sigma=float(BENCH_SIGMA),
        n_per_cluster=int(BENCH_N),
        data_seed=int(DATASET_SEED),
        split_seed=int(SPLIT_SEED),
    )

    train_cfg = TrainConfig(epochs=int(MLP_HP.epochs), lr=float(MLP_HP.lr))
    trainer = Trainer(train_cfg)

    # Collect raw results
    for h in _iter_progress(MLP_HS, desc="MLP width h"):
        spec, factory = make_mlp(h=int(h))

        for s in _iter_progress(MODEL_SEEDS, desc=f"{spec.name} seeds"):
            m = factory(int(s))
            trainer.fit(m, split.X_train, split.y_train, split.X_test, split.y_test)
            final = Trainer.final_metrics(m, split.X_train, split.y_train, split.X_test, split.y_test)

            row = {
                "experiment": exp_name,
                "dataset": "B",
                "sigma": float(BENCH_SIGMA),
                "n_per_cluster": int(BENCH_N),
                "h": int(h),
                "seed": int(s),
                "train_acc": float(final["train_acc"]),
                "train_loss": float(final["train_loss"]),
                "test_acc": float(final["test_acc"]),
                "test_loss": float(final["test_loss"]),
                "n_params": int(m.n_params()),
            }

            key = {"h": int(h), "seed": int(s), "sigma": float(BENCH_SIGMA), "n_per_cluster": int(BENCH_N)}
            if not csv_contains_row(RUNS_CSV, key):
                append_csv_row(RUNS_CSV, RUN_FIELDS, row)

    # Summaries
    # Read back runs CSV to compute mean/std robustly (single source of truth)
    import csv
    runs: List[Dict[str, Any]] = []
    with open(RUNS_CSV, "r", encoding="utf-8", newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            # filter only this experiment + this benchmark
            if row.get("experiment") != exp_name:
                continue
            if float(row["sigma"]) != float(BENCH_SIGMA):
                continue
            if int(row["n_per_cluster"]) != int(BENCH_N):
                continue
            runs.append(row)

    summaries: List[Dict[str, Any]] = []
    for h in MLP_HS:
        rs = [x for x in runs if int(x["h"]) == int(h)]
        test_acc = np.array([float(x["test_acc"]) for x in rs], dtype=float)
        test_loss = np.array([float(x["test_loss"]) for x in rs], dtype=float)

        summ = {
            "experiment": exp_name,
            "dataset": "B",
            "sigma": float(BENCH_SIGMA),
            "n_per_cluster": int(BENCH_N),
            "h": int(h),
            "test_acc_mean": float(test_acc.mean()),
            "test_acc_std": float(test_acc.std(ddof=0)),
            "test_loss_mean": float(test_loss.mean()),
            "test_loss_std": float(test_loss.std(ddof=0)),
        }

        key_s = {"h": int(h), "sigma": float(BENCH_SIGMA), "n_per_cluster": int(BENCH_N)}
        if not csv_contains_row(SUM_CSV, key_s):
            append_csv_row(SUM_CSV, SUM_FIELDS, summ)

        summaries.append(summ)

    # Plot (test acc mean ± std)
    apply_plot_style()
    hs = [int(s["h"]) for s in summaries]
    acc_m = [float(s["test_acc_mean"]) for s in summaries]
    acc_s = [float(s["test_acc_std"]) for s in summaries]

    fig = plt.figure(figsize=FIGSIZE_WIDE)
    ax = plt.gca()
    ax.errorbar(hs, acc_m, yerr=acc_s, fmt="-o", capsize=4)
    ax.set_title(f"MLP width ablation — Dataset B (σ={BENCH_SIGMA:.2f}, n={BENCH_N})")
    ax.set_xlabel("hidden width h")
    ax.set_ylabel("test accuracy (mean ± std)")
    ax.set_xticks(hs)
    ax.set_ylim(0.0, 1.05)

    fig.savefig(FIG_PATH, bbox_inches="tight", facecolor="white", edgecolor="white", transparent=False)
    plt.close(fig)

    logger.info(f"[INFO] Saved: {RUNS_CSV}")
    logger.info(f"[INFO] Saved: {SUM_CSV}")
    logger.info(f"[INFO] Saved: {FIG_PATH}")
    logger.info("[INFO] MLP width ablation done.")


if __name__ == "__main__":
    main()