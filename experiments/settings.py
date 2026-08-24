# experiments/settings.py
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import List, Optional


PROTOCOL_VERSION = "qmi-revision-p0-2026-08"


# -----------------------------
# Global methodology constants
# -----------------------------
DATASET_SEED = 42
SPLIT_SEED = 42
TRAIN_FRAC = 0.80

# Standard five-seed experiments and the main benchmark table
MODEL_SEEDS: List[int] = [0, 1, 2, 3, 4]

# Dedicated initialization-sensitivity analyses.
SEED_SENSITIVITY_SEEDS: List[int] = list(range(20))

# Expanded MLP width-ablation study reported in Appendix A.1 / Fig. 14.
WIDTH_ABLATION_SEEDS: List[int] = list(range(20))

NOISE_SWEEP: List[float] = [0.00, 0.05, 0.10, 0.20, 0.30]
SIZE_SWEEP: List[int] = [25, 50, 100, 250, 500]

# Main Dataset-B benchmark
BENCH_SIGMA = 0.10
BENCH_N = 100

# Representative settings for decision-boundary figures
DB_SIGMAS: List[float] = [0.10, 0.20]
DB_N_PER_CLUSTER = 100


# -----------------------------
# Training hyperparameters
# -----------------------------
@dataclass(frozen=True)
class TrainHP:
    epochs: int
    lr: float


LR_HP = TrainHP(
    epochs=800,
    lr=0.2,
)

MLP_HP = TrainHP(
    epochs=3000,
    lr=0.2,
)

VQC_HP = TrainHP(
    epochs=250,
    lr=0.2,
)


# -----------------------------
# Architecture / sampling grids
# -----------------------------
# Main MLP architecture used in the classical--quantum benchmark.
MLP_MAIN_H = 4

# Expanded capacity study reported in Appendix A.1 / Fig. 14.
MLP_HS: List[int] = [
    1,
    2,
    3,
    4,
    6,
    8,
    12,
    16,
    24,
    32,
]

# Representative widths shown in the BCE small-multiples panel of Fig. 14.
MLP_CURVE_HS: List[int] = [
    1,
    2,
    4,
    8,
    16,
    32,
]

VQC_LS: List[int] = [1, 2]

# None denotes analytic (shot-free) evaluation.
VQC_SHOTS_LIST: List[Optional[int]] = [None, 128, 1024]

# -----------------------------
# Optimizer settings
# -----------------------------
# All reported simulator-based training uses full-batch gradient descent.
OPTIMIZER = "gd"

# Retained only because the common training interface also supports Adam.
# They are inactive when OPTIMIZER == "gd".
ADAM_BETA1 = 0.9
ADAM_BETA2 = 0.999
ADAM_EPS = 1e-8


# -----------------------------
# Dataset C study
# -----------------------------
C_T_BENCH = 0.50
C_N_BENCH = 1000

C_T_SWEEP = [0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80]
C_N_SWEEP = [200, 500, 1000, 2000]


# -----------------------------
# Optional smoke mode for CI
# -----------------------------
SMOKE = os.getenv("XOR_SMOKE", "0") == "1"

if SMOKE:
    MODEL_SEEDS = [0]
    SEED_SENSITIVITY_SEEDS = [0]
    WIDTH_ABLATION_SEEDS = [0]

    LR_HP = TrainHP(
        epochs=min(5, LR_HP.epochs),
        lr=LR_HP.lr,
    )

    MLP_HP = TrainHP(
        epochs=min(5, MLP_HP.epochs),
        lr=MLP_HP.lr,
    )

    VQC_HP = TrainHP(
        epochs=min(2, VQC_HP.epochs),
        lr=VQC_HP.lr,
    )


# -----------------------------
# Output folders
# -----------------------------
# Smoke/CI runs must never overwrite publication-run artifacts.
OUT_ROOT = "outputs_smoke" if SMOKE else "outputs"

CSV_DIR = f"{OUT_ROOT}/csv"
FIG_DIR = f"{OUT_ROOT}/figures"
TABLE_DIR = f"{OUT_ROOT}/tables"
LOG_DIR = f"{OUT_ROOT}/logs"