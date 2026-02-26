# experiments/settings.py
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence


# -----------------------------
# Global methodology constants
# -----------------------------
DATASET_SEED = 42
SPLIT_SEED = 42
TRAIN_FRAC = 0.80

MODEL_SEEDS: List[int] = [0, 1, 2, 3, 4]

# Fig.10 (seed sensitivity) uses an extended seed range
SEED_SENSITIVITY_SEEDS: List[int] = list(range(20))

NOISE_SWEEP: List[float] = [0.00, 0.05, 0.10, 0.20, 0.30]
SIZE_SWEEP: List[int] = [25, 50, 100, 250, 500]

# Main benchmark mode used in summary table
BENCH_SIGMA = 0.10
BENCH_N = 100

# Decision boundary representative settings
DB_SIGMAS = [0.10, 0.20]
DB_N_PER_CLUSTER = 100


# -----------------------------
# Training hyperparams (match your experiments)
# -----------------------------
@dataclass(frozen=True)
class TrainHP:
    epochs: int
    lr: float


LR_HP = TrainHP(epochs=800, lr=0.2)
MLP_HP = TrainHP(epochs=3000, lr=0.2)     # for h=4 baseline
VQC_HP = TrainHP(epochs=250, lr=0.2)

# MLP ablation h values
MLP_HS: List[int] = [1, 2, 4, 8]

# VQC regimes used in experiments
VQC_LS: List[int] = [1, 2]
VQC_SHOTS_LIST: List[Optional[int]] = [None, 128, 1024]   # analytic + finite shots

# ----------------------
# Optimizer settings
# ----------------------

OPTIMIZER = "gd"   # "gd" or "adam"

ADAM_BETA1 = 0.9
ADAM_BETA2 = 0.999
ADAM_EPS = 1e-8

# -----------------------------
# Output folders
# -----------------------------
OUT_ROOT = "outputs"
CSV_DIR = f"{OUT_ROOT}/csv"
FIG_DIR = f"{OUT_ROOT}/figures"
TABLE_DIR = f"{OUT_ROOT}/tables"
LOG_DIR = f"{OUT_ROOT}/logs"


# ---------------------
# Dataset C study
# ---------------------
C_T_BENCH = 0.50
C_N_BENCH = 1000

C_T_SWEEP = [0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80]
C_N_SWEEP = [200, 500, 1000, 2000]


import os

SMOKE = os.getenv("XOR_SMOKE", "0") == "1"
if SMOKE:
    # minimal settings for CI and sanity runs
    MODEL_SEEDS = [0]
    SEED_SENSITIVITY_SEEDS = [0]
    # reduce training budget
    MLP_HP = type(MLP_HP)(epochs=min(5, MLP_HP.epochs), lr=MLP_HP.lr)
    VQC_HP = type(VQC_HP)(epochs=min(2, VQC_HP.epochs), lr=VQC_HP.lr)
