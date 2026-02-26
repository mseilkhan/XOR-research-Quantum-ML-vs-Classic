from __future__ import annotations

"""
Experiment 2 — Learning behavior
- MLP(h=4): train/test loss, train/test acc, test loss across seeds
- VQC:
  - L=1 analytic: train/test loss
  - L=1 shots=128: train/test loss
  - L=2 analytic: test loss across seeds
Artifacts:
- square learning-curve figures
- CSV histories:
  - seed0 representative (existing)
  - raw per-seed-per-epoch combined CSV (added by Fix 12)
"""

import os
from typing import Dict, List, Any, Optional

import numpy as np
from tqdm import tqdm

from core.data.xor_dataset import make_split
from core.train.trainer import TrainConfig, Trainer
from core.viz.learning_curves import (
    plot_train_test_curves,
    plot_test_loss_across_seeds,
)

from pathlib import Path
from core.utils.run_context import create_run_context
from core.utils.logging import setup_logger
from core.utils.determinism import set_global_determinism

from experiments.settings import (
    DATASET_SEED,
    SPLIT_SEED,
    MODEL_SEEDS,
    FIG_DIR,
    CSV_DIR,
    MLP_HP,
    VQC_HP,
)
from experiments.utils import ensure_output_dirs, save_records_csv, tag_shots
from experiments.models_factory import make_mlp, make_vqc
from experiments.settings import OPTIMIZER, ADAM_BETA1, ADAM_BETA2, ADAM_EPS

RAW_HISTORY_CSV = os.path.join(CSV_DIR, "exp02_learning_behavior_raw_history.csv")


def history_to_rows(history: Dict[str, np.ndarray], meta: Dict[str, Any]) -> List[dict]:
    rows = []
    epochs = len(history["train_loss"])
    for i in range(epochs):
        rows.append(
            {
                **meta,
                "epoch": int(i + 1),
                "train_loss": float(history["train_loss"][i]),
                "test_loss": float(history["test_loss"][i]),
                "train_acc": float(history["train_acc"][i]),
                "test_acc": float(history["test_acc"][i]),
            }
        )
    return rows


def main():
    exp_name = "exp_02_learning_behavior"
    settings = {
        "DATASET_SEED": DATASET_SEED,
        "SPLIT_SEED": SPLIT_SEED,
        "MODEL_SEEDS": list(MODEL_SEEDS),
        "OPTIMIZER": OPTIMIZER,
        "ADAM_BETA1": ADAM_BETA1,
        "ADAM_BETA2": ADAM_BETA2,
        "ADAM_EPS": ADAM_EPS,
        "DETERMINISM_SEED": DATASET_SEED
    }
    run_ctx = create_run_context(exp_name=exp_name, settings=settings)
    logger = setup_logger(exp_name, run_ctx.run_id, Path("outputs/logs"))
    logger.info(f"metadata: {run_ctx.metadata_path}")
    ensure_output_dirs(FIG_DIR, CSV_DIR)

    applied_seeds = set_global_determinism(DATASET_SEED)
    logger.info(f"Determinism enforced: {applied_seeds}")

    # Use benchmark dataset for learning curves (Dataset B, σ=0.10, n=100)
    split = make_split("B", sigma=0.10, n_per_cluster=100, data_seed=DATASET_SEED, split_seed=SPLIT_SEED)
    trainer_mlp = Trainer(TrainConfig(MLP_HP.epochs,
                                      MLP_HP.lr,
                                      optimizer=OPTIMIZER,
                                      adam_beta1=ADAM_BETA1,
                                      adam_beta2=ADAM_BETA2,
                                      adam_eps=ADAM_EPS,
                                      ))
    trainer_vqc = Trainer(TrainConfig(VQC_HP.epochs,
                                      VQC_HP.lr,
                                      optimizer=OPTIMIZER,
                                      adam_beta1=ADAM_BETA1,
                                      adam_beta2=ADAM_BETA2,
                                      adam_eps=ADAM_EPS,
                                      ))

    raw_rows: List[Dict[str, Any]] = []

    # ---------------- MLP(h=4) ----------------
    mlp_spec, mlp_factory = make_mlp(h=4)

    # representative run = seed 0
    mlp0 = mlp_factory(0)
    res0 = trainer_mlp.fit(mlp0, split.X_train, split.y_train, split.X_test, split.y_test)

    plot_train_test_curves(
        history=res0.history,
        title=f"{mlp_spec.name} — Train/Test Loss (Dataset B σ=0.10, n=100)",
        ylabel="BCE loss",
        key_train="train_loss",
        key_test="test_loss",
        out_path=os.path.join(FIG_DIR, "lc_mlp_h4_loss.png"),
    )
    plot_train_test_curves(
        history=res0.history,
        title=f"{mlp_spec.name} — Train/Test Accuracy (Dataset B σ=0.10, n=100)",
        ylabel="accuracy",
        key_train="train_acc",
        key_test="test_acc",
        out_path=os.path.join(FIG_DIR, "lc_mlp_h4_acc.png"),
    )

    # test loss across seeds + raw per-seed-per-epoch (Fix 12)
    histories = []
    for s in tqdm(list(MODEL_SEEDS), desc="MLP(h=4) seeds"):
        m = mlp_factory(s)
        r = trainer_mlp.fit(m, split.X_train, split.y_train, split.X_test, split.y_test)
        histories.append(r.history)

        raw_rows.extend(
            history_to_rows(
                r.history,
                {
                    "experiment": "exp_02_learning_behavior",
                    "dataset": "B",
                    "model": mlp_spec.name,
                    "seed": int(s),
                    "L": None,
                    "shots": None,
                },
            )
        )

    plot_test_loss_across_seeds(
        histories=histories,
        title=f"{mlp_spec.name} — Test Loss Across Seeds (Dataset B σ=0.10, n=100)",
        out_path=os.path.join(FIG_DIR, "lc_mlp_h4_testloss_seeds.png"),
    )

    # save representative history (existing artifact)
    save_records_csv(
        os.path.join(CSV_DIR, "lc_mlp_h4_seed0.csv"),
        history_to_rows(res0.history, {"model": mlp_spec.name, "seed": "0"}),
    )

    # ---------------- VQC (L=1) analytic & 128 shots ----------------
    for shots in tqdm([None, 128], desc="VQC(L=1) shots"):
        vqc_spec, vqc_factory = make_vqc(L=1, shots=shots)
        v = vqc_factory(0)
        r = trainer_vqc.fit(v, split.X_train, split.y_train, split.X_test, split.y_test)

        plot_train_test_curves(
            history=r.history,
            title=f"{vqc_spec.name} — Train/Test Loss (Dataset B σ=0.10, n=100)",
            ylabel="BCE loss",
            key_train="train_loss",
            key_test="test_loss",
            out_path=os.path.join(FIG_DIR, f"lc_vqc_L1_{tag_shots(shots)}_loss.png"),
        )

        save_records_csv(
            os.path.join(CSV_DIR, f"lc_vqc_L1_{tag_shots(shots)}_seed0.csv"),
            history_to_rows(r.history, {"model": vqc_spec.name, "seed": "0"}),
        )

        # Fix 12 raw history rows (seed 0 for these modes, as in the paper figures)
        raw_rows.extend(
            history_to_rows(
                r.history,
                {
                    "experiment": "exp_02_learning_behavior",
                    "dataset": "B",
                    "model": vqc_spec.name,
                    "seed": 0,
                    "L": 1,
                    "shots": shots,
                },
            )
        )

    # ---------------- VQC (L=2) analytic: test loss across seeds ----------------
    vqc2_spec, vqc2_factory = make_vqc(L=2, shots=None)
    histories_vqc2 = []
    for s in tqdm(list(MODEL_SEEDS), desc="VQC(L=2) seeds"):
        v = vqc2_factory(s)
        r = trainer_vqc.fit(v, split.X_train, split.y_train, split.X_test, split.y_test)
        histories_vqc2.append(r.history)

        # Fix 12 raw per-seed-per-epoch
        raw_rows.extend(
            history_to_rows(
                r.history,
                {
                    "experiment": "exp_02_learning_behavior",
                    "dataset": "B",
                    "model": vqc2_spec.name,
                    "seed": int(s),
                    "L": 2,
                    "shots": None,
                },
            )
        )

    plot_test_loss_across_seeds(
        histories=histories_vqc2,
        title=f"{vqc2_spec.name} — Test Loss Across Seeds (Dataset B σ=0.10, n=100)",
        out_path=os.path.join(FIG_DIR, "lc_vqc_L2_analytic_testloss_seeds.png"),
    )

    # ----- Fix 12 artifact: combined raw per-seed-per-epoch CSV -----
    save_records_csv(RAW_HISTORY_CSV, raw_rows)


if __name__ == "__main__":
    main()
