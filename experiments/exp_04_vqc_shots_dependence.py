from __future__ import annotations

"""
Experiment 3.3 — VQC shot dependence
- Evaluate VQC accuracy vs shots for a fixed dataset mode (Dataset B, σ=0.10, n=100)
- regimes: analytic, 128, 1024
Artifacts:
- CSV raw + summary
- Plot mean±std curve
"""

import os

from pathlib import Path
from tqdm import tqdm

from core.utils.logging import setup_logger
from core.utils.io import append_csv_row, csv_contains_row

from core.data.xor_dataset import make_split
from core.eval.sweeps import run_repeated
from core.train.trainer import TrainConfig
from core.viz.robustness import plot_mean_std_curve
from core.utils.determinism import set_global_determinism

from pathlib import Path
from core.utils.run_context import create_run_context
from core.utils.logging import setup_logger

from experiments.settings import (
    DATASET_SEED, SPLIT_SEED, MODEL_SEEDS,
    VQC_HP, FIG_DIR, CSV_DIR,
)
from experiments.utils import ensure_output_dirs, save_records_csv, save_summaries_csv, tag_shots
from experiments.models_factory import make_vqc
from experiments.settings import OPTIMIZER, ADAM_BETA1, ADAM_BETA2, ADAM_EPS

def main():
    exp_name = "exp_04_vqc_shots_dependence"
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
    logger.info(f"metadata: {run_ctx.metadata_path}")\

    applied_seeds = set_global_determinism(DATASET_SEED)
    logger.info(f"Determinism enforced: {applied_seeds}")

    ensure_output_dirs(FIG_DIR, CSV_DIR)

    split = make_split("B", sigma=0.10, n_per_cluster=100, data_seed=DATASET_SEED, split_seed=SPLIT_SEED)

    all_records = []
    summaries = []

    runs_csv = os.path.join(CSV_DIR, "vqc_shots_runs.csv")
    sum_csv = os.path.join(CSV_DIR, "vqc_shots_summary.csv")

    run_fields = ["dataset_name", "model_name", "seed", "sigma", "n_per_cluster", "shots", "L",
                  "train_acc", "train_loss", "test_acc", "test_loss", "train_seconds", "n_params"]
    sum_fields = ["dataset_name", "model_name", "sigma", "n_per_cluster", "shots", "L",
                  "train_acc_mean", "train_acc_std", "train_loss_mean", "train_loss_std",
                  "test_acc_mean", "test_acc_std", "test_loss_mean", "test_loss_std",
                  "train_seconds_mean", "train_seconds_std", "n_params_mean", "n_params_std"]

    for shots in tqdm([None, 128, 1024], desc="VQC shots regimes"):
        logger.info(f"Running VQC(L=1) shots={tag_shots(shots)}")
        spec, factory = make_vqc(L=1, shots=shots)
        rec, summ = run_repeated(
            model_factory=factory,
            split=split,
            dataset_name="B",
            model_name=spec.name,
            train_cfg=TrainConfig(VQC_HP.epochs, VQC_HP.lr),
            model_seeds=MODEL_SEEDS,
            meta={"sigma": 0.10, "n_per_cluster": 100, "shots": shots, "L": 1},
        )
        for r in rec:
            key = {"model_name": r["model_name"], "seed": r["seed"], "shots": r["shots"], "L": r["L"]}
            if not csv_contains_row(runs_csv, key):
                append_csv_row(runs_csv, run_fields, r)

        key_s = {"model_name": summ["model_name"], "shots": summ["shots"], "L": summ["L"]}
        if not csv_contains_row(sum_csv, key_s):
            append_csv_row(sum_csv, sum_fields, summ)

        all_records.extend(rec)
        summaries.append(summ)

    # save_records_csv(os.path.join(CSV_DIR, "vqc_shots_runs.csv"), all_records)
    # save_summaries_csv(os.path.join(CSV_DIR, "vqc_shots_summary.csv"), summaries)

    # Plot mean±std accuracy vs shots (use x-axis as 0=analytic then shots)
    xs = [0, 128, 1024]
    ys_m = [s["test_acc_mean"] for s in sorted(summaries, key=lambda r: (r["shots"] is not None, r["shots"] or 0))]
    ys_s = [s["test_acc_std"] for s in sorted(summaries, key=lambda r: (r["shots"] is not None, r["shots"] or 0))]

    plot_mean_std_curve(
        xs=xs,
        ys_mean=ys_m,
        ys_std=ys_s,
        title="VQC(L=1): Accuracy vs shots (Dataset B σ=0.10, n=100)",
        xlabel="shots (0 = analytic)",
        ylabel="test accuracy",
        label="VQC(L=1)",
        out_path=os.path.join(FIG_DIR, "rob_vqc_shots.png"),
    )


if __name__ == "__main__":
    main()
