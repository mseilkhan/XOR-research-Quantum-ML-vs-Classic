from __future__ import annotations

"""
Experiment 3.4 — Seed sensitivity
- For a fixed benchmark mode (Dataset B, σ=0.10, n=100),
  log per-seed final test acc and test loss for:
  Linear, MLP(h=4), VQC(L=1 analytic), VQC(L=1 shots=1024), VQC(L=2 analytic), VQC(L=2 shots=1024)
Artifacts:
- CSV per-seed results (acc + bce)
- (Optional) simple plots can be produced later from the CSV
"""

import os
from typing import List, Dict

from core.data.xor_dataset import make_split
from core.train.trainer import TrainConfig, Trainer
from core.utils.determinism import set_global_determinism

from pathlib import Path
from core.utils.run_context import create_run_context
from core.utils.logging import setup_logger

from experiments.settings import (
    DATASET_SEED, SPLIT_SEED, MODEL_SEEDS, SEED_SENSITIVITY_SEEDS,
    CSV_DIR,
    LR_HP, MLP_HP, VQC_HP,
)
from experiments.utils import ensure_output_dirs, save_summaries_csv
from experiments.models_factory import make_linear, make_mlp, make_vqc
from experiments.settings import OPTIMIZER, ADAM_BETA1, ADAM_BETA2, ADAM_EPS

from pathlib import Path
from tqdm import tqdm

from core.utils.logging import setup_logger
from core.utils.io import append_csv_row, csv_contains_row



def eval_per_seed(spec_name: str, factory, hp, split) -> List[Dict]:
    trainer = Trainer(TrainConfig(hp.epochs,
                                  hp.lr,
                                  optimizer=OPTIMIZER,
                                  adam_beta1=ADAM_BETA1,
                                  adam_beta2=ADAM_BETA2,
                                  adam_eps=ADAM_EPS,
                                  ))
    rows = []
    for s in SEED_SENSITIVITY_SEEDS:
        m = factory(s)
        trainer.fit(m, split.X_train, split.y_train, split.X_test, split.y_test)
        final = Trainer.final_metrics(m, split.X_train, split.y_train, split.X_test, split.y_test)
        rows.append({
            "model": spec_name,
            "seed": s,
            "test_acc": final["test_acc"],
            "test_loss": final["test_loss"],
            "train_acc": final["train_acc"],
            "train_loss": final["train_loss"],
            "n_params": m.n_params(),
        })
    return rows


def main():
    exp_name = "exp_05_seed_sensitivity"
    settings = {
        "DATASET_SEED": DATASET_SEED,
        "SPLIT_SEED": SPLIT_SEED,
        "MODEL_SEEDS": list(MODEL_SEEDS),
        "SEED_SENSITIVITY_SEEDS": list(SEED_SENSITIVITY_SEEDS),
        "OPTIMIZER": OPTIMIZER,
        "ADAM_BETA1": ADAM_BETA1,
        "ADAM_BETA2": ADAM_BETA2,
        "ADAM_EPS": ADAM_EPS,
        "DETERMINISM_SEED": DATASET_SEED
    }
    run_ctx = create_run_context(exp_name=exp_name, settings=settings)
    logger = setup_logger(exp_name, run_ctx.run_id, Path("outputs/logs"))
    logger.info(f"metadata: {run_ctx.metadata_path}")

    applied_seeds = set_global_determinism(DATASET_SEED)
    logger.info(f"Determinism enforced: {applied_seeds}")

    ensure_output_dirs(CSV_DIR)

    split = make_split("B", sigma=0.10, n_per_cluster=100, data_seed=DATASET_SEED, split_seed=SPLIT_SEED)

    specs = []
    specs.append((*make_linear(), LR_HP))
    specs.append((*make_mlp(h=4), MLP_HP))
    specs.append((*make_vqc(L=1, shots=None), VQC_HP))
    specs.append((*make_vqc(L=1, shots=1024), VQC_HP))
    specs.append((*make_vqc(L=2, shots=None), VQC_HP))
    specs.append((*make_vqc(L=2, shots=1024), VQC_HP))

    out_csv = os.path.join(CSV_DIR, "seed_sensitivity_runs.csv")
    fields = ["model", "seed", "test_acc", "test_loss", "train_acc", "train_loss", "n_params"]

    for spec, factory, hp in tqdm(specs, desc="Models (seed sensitivity)"):
        logger.info(f"Starting: {spec.name}")
        trainer = Trainer(TrainConfig(hp.epochs,
                                      hp.lr,
                                      optimizer=OPTIMIZER,
                                      adam_beta1=ADAM_BETA1,
                                      adam_beta2=ADAM_BETA2,
                                      adam_eps=ADAM_EPS,
                                      ))

        for s in tqdm(SEED_SENSITIVITY_SEEDS, desc=f"{spec.name} seeds", leave=False):
            key = {"model": spec.name, "seed": s}
            if csv_contains_row(out_csv, key):
                continue

            m = factory(s)
            trainer.fit(m, split.X_train, split.y_train, split.X_test, split.y_test)
            final = Trainer.final_metrics(m, split.X_train, split.y_train, split.X_test, split.y_test)

            row = {
                "model": spec.name,
                "seed": s,
                "test_acc": final["test_acc"],
                "test_loss": final["test_loss"],
                "train_acc": final["train_acc"],
                "train_loss": final["train_loss"],
                "n_params": m.n_params(),
            }
            append_csv_row(out_csv, fields, row)

        logger.info(f"Done: {spec.name}")


if __name__ == "__main__":
    main()
