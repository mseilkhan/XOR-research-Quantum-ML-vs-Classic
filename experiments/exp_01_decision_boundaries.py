from __future__ import annotations

"""
Experiment 1 — Decision boundaries
- MLP(h=4): Dataset A, Dataset B sigma=0.10, Dataset B sigma=0.20
- VQC: L=1/2 in analytic and 1024 shots
Artifacts:
- boundary images (square)
- raw per-seed final metrics CSV (added by Fix 12)
"""

import os
from typing import Any, Dict, List, Optional

from core.data.xor_dataset import make_split
from core.data.xor_dataset import DatasetSplit, make_dataset_A
from core.train.trainer import TrainConfig, Trainer
from core.viz.decision_boundary import plot_decision_boundary_seed_background
from core.utils.determinism import set_global_determinism
from tqdm import tqdm

from pathlib import Path
from core.utils.run_context import create_run_context
from core.utils.logging import setup_logger

from experiments.settings import (
    DATASET_SEED,
    SPLIT_SEED,
    MODEL_SEEDS,
    FIG_DIR,
    CSV_DIR,
    DB_SIGMAS,
    DB_N_PER_CLUSTER,
    MLP_HP,
    VQC_HP,
    SMOKE,
)
from experiments.utils import ensure_output_dirs, save_records_csv, tag_shots, tag_sigma, tag_n
from experiments.models_factory import make_mlp, make_vqc
from experiments.settings import OPTIMIZER, ADAM_BETA1, ADAM_BETA2, ADAM_EPS

RAW_METRICS_CSV = os.path.join(CSV_DIR, "exp01_decision_boundaries_raw_metrics.csv")


def _final_metrics_row(
    *,
    dataset: str,
    model: str,
    seed: int,
    sigma: Optional[float],
    n_per_cluster: Optional[int],
    shots: Optional[int],
    L: Optional[int],
    metrics: Dict[str, float],
    n_params: int,
) -> Dict[str, Any]:
    return {
        "experiment": "exp_01_decision_boundaries",
        "dataset": dataset,
        "model": model,
        "seed": int(seed),
        "sigma": None if sigma is None else float(sigma),
        "n_per_cluster": None if n_per_cluster is None else int(n_per_cluster),
        "shots": shots,          # None means analytic
        "L": L,                  # None for classical models
        "train_acc": float(metrics["train_acc"]),
        "train_loss": float(metrics["train_loss"]),
        "test_acc": float(metrics["test_acc"]),
        "test_loss": float(metrics["test_loss"]),
        "n_params": int(n_params),
    }


def train_models_across_seeds(
    *,
    model_factory,
    split,
    train_cfg: TrainConfig,
    dataset_label: str,
    sigma: Optional[float],
    n_per_cluster: Optional[int],
    model_label: str,
    shots: Optional[int],
    L: Optional[int],
    raw_rows: List[Dict[str, Any]],
):
    """
    Train and return list of fitted models across model seeds (for background contours),
    and collect raw per-seed final metrics (Fix 12).
    """
    trainer = Trainer(train_cfg)
    models = []
    for s in tqdm(list(MODEL_SEEDS), desc=f"{model_label} seeds", leave=False):
        m = model_factory(s)
        trainer.fit(m, split.X_train, split.y_train, split.X_test, split.y_test)

        metrics = Trainer.final_metrics(m, split.X_train, split.y_train, split.X_test, split.y_test)
        raw_rows.append(
            _final_metrics_row(
                dataset=dataset_label,
                model=model_label,
                seed=s,
                sigma=sigma,
                n_per_cluster=n_per_cluster,
                shots=shots,
                L=L,
                metrics=metrics,
                n_params=m.n_params(),
            )
        )

        models.append(m)
    return models


def main():

    # --------------
    # SMOKE MODE
    # --------------
    if SMOKE:
        # local imports to avoid touching full pipeline
        from core.viz.decision_boundary import plot_decision_function

        # dataset A full (4 points)
        X_A, y_A = make_dataset_A(seed=DATASET_SEED)
        split_A = DatasetSplit(X_train=X_A, y_train=y_A, X_test=X_A, y_test=y_A)

        # MLP(h=4), seed=0, 1 epoch
        mlp_spec, mlp_factory = make_mlp(h=4)
        trainer = Trainer(TrainConfig(epochs=1, lr=float(MLP_HP.lr)))
        m = mlp_factory(0)
        trainer.fit(m, split_A.X_train, split_A.y_train, split_A.X_test, split_A.y_test)

        # save a single figure required by manifest_smoke
        out_A = os.path.join(FIG_DIR, "db_mlp_h4_datasetA.png")
        plot_decision_function(
            model=m,
            X=X_A,
            y=y_A,
            title=f"{mlp_spec.name} — Dataset A (SMOKE)",
            out_path=out_A,
            xlim=(0.0, 1.0),
            ylim=(0.0, 1.0),
            n_grid=120,
            show_boundary=True,
        )

        # write one raw row so validator can check existence
        final = Trainer.final_metrics(m, split_A.X_train, split_A.y_train, split_A.X_test, split_A.y_test)
        raw_row = {
            "experiment": "exp_01_decision_boundaries",
            "dataset": "A",
            "model": mlp_spec.name,
            "seed": 0,
            "sigma": None,
            "n_per_cluster": None,
            "shots": None,
            "L": None,
            "train_acc": float(final["train_acc"]),
            "train_loss": float(final["train_loss"]),
            "test_acc": float(final["test_acc"]),
            "test_loss": float(final["test_loss"]),
            "n_params": int(m.n_params()),
        }
        # save_records_csv already used in exp_01 — reuse it
        save_records_csv(RAW_METRICS_CSV, [raw_row])

        return

    exp_name = "exp_01_decision_boundaries"

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


    applied_seeds = set_global_determinism(DATASET_SEED)
    logger.info(f"Determinism enforced: {applied_seeds}")

    ensure_output_dirs(FIG_DIR, CSV_DIR)

    raw_rows: List[Dict[str, Any]] = []

    # ---------- MLP decision boundaries ----------
    mlp_spec, mlp_factory = make_mlp(h=4)

    # Dataset A
    X_A, y_A = make_dataset_A(seed=DATASET_SEED)

    # Dataset A: only 4 points -> must train on ALL points (otherwise XOR is not learnable with split 80/20)
    split_A = DatasetSplit(
        X_train=X_A,
        y_train=y_A,
        X_test=X_A,
        y_test=y_A,
    )
    mlp_models_A = train_models_across_seeds(
        model_factory=mlp_factory,
        split=split_A,
        train_cfg=TrainConfig(MLP_HP.epochs,
                              MLP_HP.lr,
                              optimizer=OPTIMIZER,
                              adam_beta1=ADAM_BETA1,
                              adam_beta2=ADAM_BETA2,
                              adam_eps=ADAM_EPS,),
        dataset_label="A",
        sigma=None,
        n_per_cluster=None,
        model_label=mlp_spec.name,
        shots=None,
        L=None,
        raw_rows=raw_rows,
    )
    out_A = os.path.join(FIG_DIR, "db_mlp_h4_datasetA.png")
    plot_decision_boundary_seed_background(
        models=mlp_models_A,
        X=X_A,
        y=y_A,
        title=f"{mlp_spec.name} — Dataset A",
        out_path=out_A,
        xlim=(0, 1),
        ylim=(0, 1),
    )

    # Dataset B: sigma=0.10 and 0.20
    for sigma in tqdm(list(DB_SIGMAS), desc="MLP Dataset B sigmas"):
        split_B = make_split(
            "B",
            sigma=float(sigma),
            n_per_cluster=DB_N_PER_CLUSTER,
            data_seed=DATASET_SEED,
            split_seed=SPLIT_SEED,
        )
        mlp_models_B = train_models_across_seeds(
            model_factory=mlp_factory,
            split=split_B,
            train_cfg=TrainConfig(MLP_HP.epochs,
                                  MLP_HP.lr,
                                  optimizer=OPTIMIZER,
                                  adam_beta1=ADAM_BETA1,
                                  adam_beta2=ADAM_BETA2,
                                  adam_eps=ADAM_EPS, ),
            dataset_label="B",
            sigma=float(sigma),
            n_per_cluster=int(DB_N_PER_CLUSTER),
            model_label=mlp_spec.name,
            shots=None,
            L=None,
            raw_rows=raw_rows,
        )
        out_B = os.path.join(
            FIG_DIR,
            f"db_mlp_h4_datasetB_{tag_sigma(float(sigma))}_{tag_n(DB_N_PER_CLUSTER)}.png",
        )
        plot_decision_boundary_seed_background(
            models=mlp_models_B,
            X=split_B.X_train,
            y=split_B.y_train,
            title=f"{mlp_spec.name} — Dataset B (σ={sigma:.2f}, n={DB_N_PER_CLUSTER})",
            out_path=out_B,
            xlim=(0, 1),
            ylim=(0, 1),
            n_grid=600,
        )

    # ---------- VQC decision boundaries ----------
    # Representative benchmark dataset
    split_rep = make_split("B", sigma=0.10, n_per_cluster=100, data_seed=DATASET_SEED, split_seed=SPLIT_SEED)

    for L in tqdm([1, 2], desc="VQC depth L"):
        for shots in tqdm([None, 1024], desc=f"VQC(L={L}) shots", leave=False):
            vqc_spec, vqc_factory = make_vqc(L=L, shots=shots)
            vqc_models = train_models_across_seeds(
                model_factory=vqc_factory,
                split=split_rep,
                train_cfg=TrainConfig(VQC_HP.epochs,
                                      VQC_HP.lr,
                                      optimizer=OPTIMIZER,
                                      adam_beta1=ADAM_BETA1,
                                      adam_beta2=ADAM_BETA2,
                                      adam_eps=ADAM_EPS,
                                      ),
                dataset_label="B",
                sigma=0.10,
                n_per_cluster=100,
                model_label=vqc_spec.name,
                shots=shots,
                L=L,
                raw_rows=raw_rows,
            )

            out_path = os.path.join(FIG_DIR, f"db_vqc_L{L}_{tag_shots(shots)}.png")
            plot_decision_boundary_seed_background(
                models=vqc_models,
                X=split_rep.X_train,
                y=split_rep.y_train,
                title=f"{vqc_spec.name} — Dataset B (σ=0.10, n=100)",
                out_path=out_path,
                xlim=(0, 1),
                ylim=(0, 1),
            )

    # ----- Fix 12 artifact: raw per-seed final metrics -----
    save_records_csv(RAW_METRICS_CSV, raw_rows)


if __name__ == "__main__":
    main()
