from __future__ import annotations

"""
Experiment 3 — Robustness (Dataset B)
- accuracy vs noise sigma (B, n=100)
- accuracy vs dataset size n_per_cluster (B, sigma=0.10)
- VQC shot dependence handled in separate experiment file (exp_04)
Artifacts:
- CSV raw (per-seed runs) and summaries (mean±std)
- compare plots (square style handled by core.viz)
"""

import os
from typing import List, Dict, Any

from pathlib import Path
from tqdm import tqdm

from core.utils.logging import setup_logger
from core.utils.io import append_csv_row, csv_contains_row

from core.eval.sweeps import sweep_noise_datasetB, sweep_size_datasetB
from core.train.trainer import TrainConfig
from core.viz.robustness import plot_mean_std_curve
from core.utils.determinism import set_global_determinism

from pathlib import Path
from core.utils.run_context import create_run_context
from core.utils.logging import setup_logger

from dataclasses import asdict, is_dataclass

from experiments.settings import (
    DATASET_SEED, SPLIT_SEED, MODEL_SEEDS,
    NOISE_SWEEP, SIZE_SWEEP,
    FIG_DIR, CSV_DIR,
    LR_HP, MLP_HP, VQC_HP,
)
from experiments.utils import ensure_output_dirs, save_records_csv, save_summaries_csv
from experiments.models_factory import make_linear, make_mlp, make_vqc
from experiments.settings import OPTIMIZER, ADAM_BETA1, ADAM_BETA2, ADAM_EPS

def to_row(obj):
    """
    Convert RunRecord / dataclass / object-with-attributes / dict into a plain dict.
    """
    if obj is None:
        return {}
    if isinstance(obj, dict):
        return dict(obj)
    if is_dataclass(obj):
        return asdict(obj)
    # fallback: object with __dict__ or attributes
    if hasattr(obj, "__dict__"):
        return dict(obj.__dict__)
    # last resort: try attribute introspection
    out = {}
    for name in dir(obj):
        if name.startswith("_"):
            continue
        try:
            val = getattr(obj, name)
        except Exception:
            continue
        if callable(val):
            continue
        out[name] = val
    return out


def pick(row: dict, candidates: list[str], default=None):
    """
    Pick first existing key from candidates in row.
    """
    for k in candidates:
        if k in row:
            return row[k]
    return default


def main():
    exp_name = "exp_03_robustness_datasetB"
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

    # Models used in robustness: Linear, MLP(h=4), VQC(L=1 shots=1024), VQC(L=2 shots=1024)
    specs = []
    specs.append((*make_linear(), LR_HP))
    specs.append((*make_mlp(h=4), MLP_HP))
    specs.append((*make_vqc(L=1, shots=1024), VQC_HP))
    specs.append((*make_vqc(L=2, shots=1024), VQC_HP))

    noise_runs_csv = os.path.join(CSV_DIR, "robustness_noise_runs.csv")
    noise_sum_csv = os.path.join(CSV_DIR, "robustness_noise_summary.csv")
    size_runs_csv = os.path.join(CSV_DIR, "robustness_size_runs.csv")
    size_sum_csv = os.path.join(CSV_DIR, "robustness_size_summary.csv")

    # fieldnames will be inferred from first row we write (we keep explicit lists for stability)
    # NOTE: These keys must match what sweeps return.
    noise_run_fields = ["dataset_name", "model_name", "seed", "sigma", "n_per_cluster", "train_acc", "train_loss", "test_acc", "test_loss", "train_seconds", "n_params"]
    noise_sum_fields = ["dataset_name", "model_name", "sigma", "n_per_cluster", "train_acc_mean", "train_acc_std", "train_loss_mean", "train_loss_std", "test_acc_mean", "test_acc_std", "test_loss_mean", "test_loss_std", "train_seconds_mean", "train_seconds_std", "n_params_mean", "n_params_std"]

    size_run_fields = ["dataset_name", "model_name", "seed", "sigma", "n_per_cluster", "train_acc", "train_loss", "test_acc", "test_loss", "train_seconds", "n_params"]
    size_sum_fields = ["dataset_name", "model_name", "sigma", "n_per_cluster", "train_acc_mean", "train_acc_std", "train_loss_mean", "train_loss_std", "test_acc_mean", "test_acc_std", "test_loss_mean", "test_loss_std", "train_seconds_mean", "train_seconds_std", "n_params_mean", "n_params_std"]


    # ---------- (a) Noise sweep ----------
    all_records_noise = []
    all_summaries_noise = []

    for spec, factory, hp in tqdm(specs, desc="Noise sweep models"):
        logger.info(f"Noise sweep start: {spec.name}")
        rec, summ = sweep_noise_datasetB(
            sigmas=NOISE_SWEEP,
            n_per_cluster=100,
            model_factory=factory,
            model_name=spec.name,
            train_cfg=TrainConfig(hp.epochs,
                                  hp.lr,
                                  optimizer=OPTIMIZER,
                                  adam_beta1=ADAM_BETA1,
                                  adam_beta2=ADAM_BETA2,
                                  adam_eps=ADAM_EPS,
                                  ),
            data_seed=DATASET_SEED,
            split_seed=SPLIT_SEED,
            model_seeds=MODEL_SEEDS,
            logger=logger,
        )
        for r in rec:
            all_records_noise.append(to_row(r))
        for s in summ:
            all_summaries_noise.append(to_row(s))

        # write raw runs incrementally (resume on model_name+seed+sigma+n_per_cluster)
        for r in rec:
            rr = to_row(r)

            # normalize key fields (support multiple naming conventions)
            model_name = pick(rr, ["model_name", "model", "name"])
            seed = pick(rr, ["seed", "model_seed"])
            sigma = pick(rr, ["sigma", "noise_sigma"])
            n_per_cluster = pick(rr, ["n_per_cluster", "n", "size", "points_per_cluster"])

            key = {
                "model_name": model_name,
                "seed": seed,
                "sigma": sigma,
                "n_per_cluster": n_per_cluster,
            }

            # ensure required columns exist in row
            rr.setdefault("model_name", model_name)
            rr.setdefault("seed", seed)
            rr.setdefault("sigma", sigma)
            rr.setdefault("n_per_cluster", n_per_cluster)

            if not csv_contains_row(noise_runs_csv, key):
                append_csv_row(noise_runs_csv, noise_run_fields, rr)

        # write summaries incrementally (resume on model_name+sigma+n_per_cluster)
        for s in summ:
            ss = to_row(s)

            model_name = pick(ss, ["model_name", "model", "name"])
            sigma = pick(ss, ["sigma", "noise_sigma"])
            n_per_cluster = pick(ss, ["n_per_cluster", "n", "size", "points_per_cluster"])

            key = {
                "model_name": model_name,
                "sigma": sigma,
                "n_per_cluster": n_per_cluster,
            }

            ss.setdefault("model_name", model_name)
            ss.setdefault("sigma", sigma)
            ss.setdefault("n_per_cluster", n_per_cluster)

            if not csv_contains_row(noise_sum_csv, key):
                append_csv_row(noise_sum_csv, noise_sum_fields, ss)

        logger.info(f"Noise sweep done: {spec.name}")


    save_records_csv(os.path.join(CSV_DIR, "robustness_noise_runs.csv"), all_records_noise)
    save_summaries_csv(os.path.join(CSV_DIR, "robustness_noise_summary.csv"), all_summaries_noise)

    # Plot: one curve per model (mean±std)
    for spec, _, _ in specs:
        sub = [s for s in all_summaries_noise if s["model_name"] == spec.name]
        sub = sorted(sub, key=lambda r: r["sigma"])
        xs = [r["sigma"] for r in sub]
        ys_m = [r["test_acc_mean"] for r in sub]
        ys_s = [r["test_acc_std"] for r in sub]
        plot_mean_std_curve(
            xs=xs, ys_mean=ys_m, ys_std=ys_s,
            title=f"Dataset B (n=100): Accuracy vs noise — {spec.name}",
            xlabel="noise σ", ylabel="test accuracy",
            label=spec.name,
            out_path=os.path.join(FIG_DIR, f"rob_noise_{spec.name.replace(' ', '').replace('=', '')}.png"),
        )

    # ---------- (b) Size sweep ----------
    all_records_size = []
    all_summaries_size = []

    for spec, factory, hp in tqdm(specs, desc="Size sweep models"):
        logger.info(f"Size sweep start: {spec.name}")
        rec, summ = sweep_size_datasetB(
            sizes=SIZE_SWEEP,
            sigma=0.10,
            model_factory=factory,
            model_name=spec.name,
            train_cfg=TrainConfig(hp.epochs,
                                  hp.lr,
                                  optimizer=OPTIMIZER,
                                  adam_beta1=ADAM_BETA1,
                                  adam_beta2=ADAM_BETA2,
                                  adam_eps=ADAM_EPS,
                                  ),
            data_seed=DATASET_SEED,
            split_seed=SPLIT_SEED,
            model_seeds=MODEL_SEEDS,
            logger=logger,
        )
        for r in rec:
            all_records_size.append(to_row(r))
        for s in summ:
            all_summaries_size.append(to_row(s))

        for r in rec:
            rr = to_row(r)

            # normalize key fields (support multiple naming conventions)
            model_name = pick(rr, ["model_name", "model", "name"])
            seed = pick(rr, ["seed", "model_seed"])
            sigma = pick(rr, ["sigma", "noise_sigma"])
            n_per_cluster = pick(rr, ["n_per_cluster", "n", "size", "points_per_cluster"])

            key = {
                "model_name": model_name,
                "seed": seed,
                "sigma": sigma,
                "n_per_cluster": n_per_cluster,
            }

            # ensure required columns exist in row
            rr.setdefault("model_name", model_name)
            rr.setdefault("seed", seed)
            rr.setdefault("sigma", sigma)
            rr.setdefault("n_per_cluster", n_per_cluster)

            if not csv_contains_row(size_runs_csv, key):
                append_csv_row(size_runs_csv, size_run_fields, rr)

        for s in summ:
            ss = to_row(s)

            model_name = pick(ss, ["model_name", "model", "name"])
            sigma = pick(ss, ["sigma", "noise_sigma"])
            n_per_cluster = pick(ss, ["n_per_cluster", "n", "size", "points_per_cluster"])

            key = {
                "model_name": model_name,
                "sigma": sigma,
                "n_per_cluster": n_per_cluster,
            }

            ss.setdefault("model_name", model_name)
            ss.setdefault("sigma", sigma)
            ss.setdefault("n_per_cluster", n_per_cluster)

            if not csv_contains_row(size_sum_csv, key):
                append_csv_row(size_sum_csv, size_sum_fields, ss)

        logger.info(f"Size sweep done: {spec.name}")

    save_records_csv(os.path.join(CSV_DIR, "robustness_size_runs.csv"), all_records_size)
    save_summaries_csv(os.path.join(CSV_DIR, "robustness_size_summary.csv"), all_summaries_size)

    for spec, _, _ in specs:
        sub = [s for s in all_summaries_size if s["model_name"] == spec.name]
        sub = sorted(sub, key=lambda r: r["n_per_cluster"])
        xs = [r["n_per_cluster"] for r in sub]
        ys_m = [r["test_acc_mean"] for r in sub]
        ys_s = [r["test_acc_std"] for r in sub]
        plot_mean_std_curve(
            xs=xs, ys_mean=ys_m, ys_std=ys_s,
            title=f"Dataset B (σ=0.10): Accuracy vs dataset size — {spec.name}",
            xlabel="n_per_cluster", ylabel="test accuracy",
            label=spec.name,
            out_path=os.path.join(FIG_DIR, f"rob_size_{spec.name.replace(' ', '').replace('=', '')}.png"),
        )


if __name__ == "__main__":
    main()
