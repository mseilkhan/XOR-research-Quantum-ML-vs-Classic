from __future__ import annotations

"""
Experiment 4 — Summary tables (benchmark mode)
Mode: Dataset B, σ=0.10, n_per_cluster=100
Outputs:
- performance table (train acc, test acc, test loss) mean±std
- cost table (#params, train_seconds) mean±std
- settings tables: exp_settings_data, exp_settings_models, vqc_settings
All in LaTeX-friendly CSV or .tex.
"""

import os
from typing import Dict, Any, List

from pathlib import Path
from tqdm import tqdm

from core.utils.logging import setup_logger
from core.utils.io import append_csv_row, csv_contains_row

from core.data.xor_dataset import make_split
from core.eval.sweeps import run_repeated
from core.train.trainer import TrainConfig
from core.utils.determinism import set_global_determinism

from pathlib import Path
from core.utils.run_context import create_run_context
from core.utils.logging import setup_logger

from experiments.settings import (
    DATASET_SEED, SPLIT_SEED, MODEL_SEEDS,
    BENCH_SIGMA, BENCH_N,
    TABLE_DIR, CSV_DIR,
    LR_HP, MLP_HP, VQC_HP,
)
from dataclasses import asdict, is_dataclass

from experiments.utils import ensure_output_dirs, save_records_csv, save_summaries_csv
from experiments.models_factory import make_linear, make_mlp, make_vqc
from experiments.settings import OPTIMIZER, ADAM_BETA1, ADAM_BETA2, ADAM_EPS

def to_row(obj):
    """Convert RunRecord / dataclass / object-with-attributes / dict into a plain dict"""
    if obj is None:
        return {}
    if isinstance(obj, dict):
        return dict(obj)
    if is_dataclass(obj):
        return asdict(obj)
    if hasattr(obj, "__dict__"):
        return dict(obj.__dict__)
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
    """Pick the first available key from candidates."""
    for k in candidates:
        if k in row:
            return row[k]
    return default

def main():
    exp_name = "exp_06_summary_tables"
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
    logger = setup_logger("exp_06_summary_tables", run_ctx.run_id, Path("outputs/logs"))
    logger.info(f"metadata: {run_ctx.metadata_path}")

    applied_seeds = set_global_determinism(DATASET_SEED)
    logger.info(f"Determinism enforced: {applied_seeds}")

    ensure_output_dirs(TABLE_DIR, CSV_DIR)

    split = make_split("B", sigma=BENCH_SIGMA, n_per_cluster=BENCH_N, data_seed=DATASET_SEED, split_seed=SPLIT_SEED)

    specs = []
    specs.append((*make_linear(), LR_HP, {"model_group": "classical"}))
    specs.append((*make_mlp(h=4), MLP_HP, {"model_group": "classical"}))
    specs.append((*make_vqc(L=1, shots=None), VQC_HP, {"model_group": "quantum"}))
    specs.append((*make_vqc(L=1, shots=1024), VQC_HP, {"model_group": "quantum"}))
    specs.append((*make_vqc(L=2, shots=None), VQC_HP, {"model_group": "quantum"}))
    specs.append((*make_vqc(L=2, shots=1024), VQC_HP, {"model_group": "quantum"}))

    all_records = []
    summaries = []

    runs_csv = os.path.join(CSV_DIR, "summary_benchmark_runs.csv")
    sum_csv = os.path.join(CSV_DIR, "summary_benchmark_meanstd.csv")

    run_fields = ["dataset_name", "model_name", "seed", "sigma", "n_per_cluster",
                  "train_acc", "train_loss", "test_acc", "test_loss", "train_seconds", "n_params",
                  "model_group"]
    sum_fields = ["dataset_name", "model_name", "sigma", "n_per_cluster",
                  "train_acc_mean", "train_acc_std", "train_loss_mean", "train_loss_std",
                  "test_acc_mean", "test_acc_std", "test_loss_mean", "test_loss_std",
                  "train_seconds_mean", "train_seconds_std", "n_params_mean", "n_params_std",
                  "model_group"]

    for spec, factory, hp, extra in tqdm(specs, desc="Benchmark models"):
        logger.info(f"Benchmark start: {spec.name}")
        rec, summ = run_repeated(
            model_factory=factory,
            split=split,
            dataset_name="B",
            model_name=spec.name,
            train_cfg=TrainConfig(hp.epochs,
                                  hp.lr,
                                  optimizer=OPTIMIZER,
                                  adam_beta1=ADAM_BETA1,
                                  adam_beta2=ADAM_BETA2,
                                  adam_eps=ADAM_EPS,
                                  ),
            model_seeds=MODEL_SEEDS,
            meta={
                "sigma": BENCH_SIGMA,
                "n_per_cluster": BENCH_N,
                **extra,
            },
        )

        # append raw
        for r in rec:
            r2 = to_row(r)

            # normalize required keys (support multiple naming conventions)
            model_name = pick(r2, ["model_name", "model", "name"])
            seed = pick(r2, ["seed", "model_seed"])
            sigma = pick(r2, ["sigma", "noise_sigma"])
            n_per_cluster = pick(r2, ["n_per_cluster", "n", "size", "points_per_cluster"])

            r2.setdefault("model_name", model_name)
            r2.setdefault("seed", seed)
            r2.setdefault("sigma", sigma)
            r2.setdefault("n_per_cluster", n_per_cluster)

            if "model_group" not in r2:
                r2["model_group"] = extra.get("model_group")

            key = {
                "model_name": r2["model_name"],
                "seed": r2["seed"],
                "model_group": r2["model_group"],
            }

            if not csv_contains_row(runs_csv, key):
                append_csv_row(runs_csv, run_fields, r2)

        # append summary
        s2 = to_row(summ)

        model_name = pick(s2, ["model_name", "model", "name"])
        sigma = pick(s2, ["sigma", "noise_sigma"])
        n_per_cluster = pick(s2, ["n_per_cluster", "n", "size", "points_per_cluster"])

        s2.setdefault("model_name", model_name)
        s2.setdefault("sigma", sigma)
        s2.setdefault("n_per_cluster", n_per_cluster)

        if "model_group" not in s2:
            s2["model_group"] = extra.get("model_group")

        key_s = {
            "model_name": s2["model_name"],
            "model_group": s2["model_group"],
        }

        if not csv_contains_row(sum_csv, key_s):
            append_csv_row(sum_csv, sum_fields, s2)

        logger.info(f"Benchmark done: {spec.name}")

    # Save raw + summary
    # save_records_csv(os.path.join(CSV_DIR, "summary_benchmark_runs.csv"), all_records)
    # save_summaries_csv(os.path.join(CSV_DIR, "summary_benchmark_meanstd.csv"), summaries)

    # Minimal settings tables (data + models) in CSV
    exp_settings_data = [{
        "dataset_seed": DATASET_SEED,
        "split_seed": SPLIT_SEED,
        "train_frac": 0.80,
        "dataset": "B",
        "sigma_grid": str([0.00, 0.05, 0.10, 0.20, 0.30]),
        "n_per_cluster_grid": str([25, 50, 100, 250, 500]),
    }]
    save_summaries_csv(os.path.join(TABLE_DIR, "exp_settings_data.csv"), exp_settings_data)

    exp_settings_models = [
        {"model": "Linear", "params": 3, "epochs": LR_HP.epochs, "lr": LR_HP.lr},
        {"model": "MLP", "activation_hidden": "sigmoid", "activation_out": "sigmoid", "h_grid": str([1,2,4,8]),
         "epochs": MLP_HP.epochs, "lr": MLP_HP.lr},
        {"model": "VQC", "qubits": 2, "encoding": "RX(pi*x1), RX(pi*x2)", "observable": "<Z0>",
         "L_grid": str([1,2]), "shots_grid": str(["analytic", 128, 1024]), "epochs": VQC_HP.epochs, "lr": VQC_HP.lr,
         "n_params": "6L"},
    ]
    save_summaries_csv(os.path.join(TABLE_DIR, "exp_settings_models.csv"), exp_settings_models)

    vqc_settings = []
    for L in [1, 2]:
        for shots in [None, 128, 1024]:
            vqc_settings.append({
                "L": L,
                "shots": "analytic" if shots is None else shots,
                "n_params": 6 * L,
                "encoding": "RX(pi*x1), RX(pi*x2)",
                "observable": "<Z0>",
                "device": "default.qubit",
            })
    save_summaries_csv(os.path.join(TABLE_DIR, "vqc_settings.csv"), vqc_settings)


if __name__ == "__main__":
    main()
