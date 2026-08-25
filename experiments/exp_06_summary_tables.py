from __future__ import annotations

import csv
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
from tqdm import tqdm

from core.data.xor_dataset import make_split
from core.eval.sweeps import run_repeated
from core.train.trainer import TrainConfig
from core.utils.determinism import set_global_determinism
from core.utils.logging import setup_logger
from core.utils.run_context import create_run_context

from experiments.models_factory import (
    make_linear,
    make_mlp,
    make_vqc,
)

from experiments.settings import (
    DATASET_SEED,
    SPLIT_SEED,
    MODEL_SEEDS,
    BENCH_SIGMA,
    BENCH_N,
    TABLE_DIR,
    CSV_DIR,
    LR_HP,
    MLP_HP,
    VQC_HP,
    MLP_HS,
    VQC_LS,
    VQC_SHOTS_LIST,
    OPTIMIZER,
    ADAM_BETA1,
    ADAM_BETA2,
    ADAM_EPS,
)

from experiments.utils import ensure_output_dirs


def _train_cfg(hp) -> TrainConfig:
    return TrainConfig(
        epochs=int(hp.epochs),
        lr=float(hp.lr),
        optimizer=OPTIMIZER,
        adam_beta1=ADAM_BETA1,
        adam_beta2=ADAM_BETA2,
        adam_eps=ADAM_EPS,
    )


def _write_csv(
    path: str,
    fieldnames: List[str],
    rows: List[Dict[str, Any]],
) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)

    with out.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames,
        )

        writer.writeheader()

        for row in rows:
            writer.writerow(
                {
                    key: row.get(key)
                    for key in fieldnames
                }
            )


def _mean_std(
    values: List[float],
) -> tuple[float, float]:
    arr = np.asarray(
        values,
        dtype=float,
    )

    return (
        float(arr.mean()),
        float(arr.std(ddof=0)),
    )


def main() -> None:
    exp_name = "exp_06_summary_tables"

    run_ctx = create_run_context(
        exp_name=exp_name,
        settings={
            "DATASET_SEED": DATASET_SEED,
            "SPLIT_SEED": SPLIT_SEED,
            "MODEL_SEEDS": list(MODEL_SEEDS),
            "BENCH_SIGMA": BENCH_SIGMA,
            "BENCH_N": BENCH_N,
            "OPTIMIZER": OPTIMIZER,
            "ADAM_BETA1": ADAM_BETA1,
            "ADAM_BETA2": ADAM_BETA2,
            "ADAM_EPS": ADAM_EPS,
            "DETERMINISM_SEED": DATASET_SEED,
        },
    )

    logger = setup_logger(
        exp_name,
        run_ctx.run_id,
        Path("outputs/logs"),
    )

    logger.info(
        f"metadata: {run_ctx.metadata_path}"
    )

    applied_seeds = set_global_determinism(
        DATASET_SEED
    )

    logger.info(
        f"Determinism enforced: {applied_seeds}"
    )

    ensure_output_dirs(
        TABLE_DIR,
        CSV_DIR,
    )

    split = make_split(
        "B",
        sigma=float(BENCH_SIGMA),
        n_per_cluster=int(BENCH_N),
        data_seed=int(DATASET_SEED),
        split_seed=int(SPLIT_SEED),
    )

    # Order intentionally matches manuscript Table 4.
    specs = [
        (
            *make_linear(),
            LR_HP,
            {
                "model_group": "classical",
                "L": None,
                "shots": None,
                "h": None,
            },
        ),
        (
            *make_mlp(h=4),
            MLP_HP,
            {
                "model_group": "classical",
                "L": None,
                "shots": None,
                "h": 4,
            },
        ),
        (
            *make_vqc(L=1, shots=None),
            VQC_HP,
            {
                "model_group": "quantum",
                "L": 1,
                "shots": None,
                "h": None,
            },
        ),
        (
            *make_vqc(L=1, shots=1024),
            VQC_HP,
            {
                "model_group": "quantum",
                "L": 1,
                "shots": 1024,
                "h": None,
            },
        ),
        (
            *make_vqc(L=1, shots=128),
            VQC_HP,
            {
                "model_group": "quantum",
                "L": 1,
                "shots": 128,
                "h": None,
            },
        ),
        (
            *make_vqc(L=2, shots=None),
            VQC_HP,
            {
                "model_group": "quantum",
                "L": 2,
                "shots": None,
                "h": None,
            },
        ),
        (
            *make_vqc(L=2, shots=1024),
            VQC_HP,
            {
                "model_group": "quantum",
                "L": 2,
                "shots": 1024,
                "h": None,
            },
        ),
    ]

    raw_rows: List[Dict[str, Any]] = []
    summary_rows: List[Dict[str, Any]] = []
    table4_rows: List[Dict[str, Any]] = []
    table5_rows: List[Dict[str, Any]] = []

    for spec, factory, hp, extra in tqdm(
        specs,
        desc="Benchmark models",
    ):
        logger.info(
            f"Benchmark start: {spec.name}"
        )

        records, summary = run_repeated(
            model_factory=factory,
            split=split,
            dataset_name="B",
            model_name=spec.name,
            train_cfg=_train_cfg(hp),
            model_seeds=MODEL_SEEDS,
            meta={
                "sigma": float(BENCH_SIGMA),
                "n_per_cluster": int(BENCH_N),
                **extra,
            },
            logger=logger,
        )

        record_dicts = [
            asdict(record)
            for record in records
        ]

        raw_rows.extend(record_dicts)

        summary_row = dict(summary)
        summary_row["model_group"] = extra[
            "model_group"
        ]

        summary_rows.append(summary_row)

        # Reviewer-requested empirical generalization gap:
        # compute per run first, then aggregate.
        gaps = [
            float(row["train_acc"])
            - float(row["test_acc"])
            for row in record_dicts
        ]

        gap_mean, gap_std = _mean_std(gaps)

        table4_rows.append(
            {
                "model": spec.name,
                "train_acc_mean":
                    summary["train_acc_mean"],
                "train_acc_std":
                    summary["train_acc_std"],
                "test_acc_mean":
                    summary["test_acc_mean"],
                "test_acc_std":
                    summary["test_acc_std"],
                "gen_gap_mean":
                    gap_mean,
                "gen_gap_std":
                    gap_std,
                "test_bce_mean":
                    summary["test_loss_mean"],
                "test_bce_std":
                    summary["test_loss_std"],
                "seeds":
                    ",".join(
                        str(seed)
                        for seed in MODEL_SEEDS
                    ),
                "sigma": BENCH_SIGMA,
                "n_per_cluster": BENCH_N,
            }
        )

        table5_rows.append(
            {
                "model": spec.name,
                "n_params_mean":
                    summary["n_params_mean"],
                "n_params_std":
                    summary["n_params_std"],
                "train_seconds_mean":
                    summary["train_seconds_mean"],
                "train_seconds_std":
                    summary["train_seconds_std"],
                "seeds":
                    ",".join(
                        str(seed)
                        for seed in MODEL_SEEDS
                    ),
                "sigma": BENCH_SIGMA,
                "n_per_cluster": BENCH_N,
            }
        )

        logger.info(
            f"Benchmark done: {spec.name}"
        )

    raw_fields = [
        "model_name",
        "model_seed",
        "dataset_name",
        "sigma",
        "n_per_cluster",
        "shots",
        "L",
        "h",
        "train_acc",
        "test_acc",
        "train_loss",
        "test_loss",
        "n_params",
        "train_seconds",
    ]

    _write_csv(
        str(
            Path(CSV_DIR)
            / "summary_benchmark_runs.csv"
        ),
        raw_fields,
        raw_rows,
    )

    summary_fields = [
        "model_name",
        "dataset_name",
        "model_group",
        "sigma",
        "n_per_cluster",
        "shots",
        "L",
        "h",
        "train_acc_mean",
        "train_acc_std",
        "test_acc_mean",
        "test_acc_std",
        "train_loss_mean",
        "train_loss_std",
        "test_loss_mean",
        "test_loss_std",
        "n_params_mean",
        "n_params_std",
        "train_seconds_mean",
        "train_seconds_std",
    ]

    _write_csv(
        str(
            Path(CSV_DIR)
            / "summary_benchmark_meanstd.csv"
        ),
        summary_fields,
        summary_rows,
    )

    _write_csv(
        str(
            Path(TABLE_DIR)
            / "table4.csv"
        ),
        [
            "model",
            "train_acc_mean",
            "train_acc_std",
            "test_acc_mean",
            "test_acc_std",
            "gen_gap_mean",
            "gen_gap_std",
            "test_bce_mean",
            "test_bce_std",
            "seeds",
            "sigma",
            "n_per_cluster",
        ],
        table4_rows,
    )

    _write_csv(
        str(
            Path(TABLE_DIR)
            / "table5.csv"
        ),
        [
            "model",
            "n_params_mean",
            "n_params_std",
            "train_seconds_mean",
            "train_seconds_std",
            "seeds",
            "sigma",
            "n_per_cluster",
        ],
        table5_rows,
    )

    _write_csv(
        str(
            Path(TABLE_DIR)
            / "exp_settings_data.csv"
        ),
        [
            "dataset_seed",
            "split_seed",
            "train_frac",
            "benchmark_dataset",
            "benchmark_sigma",
            "benchmark_n_per_cluster",
        ],
        [
            {
                "dataset_seed": DATASET_SEED,
                "split_seed": SPLIT_SEED,
                "train_frac": 0.80,
                "benchmark_dataset": "B",
                "benchmark_sigma": BENCH_SIGMA,
                "benchmark_n_per_cluster":
                    BENCH_N,
            }
        ],
    )

    _write_csv(
        str(
            Path(TABLE_DIR)
            / "exp_settings_models.csv"
        ),
        [
            "model",
            "epochs",
            "lr",
            "optimizer",
            "hidden_widths",
            "vqc_depths",
            "vqc_shots",
        ],
        [
            {
                "model": "LR",
                "epochs": LR_HP.epochs,
                "lr": LR_HP.lr,
                "optimizer": OPTIMIZER,
                "hidden_widths": "",
                "vqc_depths": "",
                "vqc_shots": "",
            },
            {
                "model": "MLP",
                "epochs": MLP_HP.epochs,
                "lr": MLP_HP.lr,
                "optimizer": OPTIMIZER,
                "hidden_widths":
                    str(list(MLP_HS)),
                "vqc_depths": "",
                "vqc_shots": "",
            },
            {
                "model": "VQC",
                "epochs": VQC_HP.epochs,
                "lr": VQC_HP.lr,
                "optimizer": OPTIMIZER,
                "hidden_widths": "",
                "vqc_depths":
                    str(list(VQC_LS)),
                "vqc_shots":
                    str(
                        [
                            "analytic"
                            if shots is None
                            else shots
                            for shots
                            in VQC_SHOTS_LIST
                        ]
                    ),
            },
        ],
    )

    logger.info(
        "Summary tables written from one "
        "benchmark run set."
    )


if __name__ == "__main__":
    main()