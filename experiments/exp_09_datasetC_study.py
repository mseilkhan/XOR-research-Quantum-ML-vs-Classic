from __future__ import annotations

"""
Experiment 9 — Dataset C study 
Dataset C: uniform in [0,1]^2 with threshold-based XOR:
  y = 1 iff exactly one coordinate > t

We run two sweeps:
  1) threshold sweep: vary t, fixed n
  2) size sweep: vary n, fixed t

Outputs:
  outputs/csv/datasetC_threshold_runs.csv
  outputs/csv/datasetC_threshold_summary.csv
  outputs/figures/datasetC_threshold_testacc.png

  outputs/csv/datasetC_size_runs.csv
  outputs/csv/datasetC_size_summary.csv
  outputs/figures/datasetC_size_testacc.png
"""

import os
import csv
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import matplotlib.pyplot as plt

from core.data.xor_dataset import make_split
from core.train.trainer import TrainConfig, Trainer
from core.utils.io import ensure_dir, append_csv_row, csv_contains_row
from core.viz.style import apply_plot_style, FIGSIZE_WIDE

from experiments.models_factory import make_linear, make_mlp, make_vqc
from experiments.settings import (
    DATASET_SEED,
    SPLIT_SEED,
    MODEL_SEEDS,
    CSV_DIR,
    FIG_DIR,
    LOG_DIR,
    # training HPs
    LR_HP,
    MLP_HP,
    VQC_HP,
    # VQC regimes
    VQC_LS,
    VQC_SHOTS_LIST,
    # Dataset C params
    C_T_BENCH,
    C_N_BENCH,
    C_T_SWEEP,
    C_N_SWEEP,
)

# Optional integrations
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


SMOKE = os.getenv("XOR_SMOKE", "0") == "1"

TH_RUNS = os.path.join(CSV_DIR, "datasetC_threshold_runs.csv")
TH_SUMM = os.path.join(CSV_DIR, "datasetC_threshold_summary.csv")
TH_FIG = os.path.join(FIG_DIR, "datasetC_threshold_testacc.png")

SZ_RUNS = os.path.join(CSV_DIR, "datasetC_size_runs.csv")
SZ_SUMM = os.path.join(CSV_DIR, "datasetC_size_summary.csv")
SZ_FIG = os.path.join(FIG_DIR, "datasetC_size_testacc.png")


RUN_FIELDS = [
    "experiment",
    "dataset",
    "sweep",
    "t",
    "n",
    "model_name",
    "model_seed",
    "L",
    "shots",
    "h",
    "train_acc",
    "train_loss",
    "test_acc",
    "test_loss",
    "n_params",
    "train_seconds",
]

SUM_FIELDS = [
    "experiment",
    "dataset",
    "sweep",
    "t",
    "n",
    "model_name",
    "L",
    "shots",
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
            "C_T_BENCH": float(C_T_BENCH),
            "C_N_BENCH": int(C_N_BENCH),
            "C_T_SWEEP": list(C_T_SWEEP),
            "C_N_SWEEP": list(C_N_SWEEP),
            "DETERMINISM_SEED": int(DATASET_SEED),
            "SMOKE": bool(SMOKE),
        }
        run_ctx = create_run_context(exp_name=exp_name, settings=settings)
        logger = setup_logger(exp_name, run_ctx.run_id, Path(LOG_DIR))
        logger.info(f"metadata: {run_ctx.metadata_path}")
        if set_global_determinism is not None:
            applied = set_global_determinism(int(DATASET_SEED))
            logger.info(f"Determinism enforced: {applied}")
        return logger

    class _Fallback:
        def info(self, msg: str):
            print(msg)

    if set_global_determinism is not None:
        set_global_determinism(int(DATASET_SEED))
    return _Fallback()


def _iter_progress(xs, desc: str):
    xs = list(xs)
    if tqdm is not None:
        return tqdm(xs, desc=desc)
    if SimpleProgress is not None:
        prog = SimpleProgress(total=len(xs), prefix=f"{desc}: ")
        def gen():
            for i, x in enumerate(xs, 1):
                prog.update(i, msg=str(x))
                yield x
            prog.done()
        return gen()
    return xs


def _mean_std(vals: Sequence[float]) -> Tuple[float, float]:
    a = np.array(list(vals), dtype=float)
    return float(a.mean()), float(a.std(ddof=0))


def _run_model_repeated(
    *,
    split,
    model_name: str,
    model_factory,
    train_cfg: TrainConfig,
    model_seeds: Sequence[int],
    meta: Dict[str, Any],
    runs_csv: str,
    summaries: List[Dict[str, Any]],
) -> None:
    trainer = Trainer(train_cfg)
    test_accs: List[float] = []
    test_losses: List[float] = []

    for s in model_seeds:
        m = model_factory(int(s))
        fit_res = trainer.fit(m, split.X_train, split.y_train, split.X_test, split.y_test)
        final = Trainer.final_metrics(m, split.X_train, split.y_train, split.X_test, split.y_test)

        row = dict(meta)
        row.update({
            "model_name": model_name,
            "model_seed": int(s),
            "train_acc": float(final["train_acc"]),
            "train_loss": float(final["train_loss"]),
            "test_acc": float(final["test_acc"]),
            "test_loss": float(final["test_loss"]),
            "n_params": int(fit_res.n_params),
            "train_seconds": float(fit_res.train_seconds),
        })

        key = {
            "sweep": meta["sweep"],
            "t": meta["t"],
            "n": meta["n"],
            "model_name": model_name,
            "model_seed": int(s),
            "L": meta.get("L"),
            "shots": meta.get("shots"),
            "h": meta.get("h"),
        }
        if not csv_contains_row(runs_csv, key):
            append_csv_row(runs_csv, RUN_FIELDS, row)

        test_accs.append(float(final["test_acc"]))
        test_losses.append(float(final["test_loss"]))

    acc_m, acc_s = _mean_std(test_accs)
    loss_m, loss_s = _mean_std(test_losses)

    summ = dict(meta)
    summ.update({
        "model_name": model_name,
        "test_acc_mean": float(acc_m),
        "test_acc_std": float(acc_s),
        "test_loss_mean": float(loss_m),
        "test_loss_std": float(loss_s),
    })
    summaries.append(summ)


def _plot_summary_curve(
    *,
    xs: List[float],
    series: Dict[str, Tuple[List[float], List[float]]],  # name -> (mean, std)
    title: str,
    xlabel: str,
    out_path: str,
) -> None:
    apply_plot_style()
    fig = plt.figure(figsize=FIGSIZE_WIDE)
    ax = plt.gca()

    for name, (m, s) in series.items():
        ax.errorbar(xs, m, yerr=s, fmt="-o", capsize=4, label=name)

    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("test accuracy (mean ± std)")
    ax.set_ylim(0.0, 1.05)
    ax.legend()

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight", facecolor="white", edgecolor="white", transparent=False)
    plt.close(fig)


def main():
    exp_name = "exp_09_datasetC_study"
    logger = _get_logger(exp_name)

    ensure_dir(CSV_DIR)
    ensure_dir(FIG_DIR)

    # Smoke policy: 1 seed, tiny epochs, VQC only analytic (no shots=1024)
    if SMOKE:
        model_seeds = [0]
        lr_cfg = TrainConfig(epochs=min(5, int(LR_HP.epochs)), lr=float(LR_HP.lr))
        mlp_cfg = TrainConfig(epochs=min(5, int(MLP_HP.epochs)), lr=float(MLP_HP.lr))
        vqc_cfg = TrainConfig(epochs=min(2, int(VQC_HP.epochs)), lr=float(VQC_HP.lr))
        vqc_shots_list = [None]
    else:
        model_seeds = list(MODEL_SEEDS)
        lr_cfg = TrainConfig(epochs=int(LR_HP.epochs), lr=float(LR_HP.lr))
        mlp_cfg = TrainConfig(epochs=int(MLP_HP.epochs), lr=float(MLP_HP.lr))
        vqc_cfg = TrainConfig(epochs=int(VQC_HP.epochs), lr=float(VQC_HP.lr))
        vqc_shots_list = list(VQC_SHOTS_LIST)

    # Model factories (Linear, MLP h=4, VQC L in VQC_LS, shots in vqc_shots_list)
    linear_spec, linear_factory = make_linear()
    mlp_spec, mlp_factory = make_mlp(h=4)

    # -------------------------
    # Sweep 1: threshold sweep
    # -------------------------
    logger.info("[INFO] Dataset C threshold sweep start")
    th_summaries: List[Dict[str, Any]] = []

    for t in _iter_progress(C_T_SWEEP, desc="Dataset C t"):
        split = make_split(
            "C",
            n=int(C_N_BENCH),
            t=float(t),
            data_seed=int(DATASET_SEED),
            split_seed=int(SPLIT_SEED),
        )

        base_meta = {
            "experiment": exp_name,
            "dataset": "C",
            "sweep": "threshold",
            "t": float(t),
            "n": int(C_N_BENCH),
        }

        # Linear
        meta = dict(base_meta, L=None, shots=None, h=None)
        _run_model_repeated(
            split=split,
            model_name=linear_spec.name,
            model_factory=linear_factory,
            train_cfg=lr_cfg,
            model_seeds=model_seeds,
            meta=meta,
            runs_csv=TH_RUNS,
            summaries=th_summaries,
        )

        # MLP(h=4)
        meta = dict(base_meta, L=None, shots=None, h=4)
        _run_model_repeated(
            split=split,
            model_name=mlp_spec.name,
            model_factory=mlp_factory,
            train_cfg=mlp_cfg,
            model_seeds=model_seeds,
            meta=meta,
            runs_csv=TH_RUNS,
            summaries=th_summaries,
        )

        # VQC
        for L in VQC_LS:
            for shots in vqc_shots_list:
                vqc_spec, vqc_factory = make_vqc(L=int(L), shots=shots)
                meta = dict(base_meta, L=int(L), shots=shots, h=None)
                _run_model_repeated(
                    split=split,
                    model_name=vqc_spec.name,
                    model_factory=vqc_factory,
                    train_cfg=vqc_cfg,
                    model_seeds=model_seeds,
                    meta=meta,
                    runs_csv=TH_RUNS,
                    summaries=th_summaries,
                )

    # Write threshold summary CSV (append unique)
    for s in th_summaries:
        key = {
            "sweep": s["sweep"],
            "t": s["t"],
            "n": s["n"],
            "model_name": s["model_name"],
            "L": s.get("L"),
            "shots": s.get("shots"),
            "h": s.get("h"),
        }
        if not csv_contains_row(TH_SUMM, key):
            append_csv_row(TH_SUMM, SUM_FIELDS, s)

    # Plot threshold sweep (test acc)
    xs_t = [float(x) for x in C_T_SWEEP]
    series_t: Dict[str, Tuple[List[float], List[float]]] = {}

    def _collect(name_filter: str):
        ms, ss = [], []
        for t in xs_t:
            rows = [r for r in th_summaries if r["model_name"] == name_filter and float(r["t"]) == float(t)]
            # if multiple entries (e.g., VQC L/shots), this collector is for unique names only
            if len(rows) != 1:
                return None
            ms.append(float(rows[0]["test_acc_mean"]))
            ss.append(float(rows[0]["test_acc_std"]))
        return ms, ss

    # Linear & MLP are unique
    series_t[linear_spec.name] = _collect(linear_spec.name)
    series_t[mlp_spec.name] = _collect(mlp_spec.name)

    # For VQC, include each regime as separate curve
    for L in VQC_LS:
        for shots in vqc_shots_list:
            tag = f"VQC(L={L},{'analytic' if shots is None else f'shots={shots}'})"
            ms, ss = [], []
            for t in xs_t:
                rows = [
                    r for r in th_summaries
                    if (r["model_name"].startswith("VQC") and int(r["L"]) == int(L) and r.get("shots") == shots and float(r["t"]) == float(t))
                ]
                # exactly one per t
                ms.append(float(rows[0]["test_acc_mean"]))
                ss.append(float(rows[0]["test_acc_std"]))
            series_t[tag] = (ms, ss)

    # remove None entries (paranoia)
    series_t = {k: v for k, v in series_t.items() if v is not None}

    _plot_summary_curve(
        xs=xs_t,
        series=series_t,
        title=f"Dataset C — threshold sweep (n={C_N_BENCH})",
        xlabel="threshold t",
        out_path=TH_FIG,
    )

    logger.info(f"[INFO] Saved: {TH_RUNS}")
    logger.info(f"[INFO] Saved: {TH_SUMM}")
    logger.info(f"[INFO] Saved: {TH_FIG}")
    logger.info("[INFO] Dataset C threshold sweep done")

    # -------------------------
    # Sweep 2: size sweep
    # -------------------------
    logger.info("[INFO] Dataset C size sweep start")
    sz_summaries: List[Dict[str, Any]] = []

    for n in _iter_progress(C_N_SWEEP, desc="Dataset C n"):
        split = make_split(
            "C",
            n=int(n),
            t=float(C_T_BENCH),
            data_seed=int(DATASET_SEED),
            split_seed=int(SPLIT_SEED),
        )

        base_meta = {
            "experiment": exp_name,
            "dataset": "C",
            "sweep": "size",
            "t": float(C_T_BENCH),
            "n": int(n),
        }

        # Linear
        meta = dict(base_meta, L=None, shots=None, h=None)
        _run_model_repeated(
            split=split,
            model_name=linear_spec.name,
            model_factory=linear_factory,
            train_cfg=lr_cfg,
            model_seeds=model_seeds,
            meta=meta,
            runs_csv=SZ_RUNS,
            summaries=sz_summaries,
        )

        # MLP(h=4)
        meta = dict(base_meta, L=None, shots=None, h=4)
        _run_model_repeated(
            split=split,
            model_name=mlp_spec.name,
            model_factory=mlp_factory,
            train_cfg=mlp_cfg,
            model_seeds=model_seeds,
            meta=meta,
            runs_csv=SZ_RUNS,
            summaries=sz_summaries,
        )

        # VQC
        for L in VQC_LS:
            for shots in vqc_shots_list:
                vqc_spec, vqc_factory = make_vqc(L=int(L), shots=shots)
                meta = dict(base_meta, L=int(L), shots=shots, h=None)
                _run_model_repeated(
                    split=split,
                    model_name=vqc_spec.name,
                    model_factory=vqc_factory,
                    train_cfg=vqc_cfg,
                    model_seeds=model_seeds,
                    meta=meta,
                    runs_csv=SZ_RUNS,
                    summaries=sz_summaries,
                )

    # Write size summary CSV (append unique)
    for s in sz_summaries:
        key = {
            "sweep": s["sweep"],
            "t": s["t"],
            "n": s["n"],
            "model_name": s["model_name"],
            "L": s.get("L"),
            "shots": s.get("shots"),
            "h": s.get("h"),
        }
        if not csv_contains_row(SZ_SUMM, key):
            append_csv_row(SZ_SUMM, SUM_FIELDS, s)

    # Plot size sweep
    xs_n = [float(x) for x in C_N_SWEEP]
    series_n: Dict[str, Tuple[List[float], List[float]]] = {}

    # Linear & MLP
    for name in [linear_spec.name, mlp_spec.name]:
        ms, ss = [], []
        for n in xs_n:
            rows = [r for r in sz_summaries if r["model_name"] == name and float(r["n"]) == float(n)]
            ms.append(float(rows[0]["test_acc_mean"]))
            ss.append(float(rows[0]["test_acc_std"]))
        series_n[name] = (ms, ss)

    # VQC regimes
    for L in VQC_LS:
        for shots in vqc_shots_list:
            tag = f"VQC(L={L},{'analytic' if shots is None else f'shots={shots}'})"
            ms, ss = [], []
            for n in xs_n:
                rows = [
                    r for r in sz_summaries
                    if (r["model_name"].startswith("VQC") and int(r["L"]) == int(L) and r.get("shots") == shots and float(r["n"]) == float(n))
                ]
                ms.append(float(rows[0]["test_acc_mean"]))
                ss.append(float(rows[0]["test_acc_std"]))
            series_n[tag] = (ms, ss)

    _plot_summary_curve(
        xs=xs_n,
        series=series_n,
        title=f"Dataset C — size sweep (t={C_T_BENCH})",
        xlabel="dataset size n",
        out_path=SZ_FIG,
    )

    logger.info(f"[INFO] Saved: {SZ_RUNS}")
    logger.info(f"[INFO] Saved: {SZ_SUMM}")
    logger.info(f"[INFO] Saved: {SZ_FIG}")
    logger.info("[INFO] Dataset C size sweep done")
    logger.info("[INFO] Experiment 9 done.")


if __name__ == "__main__":
    main()