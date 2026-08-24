#!/usr/bin/env python3
from __future__ import annotations

"""
Canonical experiment 07 — expanded MLP width ablation for revised Fig. 14.

Goal
----
The original Fig. 14 was based on a limited width ablation. For the revision,
we preserve the original two-panel logic, but we run a broader experiment so
that the conclusions are better supported and the meaning of the uncertainty
visualization is unambiguous.

What this script does
---------------------
1. Uses the CURRENT repository methodology and settings:
   - Dataset B
   - sigma = BENCH_SIGMA
   - n_per_cluster = BENCH_N
   - same fixed dataset/split seeds from experiments.settings
   - same MLP training hyperparameters from MLP_HP / OPTIMIZER
2. Trains MLPs for a broader width grid.
3. Repeats each width over a larger set of model seeds.
4. Recreates the same figure concept as the manuscript:
   (a) final test accuracy vs hidden width
   (b) test BCE vs epoch
5. Also creates stronger support figures:
   - final test BCE vs hidden width
   - train/test generalization gap vs hidden width
   - distribution/variability plots across seeds
6. Saves all raw runs, per-epoch histories, summaries, and metadata.

Recommended publication use
---------------------------
- Keep the revised two-panel figure as the replacement for Fig. 14.
- Use the companion support figures internally to answer reviewer questions,
  and optionally move one of them to appendix/supplement if desired.

Run from repository root:
    unset XOR_SMOKE
    python -m experiments.exp_07_mlp_width_ablation

Quick smoke-style test only:
    python -m experiments.exp_07_mlp_width_ablation --widths 1 2 4 --seeds 0 1 --epochs 30

DO NOT use a quick-test run for the manuscript.
"""

import argparse
import hashlib
import json
import math
import os
import platform
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from core.data.xor_dataset import make_split
from core.train.trainer import TrainConfig, Trainer
from core.utils.determinism import set_global_determinism
from core.viz.style import apply_plot_style
from experiments.models_factory import make_mlp
from experiments.settings import (
    DATASET_SEED,
    SPLIT_SEED,
    BENCH_SIGMA,
    BENCH_N,
    MLP_HP,
    MLP_HS,
    MLP_CURVE_HS,
    WIDTH_ABLATION_SEEDS,
    OPTIMIZER,
    ADAM_BETA1,
    ADAM_BETA2,
    ADAM_EPS,
    OUT_ROOT,
)

# Canonical width-ablation settings are centralized in experiments.settings.
DEFAULT_WIDTHS = list(MLP_HS)
DEFAULT_SEEDS = list(WIDTH_ABLATION_SEEDS)
# For the BCE learning-curve panel, plotting every width would be too crowded.
DEFAULT_CURVE_WIDTHS = list(MLP_CURVE_HS)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--out-root",
        default=f"{OUT_ROOT}/revision_fig11_expanded",
    )
    p.add_argument("--widths", nargs="*", type=int, default=DEFAULT_WIDTHS)
    p.add_argument("--seeds", nargs="*", type=int, default=DEFAULT_SEEDS)
    p.add_argument(
        "--curve-widths",
        nargs="*",
        type=int,
        default=DEFAULT_CURVE_WIDTHS,
        help="Subset of widths to show in panel (b) learning curves.",
    )
    p.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Override only for testing. Publication default is MLP_HP.epochs."
    )
    p.add_argument(
        "--lr",
        type=float,
        default=None,
        help="Override only for testing. Publication default is MLP_HP.lr."
    )
    return p.parse_args()


def ensure_repo_root():
    if not Path("core").exists() or not Path("experiments").exists():
        raise SystemExit("Run this script from the repository root.")


def git_commit():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return "unavailable"


def git_status_porcelain():
    try:
        return subprocess.check_output(
            ["git", "status", "--porcelain"],
            text=True,
            stderr=subprocess.DEVNULL,
        )
    except Exception:
        return "unavailable"


def package_version(name):
    try:
        from importlib.metadata import version
        return version(name)
    except Exception:
        return "unavailable"


def sha256_file(path: Path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def write_manifest(root: Path):
    rows = []
    for p in sorted(root.rglob("*")):
        if p.is_file() and p.name != "SHA256SUMS.txt":
            rows.append(f"{sha256_file(p)}  {p.relative_to(root).as_posix()}")
    (root / "SHA256SUMS.txt").write_text(
        "\n".join(rows) + "\n", encoding="utf-8"
    )


def mean_sd_pop(x):
    arr = np.asarray(x, dtype=float)
    return float(np.mean(arr)), float(np.std(arr, ddof=0))


def mean_sem_pop(x):
    arr = np.asarray(x, dtype=float)
    sd = float(np.std(arr, ddof=0))
    n = len(arr)
    sem = sd / math.sqrt(n) if n > 0 else float("nan")
    return float(np.mean(arr)), sd, sem


def train_all(widths, seeds, epochs, lr, split):
    run_rows = []
    hist_rows = []

    for h in widths:
        print(f"\nWidth h={h}")
        spec, factory = make_mlp(h=h)

        for seed in seeds:
            print(f"  seed={seed}")
            model = factory(seed)

            trainer = Trainer(
                TrainConfig(
                    epochs=epochs,
                    lr=lr,
                    optimizer=OPTIMIZER,
                    adam_beta1=ADAM_BETA1,
                    adam_beta2=ADAM_BETA2,
                    adam_eps=ADAM_EPS,
                )
            )
            fit = trainer.fit(
                model,
                split.X_train,
                split.y_train,
                split.X_test,
                split.y_test,
            )

            final = Trainer.final_metrics(
                model,
                split.X_train,
                split.y_train,
                split.X_test,
                split.y_test,
            )

            gen_gap = float(final["train_acc"] - final["test_acc"])

            run_rows.append(
                {
                    "h": int(h),
                    "seed": int(seed),
                    "model_name": spec.name,
                    "train_acc": float(final["train_acc"]),
                    "test_acc": float(final["test_acc"]),
                    "train_bce": float(final["train_loss"]),
                    "test_bce": float(final["test_loss"]),
                    "gen_gap": gen_gap,
                    "n_params": int(fit.n_params),
                    "train_seconds": float(fit.train_seconds),
                }
            )

            hist = fit.history
            n_epochs = len(hist["test_loss"])
            for i in range(n_epochs):
                hist_rows.append(
                    {
                        "h": int(h),
                        "seed": int(seed),
                        "epoch": i + 1,
                        "train_bce": float(hist["train_loss"][i]),
                        "test_bce": float(hist["test_loss"][i]),
                        "train_acc": float(hist["train_acc"][i]),
                        "test_acc": float(hist["test_acc"][i]),
                    }
                )

    return pd.DataFrame(run_rows), pd.DataFrame(hist_rows)


def summarize_runs(runs: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for h in sorted(runs["h"].unique()):
        g = runs[runs["h"] == h]
        train_acc_m, train_acc_sd, train_acc_sem = mean_sem_pop(g["train_acc"])
        test_acc_m, test_acc_sd, test_acc_sem = mean_sem_pop(g["test_acc"])
        train_bce_m, train_bce_sd, train_bce_sem = mean_sem_pop(g["train_bce"])
        test_bce_m, test_bce_sd, test_bce_sem = mean_sem_pop(g["test_bce"])
        gap_m, gap_sd, gap_sem = mean_sem_pop(g["gen_gap"])
        sec_m, sec_sd, sec_sem = mean_sem_pop(g["train_seconds"])

        rows.append(
            {
                "h": int(h),
                "n_runs": int(len(g)),
                "n_params": int(g["n_params"].iloc[0]),

                "train_acc_mean": train_acc_m,
                "train_acc_std": train_acc_sd,
                "train_acc_sem": train_acc_sem,

                "test_acc_mean": test_acc_m,
                "test_acc_std": test_acc_sd,
                "test_acc_sem": test_acc_sem,

                "train_bce_mean": train_bce_m,
                "train_bce_std": train_bce_sd,
                "train_bce_sem": train_bce_sem,

                "test_bce_mean": test_bce_m,
                "test_bce_std": test_bce_sd,
                "test_bce_sem": test_bce_sem,

                "gen_gap_mean": gap_m,
                "gen_gap_std": gap_sd,
                "gen_gap_sem": gap_sem,

                "train_seconds_mean": sec_m,
                "train_seconds_std": sec_sd,
                "train_seconds_sem": sec_sem,
            }
        )
    return pd.DataFrame(rows).sort_values("h").reset_index(drop=True)


def summarize_curves(histories: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (h, epoch), g in histories.groupby(["h", "epoch"], sort=True):
        for metric in ["train_bce", "test_bce", "train_acc", "test_acc"]:
            pass

        row = {"h": int(h), "epoch": int(epoch)}
        for metric in ["train_bce", "test_bce", "train_acc", "test_acc"]:
            arr = g[metric].to_numpy(dtype=float)
            row[f"{metric}_mean"] = float(np.mean(arr))
            row[f"{metric}_std"] = float(np.std(arr, ddof=0))
            row[f"{metric}_sem"] = (
                float(np.std(arr, ddof=0) / math.sqrt(len(arr)))
                if len(arr) > 0 else float("nan")
            )
        rows.append(row)

    return pd.DataFrame(rows).sort_values(["h", "epoch"]).reset_index(drop=True)


def save_csvs(data_dir, runs, histories, summary, curves):
    runs.to_csv(
        data_dir / "fig14_expanded_runs.csv",
        index=False,
        float_format="%.17g",
    )
    histories.to_csv(
        data_dir / "fig14_expanded_histories_long.csv.gz",
        index=False,
        compression="gzip",
        float_format="%.17g",
    )
    summary.to_csv(
        data_dir / "fig14_expanded_summary.csv",
        index=False,
        float_format="%.17g",
    )
    curves.to_csv(
        data_dir / "fig14_expanded_curve_summary.csv",
        index=False,
        float_format="%.17g",
    )


def make_publication_figure(fig_dir, summary, curves, curve_widths, sigma, n_per_cluster, n_seeds):
    """
    This is the manuscript replacement: same conceptual structure as the old
    Fig. 14, but based on the broader experiment.
    """
    apply_plot_style()
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.7))

    # Panel (a): final test accuracy vs width (all widths).
    ax = axes[0]
    ax.errorbar(
        summary["h"],
        summary["test_acc_mean"],
        yerr=summary["test_acc_std"],
        fmt="-o",
        capsize=4,
        linewidth=1.5,
        markersize=4.5,
    )
    ax.set_xlabel("hidden units $h$")
    ax.set_ylabel("final test accuracy")
    ax.set_title("(a) Final test accuracy vs width")
    ax.set_xticks(summary["h"].tolist())
    ax.set_ylim(0.0, 1.05)

    # Panel (b): test BCE learning curves for a representative subset.
    ax = axes[1]
    for h in curve_widths:
        g = curves[curves["h"] == h]
        if g.empty:
            continue
        x = g["epoch"].to_numpy()
        m = g["test_bce_mean"].to_numpy()
        s = g["test_bce_std"].to_numpy()
        line = ax.plot(x, m, linewidth=1.4, label=rf"$h={h}$")[0]
        ax.fill_between(
            x,
            np.maximum(0.0, m - s),
            m + s,
            alpha=0.16,
        )
    ax.set_xlabel("epoch")
    ax.set_ylabel("test BCE")
    ax.set_title("(b) Test BCE across training")
    ax.legend(ncol=2)

    fig.suptitle(
        r"Expanded MLP width ablation on Dataset B "
        rf"($\sigma={sigma:.2f}$, $n={n_per_cluster}$ per cluster; {n_seeds} seeds)"
    )
    fig.savefig(
        fig_dir / "fig14_mlp_width_ablation_expanded.png",
        dpi=300,
        bbox_inches="tight",
        facecolor="white",
    )
    fig.savefig(
        fig_dir / "fig14_mlp_width_ablation_expanded.pdf",
        bbox_inches="tight",
        facecolor="white",
    )
    plt.close(fig)


def make_support_figure(fig_dir, summary, runs, curves, sigma, n_per_cluster, n_seeds):
    """
    Stronger supporting figure for internal audit / possible appendix.
    """
    apply_plot_style()
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 8.6))
    axes = axes.ravel()

    # (a) final test accuracy mean ± SD vs width
    ax = axes[0]
    ax.errorbar(
        summary["h"],
        summary["test_acc_mean"],
        yerr=summary["test_acc_std"],
        fmt="-o",
        capsize=4,
        linewidth=1.5,
        markersize=4.5,
    )
    ax.set_xlabel("hidden units $h$")
    ax.set_ylabel("final test accuracy")
    ax.set_xticks(summary["h"].tolist())
    ax.set_ylim(0.0, 1.05)
    ax.set_title("(a) Final test accuracy (mean ± 1 SD)")

    # (b) final test BCE mean ± SD vs width
    ax = axes[1]
    ax.errorbar(
        summary["h"],
        summary["test_bce_mean"],
        yerr=summary["test_bce_std"],
        fmt="-o",
        capsize=4,
        linewidth=1.5,
        markersize=4.5,
    )
    ax.set_xlabel("hidden units $h$")
    ax.set_ylabel("final test BCE")
    ax.set_xticks(summary["h"].tolist())
    ax.set_title("(b) Final test BCE (mean ± 1 SD)")

    # (c) generalization gap mean ± SD vs width
    ax = axes[2]
    ax.errorbar(
        summary["h"],
        summary["gen_gap_mean"],
        yerr=summary["gen_gap_std"],
        fmt="-o",
        capsize=4,
        linewidth=1.5,
        markersize=4.5,
    )
    ax.axhline(0.0, linewidth=1.0)
    ax.set_xlabel("hidden units $h$")
    ax.set_ylabel(r"$A_{\mathrm{train}}-A_{\mathrm{test}}$")
    ax.set_xticks(summary["h"].tolist())
    ax.set_title("(c) Empirical generalization gap")

    # (d) seed-level distribution plot for final test accuracy
    ax = axes[3]
    widths = sorted(runs["h"].unique())
    data = [runs.loc[runs["h"] == h, "test_acc"].to_numpy(dtype=float) for h in widths]
    ax.boxplot(data, positions=np.arange(1, len(widths) + 1), widths=0.55)
    # Overlay seed-level points with deterministic jitter.
    rng = np.random.default_rng(0)
    for i, h in enumerate(widths, start=1):
        vals = runs.loc[runs["h"] == h, "test_acc"].to_numpy(dtype=float)
        jitter = rng.uniform(-0.12, 0.12, size=len(vals))
        ax.plot(
            np.full_like(vals, i, dtype=float) + jitter,
            vals,
            "o",
            markersize=3.0,
            alpha=0.65,
        )
    ax.set_xticks(np.arange(1, len(widths) + 1))
    ax.set_xticklabels(widths)
    ax.set_xlabel("hidden units $h$")
    ax.set_ylabel("final test accuracy")
    ax.set_ylim(0.0, 1.05)
    ax.set_title("(d) Distribution across seeds")

    fig.suptitle(
        r"Expanded MLP width study on Dataset B "
        rf"($\sigma={sigma:.2f}$, $n={n_per_cluster}$ per cluster; {n_seeds} seeds)"
    )
    fig.savefig(
        fig_dir / "fig14_mlp_width_support_expanded.png",
        dpi=300,
        bbox_inches="tight",
        facecolor="white",
    )
    fig.savefig(
        fig_dir / "fig14_mlp_width_support_expanded.pdf",
        bbox_inches="tight",
        facecolor="white",
    )
    plt.close(fig)


def make_exact_single_panels(fig_dir, summary, curves, curve_widths, sigma, n_per_cluster, n_seeds):
    """
    For drop-in compatibility with existing LaTeX if needed.
    """
    apply_plot_style()

    fig, ax = plt.subplots(figsize=(6, 5.4))
    ax.errorbar(
        summary["h"],
        summary["test_acc_mean"],
        yerr=summary["test_acc_std"],
        fmt="-o",
        capsize=4,
        linewidth=1.5,
        markersize=4.5,
    )
    ax.set_xlabel("hidden units $h$")
    ax.set_ylabel("final test accuracy")
    ax.set_xticks(summary["h"].tolist())
    ax.set_ylim(0.0, 1.05)
    ax.set_title(
        rf"MLP width ablation (Dataset B: $\sigma={sigma:.2f}$, $n={n_per_cluster}$)"
    )
    fig.savefig(
        fig_dir / "mlpH_acc_vs_h_expanded.png",
        dpi=300,
        bbox_inches="tight",
        facecolor="white",
    )
    fig.savefig(
        fig_dir / "mlpH_acc_vs_h_expanded.pdf",
        bbox_inches="tight",
        facecolor="white",
    )
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 5.4))
    for h in curve_widths:
        g = curves[curves["h"] == h]
        if g.empty:
            continue
        x = g["epoch"].to_numpy()
        m = g["test_bce_mean"].to_numpy()
        s = g["test_bce_std"].to_numpy()
        ax.plot(x, m, linewidth=1.4, label=rf"$h={h}$")
        ax.fill_between(
            x,
            np.maximum(0.0, m - s),
            m + s,
            alpha=0.16,
        )
    ax.set_xlabel("epoch")
    ax.set_ylabel("test BCE")
    ax.set_title(
        rf"MLP test BCE across seeds "
        rf"(Dataset B: $\sigma={sigma:.2f}$, $n={n_per_cluster}$)"
    )
    ax.legend(ncol=2)
    fig.savefig(
        fig_dir / "mlpH_testloss_all_h_expanded.png",
        dpi=300,
        bbox_inches="tight",
        facecolor="white",
    )
    fig.savefig(
        fig_dir / "mlpH_testloss_all_h_expanded.pdf",
        bbox_inches="tight",
        facecolor="white",
    )
    plt.close(fig)


def main():
    args = parse_args()
    ensure_repo_root()

    if os.getenv("XOR_SMOKE", "0") == "1":
        raise SystemExit("Disable XOR_SMOKE for publication runs.")

    widths = sorted(set(int(x) for x in args.widths))
    seeds = sorted(set(int(x) for x in args.seeds))
    curve_widths = [int(x) for x in args.curve_widths if int(x) in widths]

    if not widths:
        raise SystemExit("At least one width is required.")
    if not seeds:
        raise SystemExit("At least one model seed is required.")
    if not curve_widths:
        raise SystemExit("curve-widths must overlap with widths.")

    epochs = int(MLP_HP.epochs if args.epochs is None else args.epochs)
    lr = float(MLP_HP.lr if args.lr is None else args.lr)

    if args.epochs is not None or args.lr is not None:
        print("WARNING: hyperparameters overridden. Do not use for publication.")

    out_root = Path(args.out_root)
    fig_dir = out_root / "figures"
    data_dir = out_root / "data"
    meta_dir = out_root / "metadata"
    for d in [fig_dir, data_dir, meta_dir]:
        d.mkdir(parents=True, exist_ok=True)

    applied = set_global_determinism(int(DATASET_SEED))

    split = make_split(
        "B",
        sigma=float(BENCH_SIGMA),
        n_per_cluster=int(BENCH_N),
        data_seed=int(DATASET_SEED),
        split_seed=int(SPLIT_SEED),
    )

    print("=" * 80)
    print("Expanded Fig. 14 rebuild")
    print(f"Dataset B, sigma={BENCH_SIGMA}, n_per_cluster={BENCH_N}")
    print(f"Widths: {widths}")
    print(f"Seeds: {seeds}")
    print(f"Curve widths: {curve_widths}")
    print(f"epochs={epochs}, lr={lr}, optimizer={OPTIMIZER}")
    print("=" * 80)

    runs, histories = train_all(
        widths=widths,
        seeds=seeds,
        epochs=epochs,
        lr=lr,
        split=split,
    )
    summary = summarize_runs(runs)
    curves = summarize_curves(histories)

    save_csvs(data_dir, runs, histories, summary, curves)
    make_publication_figure(
        fig_dir, summary, curves, curve_widths,
        sigma=float(BENCH_SIGMA),
        n_per_cluster=int(BENCH_N),
        n_seeds=len(seeds),
    )
    make_support_figure(
        fig_dir, summary, runs, curves,
        sigma=float(BENCH_SIGMA),
        n_per_cluster=int(BENCH_N),
        n_seeds=len(seeds),
    )
    make_exact_single_panels(
        fig_dir, summary, curves, curve_widths,
        sigma=float(BENCH_SIGMA),
        n_per_cluster=int(BENCH_N),
        n_seeds=len(seeds),
    )

    # Cross-check width h=4 against revised Table 4 if present.
    h4_row = None
    if 4 in summary["h"].tolist():
        h4_row = summary.loc[summary["h"] == 4].iloc[0].to_dict()

    metadata = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "purpose": "QMI revision — Reviewer Comment 8 / expanded Fig. 14",
        "git_commit": git_commit(),
        "git_status_porcelain": git_status_porcelain(),
        "dataset": {
            "name": "B",
            "sigma": float(BENCH_SIGMA),
            "n_per_cluster": int(BENCH_N),
            "dataset_seed": int(DATASET_SEED),
            "split_seed": int(SPLIT_SEED),
        },
        "experiment_design": {
            "widths": widths,
            "model_seeds": seeds,
            "curve_widths": curve_widths,
            "n_widths": len(widths),
            "n_seeds": len(seeds),
            "n_total_runs": len(widths) * len(seeds),
        },
        "training": {
            "epochs": epochs,
            "learning_rate": lr,
            "optimizer": OPTIMIZER,
            "adam_beta1": float(ADAM_BETA1),
            "adam_beta2": float(ADAM_BETA2),
            "adam_eps": float(ADAM_EPS),
        },
        "uncertainty_definition": {
            "manuscript_panel_a": "mean final test accuracy ± 1 population SD across seeds",
            "manuscript_panel_b": "mean test BCE at each epoch ± 1 population SD across seeds",
            "support_panel_b": "mean final test BCE ± 1 population SD across seeds",
            "support_panel_c": "mean empirical generalization gap ± 1 population SD across seeds",
            "support_panel_d": "boxplot and seed-level points of final test accuracy",
            "ddof": 0,
        },
        "h4_crosscheck": h4_row,
        "determinism": applied,
        "package_versions": {
            "python": sys.version,
            "platform": platform.platform(),
            "numpy": package_version("numpy"),
            "pandas": package_version("pandas"),
            "matplotlib": package_version("matplotlib"),
            "pennylane": package_version("pennylane"),
        },
        "outputs": {
            "publication_figure_png": str(fig_dir / "fig14_mlp_width_ablation_expanded.png"),
            "support_figure_png": str(fig_dir / "fig14_mlp_width_support_expanded.png"),
            "summary_csv": str(data_dir / "fig14_expanded_summary.csv"),
            "runs_csv": str(data_dir / "fig14_expanded_runs.csv"),
            "curve_summary_csv": str(data_dir / "fig14_expanded_curve_summary.csv"),
            "histories_csv_gz": str(data_dir / "fig14_expanded_histories_long.csv.gz"),
        },
    }

    (meta_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )

    try:
        shutil.copy2(Path(__file__), meta_dir / Path(__file__).name)
    except Exception:
        pass

    write_manifest(out_root)

    print("\nFINAL SUMMARY")
    print(summary.to_string(index=False))
    if h4_row is not None:
        print("\nCross-check width h=4:")
        print(h4_row)

    print("\nPrimary manuscript replacement:")
    print(fig_dir / "fig14_mlp_width_ablation_expanded.png")
    print("\nAdditional support figure:")
    print(fig_dir / "fig14_mlp_width_support_expanded.png")
    print("\nSend the whole directory back for audit:")
    print(out_root)


if __name__ == "__main__":
    main()
