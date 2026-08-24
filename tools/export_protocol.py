from __future__ import annotations

import csv
from pathlib import Path

from experiments.settings import (
    LR_HP,
    MLP_HP,
    VQC_HP,
    MODEL_SEEDS,
    SEED_SENSITIVITY_SEEDS,
    WIDTH_ABLATION_SEEDS,
    MLP_HS,
    OPTIMIZER,
    SMOKE,
)


ROOT = Path(__file__).resolve().parents[1]

PAPER_DIR = ROOT / "paper"

CSV_PATH = (
    PAPER_DIR
    / "experiment_protocol.csv"
)

TEX_PATH = (
    PAPER_DIR
    / "experiment_protocol_rows.tex"
)


def seed_range_label(
    seeds: list[int],
) -> str:
    if not seeds:
        return "none"

    if seeds == list(
        range(
            seeds[0],
            seeds[-1] + 1,
        )
    ):
        return (
            f"{seeds[0]}--{seeds[-1]}"
        )

    return ",".join(
        str(seed)
        for seed in seeds
    )


def optimizer_label() -> str:
    if OPTIMIZER == "gd":
        return "full-batch GD"

    if OPTIMIZER == "adam":
        return "Adam"

    raise ValueError(
        f"Unsupported optimizer: {OPTIMIZER}"
    )


def main() -> None:
    if SMOKE:
        raise RuntimeError(
            "Protocol export must be run "
            "without XOR_SMOKE so that the "
            "manuscript table reflects the "
            "full experimental protocol."
        )

    standard_seeds = seed_range_label(
        list(MODEL_SEEDS)
    )

    sensitivity_seeds = seed_range_label(
        list(SEED_SENSITIVITY_SEEDS)
    )

    width_seeds = seed_range_label(
        list(WIDTH_ABLATION_SEEDS)
    )

    optimizer = optimizer_label()

    rows = [
        {
            "experiment": "Exp. 1",
            "models":
                r"MLP($h=4$); VQC($L=1,2$)",
            "epochs":
                f"MLP {MLP_HP.epochs}; "
                f"VQC {VQC_HP.epochs}",
            "lr": f"{MLP_HP.lr:.1f}",
            "optimizer": optimizer,
            "seeds": standard_seeds,
            "shots":
                r"analytic training; "
                r"1024-shot VQC evaluation",
        },
        {
            "experiment": "Exp. 2",
            "models":
                r"MLP($h=4$); VQC($L=1,2$)",
            "epochs":
                f"MLP {MLP_HP.epochs}; "
                f"VQC {VQC_HP.epochs}",
            "lr": f"{MLP_HP.lr:.1f}",
            "optimizer": optimizer,
            "seeds":
                r"MLP/$L=2$: "
                + standard_seeds
                + r"; $L=1$: 0",
            "shots":
                r"$L=1$: analytic, 128; "
                r"$L=2$: analytic",
        },
        {
            "experiment": "Exp. 3",
            "models":
                r"LR; MLP($h=4$); "
                r"VQC($L=1,2$)",
            "epochs":
                f"LR {LR_HP.epochs}; "
                f"MLP {MLP_HP.epochs}; "
                f"VQC {VQC_HP.epochs}",
            "lr": f"{LR_HP.lr:.1f}",
            "optimizer": optimizer,
            "seeds": standard_seeds,
            "shots":
                r"VQC: 1024; "
                r"classical: n/a",
        },
        {
            "experiment": "Exp. 4",
            "models":
                r"VQC($L=1$)",
            "epochs":
                f"VQC {VQC_HP.epochs}",
            "lr": f"{VQC_HP.lr:.1f}",
            "optimizer": optimizer,
            "seeds": standard_seeds,
            "shots":
                r"analytic, 128, 1024",
        },
        {
            "experiment": "Exp. 5",
            "models":
                r"LR; MLP($h=4$); "
                r"VQC($L=1,2$)",
            "epochs":
                f"LR {LR_HP.epochs}; "
                f"MLP {MLP_HP.epochs}; "
                f"VQC {VQC_HP.epochs}",
            "lr": f"{LR_HP.lr:.1f}",
            "optimizer": optimizer,
            "seeds":
                sensitivity_seeds,
            "shots":
                r"VQC: analytic, 1024; "
                r"classical: n/a",
        },
        {
            "experiment": "Exp. 6",
            "models":
                r"LR; MLP($h=4$); "
                r"VQC($L=1,2$)",
            "epochs":
                f"LR {LR_HP.epochs}; "
                f"MLP {MLP_HP.epochs}; "
                f"VQC {VQC_HP.epochs}",
            "lr": f"{LR_HP.lr:.1f}",
            "optimizer": optimizer,
            "seeds": standard_seeds,
            "shots":
                r"$L=1$: analytic, 128, 1024; "
                r"$L=2$: analytic, 1024",
        },
        {
            "experiment": "Exp. 7",
            "models":
                r"MLP($h\in\{"
                + ",".join(
                    map(
                        str,
                        MLP_HS,
                    )
                )
                + r"\}$)",
            "epochs":
                f"MLP {MLP_HP.epochs}",
            "lr": f"{MLP_HP.lr:.1f}",
            "optimizer": optimizer,
            "seeds": width_seeds,
            "shots": r"n/a",
        },
        {
            "experiment": "Exp. 8",
            "models":
                r"LR; MLP($h=4$); "
                r"VQC($L=1,2$)",
            "epochs":
                f"LR {LR_HP.epochs}; "
                f"MLP {MLP_HP.epochs}; "
                f"VQC {VQC_HP.epochs}",
            "lr": f"{LR_HP.lr:.1f}",
            "optimizer": optimizer,
            "seeds": "0",
            "shots":
                r"VQC: analytic; "
                r"classical: n/a",
        },
        {
            "experiment": "Exp. 9",
            "models":
                r"LR; MLP($h=4$); "
                r"VQC($L=1,2$)",
            "epochs":
                f"LR {LR_HP.epochs}; "
                f"MLP {MLP_HP.epochs}; "
                f"VQC {VQC_HP.epochs}",
            "lr": f"{LR_HP.lr:.1f}",
            "optimizer": optimizer,
            "seeds": standard_seeds,
            "shots":
                r"VQC: analytic, 128, 1024; "
                r"classical: n/a",
        },
    ]

    PAPER_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    with CSV_PATH.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "experiment",
                "models",
                "epochs",
                "lr",
                "optimizer",
                "seeds",
                "shots",
            ],
        )

        writer.writeheader()
        writer.writerows(rows)

    tex_lines = []

    for index, row in enumerate(rows):
        row_end = r" \\" if index < len(rows) - 1 else ""

        tex_lines.append(
            f'{row["experiment"]} '
            f'& {row["models"]} '
            f'& {row["epochs"]} '
            f'& {row["lr"]} '
            f'& {row["optimizer"]} '
            f'& {row["seeds"]} '
            f'& {row["shots"]}{row_end}'
        )

    TEX_PATH.write_text(
        "\n".join(tex_lines) + "\n",
        encoding="utf-8",
    )

    print(
        f"Wrote {CSV_PATH}"
    )

    print(
        f"Wrote {TEX_PATH}"
    )


if __name__ == "__main__":
    main()