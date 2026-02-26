from __future__ import annotations

"""
Run all experiments in paper order.
Usage:
  python -m experiments.run_all
"""

import importlib


ORDER = [
    "experiments.exp_01_decision_boundaries",
    "experiments.exp_02_learning_behavior",
    "experiments.exp_03_robustness_datasetB",
    "experiments.exp_04_vqc_shots_dependence",
    "experiments.exp_05_seed_sensitivity",
    "experiments.exp_06_summary_tables",
    "experiments.exp_07_mlp_width_ablation",
    "experiments.exp_08_loss_landscape_slices",
    "experiments.exp_09_datasetC_study"
]


def main():
    for i, mod in enumerate(ORDER, start=1):
        print(f"[{i}/{len(ORDER)}] {mod}")
        m = importlib.import_module(mod)
        m.main()
    print("All experiments completed.")


if __name__ == "__main__":
    main()
