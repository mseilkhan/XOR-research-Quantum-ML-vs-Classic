# XOR Classification: Classical vs Quantum Machine Learning

This repository contains the implementation and experimental pipeline accompanying the study comparing classical machine learning models (Logistic Regression, MLP) and a Variational Quantum Classifier (VQC) on several XOR dataset variants.

The repository is structured to allow reproducible execution of all simulator-based experiments reported in the main body of the paper.

IBM Quantum hardware experiments described in the manuscript are not included in this public artifact.

---

## 1. Environment Setup

### 1.1 Create a virtual environment

**macOS / Linux**

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

**Windows (PowerShell)**

```bash
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

### 1.2 Install dependencies

```bash
pip install -r requirements.txt
```

---

## 2. Running Experiments

All commands must be executed from the repository root directory.

### 2.1 Run all experiments

```bash
python -m experiments.run_all
```

### 2.2 Run a single experiment

```bash
python -m experiments.exp_01_decision_boundaries
python -m experiments.exp_02_learning_behavior
python -m experiments.exp_03_robustness_datasetB
python -m experiments.exp_04_vqc_shots_dependence
python -m experiments.exp_05_seed_sensitivity
python -m experiments.exp_06_summary_tables
python -m experiments.exp_07_mlp_width_ablation
python -m experiments.exp_08_loss_landscape_slices
python -m experiments.exp_09_datasetC_study
```

### 2.3 Optional smoke mode

If smoke mode is supported via environment variable:

macOS / Linux:

```bash
XOR_SMOKE=1 python -m experiments.run_all
```

Windows PowerShell:

```bash
$env:XOR_SMOKE="1"
python -m experiments.run_all
```

---

## 3. Output Structure

All experiment outputs are written to the `outputs/` directory (created automatically if absent):

```
outputs/
  figures/     # Generated figures (PNG)
  csv/         # Raw and aggregated results
  logs/        # Execution logs
  metadata/    # Run metadata
```

If directories are missing, they can be created manually:

```bash
mkdir -p outputs/figures outputs/csv outputs/logs outputs/metadata
```

---

## 4. Reproducibility Scope

This public repository reproduces all classical and simulator-based quantum experiments presented in the main body of the paper.

Hardware executions performed on IBM Quantum devices are reported in the manuscript for completeness but are not included in this artifact.

---

## 5. Troubleshooting

### Missing dependency

Reinstall dependencies:

```bash
pip install -r requirements.txt
```

### Headless environments (no display)

If running on a server without graphical backend:

```bash
export MPLBACKEND=Agg
python -m experiments.run_all
```

