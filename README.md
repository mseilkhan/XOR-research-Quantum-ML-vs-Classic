# XOR Classification: Classical vs Quantum Machine Learning

This repository contains the implementation and experimental pipeline accompanying the study comparing classical machine learning models (Logistic Regression, MLP) and a Variational Quantum Classifier (VQC) on several XOR dataset variants.

The repository is structured to allow reproducible execution of all simulator-based experiments reported in the main body of the paper.

IBM Quantum hardware experiments described in the manuscript are not included in this public artifact.

## Authors

**Miras Seilkhan**  
Email: seilkhan.miras6117@gmail.com  

**Adilbek Taizhanov**  
Email: adilbek300108@gmail.com  

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

### 2.3 Smoke / CI checks

The repository includes reduced smoke checks for the core decision-boundary
and benchmark pipelines. These checks validate code paths only and must not be
used as manuscript results.

Run:

```bash
pytest -q
```

The smoke tests execute reduced versions of Experiments 1 and 6 and validate
their outputs using `manifest_smoke.json`

---

## 3. Output Structure

Full experimental runs and smoke/CI runs use separate output roots so that
reduced sanity checks cannot overwrite publication-run artifacts.

A standard run writes to:

```text
outputs/
  figures/
  csv/
  tables/
  logs/
```

A smoke run invoked with `XOR_SMOKE=1` writes instead to:

```text
outputs_smoke/
  figures/
  csv/
  tables/
  logs/
```

Files produced in smoke mode are intended only for code-path validation and
**must not be used as numerical results for the manuscript**.
---
## 4. Reproducibility Scope and Experimental Protocol

The simulator-based experimental protocol is centralized in
`experiments/settings.py`. This file defines the dataset and split seeds,
training horizons, learning rates, optimizer, model-seed sets, architecture
grids, and VQC shot settings used by the experiment scripts.

An experiment-level protocol table can be generated directly from these
settings with:

```bash
python tools/export_protocol.py
```

The command writes the following files:

- `paper/experiment_protocol.csv`
- `paper/experiment_protocol_rows.tex`

The fixed Dataset-B benchmark reported in the manuscript is treated separately
from transient experiment outputs. Its publication-level aggregate values are
stored in `paper/canonical_benchmark.csv`. This file records the frozen
five-seed benchmark used for the manuscript tables and quantitative claims;
smoke-run outputs are never authoritative publication results.

The public repository implements the classical and simulator-based quantum experiments described in the manuscript.

IBM Quantum hardware executions are reported separately in the manuscript and are not included in this public artifact.

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


This repository is tested with Python 3.13. The pinned dependency versions in
`requirements.txt` should be installed in a Python 3.13 environment.
