# Artifact Manifest and Validation

This repository provides the classical and simulator-based experimental
protocol used in the revised manuscript.

## Scope

`manifest.json` validates the core simulator artifacts produced by experiments
`exp_01` through `exp_06`. Artifact identifiers are experiment-oriented because
the figure numbering changed during revision.

The repository additionally contains:

- `exp_07_mlp_width_ablation.py` — expanded 20-seed MLP width study reported in Appendix A.1;
- `exp_08_loss_landscape_slices.py` — loss-landscape analysis reported in Appendix A.2;
- `exp_09_datasetC_study.py` — Dataset C threshold and size studies.

The IBM Quantum hardware evaluation is documented in the manuscript but is not
included in this public reproduction artifact.

## How to reproduce

Run the simulator experiment suite from the repository root:

```bash
python -m experiments.run_all
```

For a reduced code-path check:

```bash
pytest -q
```

Smoke-mode outputs are diagnostic only and must not be used as manuscript
results.