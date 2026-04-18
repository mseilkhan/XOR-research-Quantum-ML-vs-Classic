# tests/test_smoke.py
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def run(cmd: list[str], env=None):
    r = subprocess.run(cmd, cwd=str(ROOT), env=env, capture_output=True, text=True)
    assert r.returncode == 0, f"cmd failed: {cmd}\nstdout:\n{r.stdout}\nstderr:\n{r.stderr}"


def test_imports():
    run([sys.executable, "-c", "import core; import experiments"])


def test_smoke_exp01():
    env = dict(os.environ)
    env["XOR_SMOKE"] = "1"
    run([sys.executable, "-m", "experiments.exp_01_decision_boundaries"], env=env)


def test_smoke_exp06():
    env = dict(os.environ)
    env["XOR_SMOKE"] = "1"
    run([sys.executable, "-m", "experiments.exp_06_summary_tables"], env=env)


def test_validate_artifacts():
    env = dict(os.environ)
    env["XOR_SMOKE"] = "1"
    # Валидация запускается последней, когда оба эксперимента уже сгенерировали файлы
    run([sys.executable, "tools/validate_artifacts.py", "--manifest", "manifest_smoke.json"], env=env)