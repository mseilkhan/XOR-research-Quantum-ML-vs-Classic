# core/utils/logging.py
from __future__ import annotations

import logging
from pathlib import Path


def setup_logger(exp_name: str, run_id: str, log_dir: Path) -> logging.Logger:
    """
    Logger that writes to:
      - stdout
      - outputs/logs/<exp_name>_<run_id>.log

    Each log line includes run_id for traceability.
    """
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / f"{exp_name}_{run_id}.log"

    logger = logging.getLogger(f"{exp_name}:{run_id}")
    logger.setLevel(logging.INFO)
    logger.handlers = []

    fmt = logging.Formatter(
        f"[%(asctime)s][%(levelname)s][run_id={run_id}] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    fh = logging.FileHandler(log_file, encoding="utf-8")
    fh.setFormatter(fmt)

    sh = logging.StreamHandler()
    sh.setFormatter(fmt)

    logger.addHandler(fh)
    logger.addHandler(sh)
    logger.propagate = False

    return logger