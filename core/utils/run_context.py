# core/utils/run_context.py
from __future__ import annotations

import hashlib
import json
import platform
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional


@dataclass(frozen=True)
class RunContext:
    exp_name: str
    run_id: str
    metadata_path: Path


def _short_hash(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()[:8]


def _pkg_version(name: str) -> Optional[str]:
    try:
        mod = __import__(name)
        return getattr(mod, "__version__", None)
    except Exception:
        return None


def create_run_context(
    *,
    exp_name: str,
    settings: Dict[str, Any],
    metadata_dir: Path = Path("outputs/metadata"),
) -> RunContext:
    metadata_dir.mkdir(parents=True, exist_ok=True)

    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    seed = f"{exp_name}|{ts}|{' '.join(sys.argv)}"
    run_id = f"{ts}_{_short_hash(seed)}"

    meta = {
        "exp_name": exp_name,
        "run_id": run_id,
        "argv": list(sys.argv),
        "python": sys.version,
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "version": platform.version(),
            "machine": platform.machine(),
            "processor": platform.processor(),
        },
        "packages": {
            "numpy": _pkg_version("numpy"),
            "matplotlib": _pkg_version("matplotlib"),
            "pennylane": _pkg_version("pennylane"),
        },
        "settings": settings,
        "timestamp_utc": ts,
    }

    path = metadata_dir / f"{exp_name}_{run_id}.json"
    path.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")

    return RunContext(exp_name=exp_name, run_id=run_id, metadata_path=path)