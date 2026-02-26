from __future__ import annotations

import os
from dataclasses import asdict
from typing import Any, Dict, Iterable, List, Optional, Sequence

import numpy as np

from core.utils.io import ensure_dir, write_csv_rows
from core.utils.timer import Timer


def ensure_output_dirs(*dirs: str) -> None:
    for d in dirs:
        ensure_dir(d)


def mean_std(xs: Sequence[float]) -> tuple[float, float]:
    arr = np.array(xs, dtype=float)
    return float(arr.mean()), float(arr.std(ddof=0))


def rows_from_records(records: Iterable[Any]) -> List[Dict[str, Any]]:
    """Convert dataclass records (or dict-like) to list[dict]."""
    out: List[Dict[str, Any]] = []
    for r in records:
        if hasattr(r, "__dataclass_fields__"):
            out.append(asdict(r))
        elif isinstance(r, dict):
            out.append(r)
        else:
            raise TypeError(f"Unsupported record type: {type(r)}")
    return out


def save_records_csv(path: str, records: List[Any]) -> None:
    rows = rows_from_records(records)
    if not rows:
        raise ValueError("No rows to write.")
    fieldnames = list(rows[0].keys())
    write_csv_rows(path, fieldnames, rows)

def save_summaries_csv(path: str, summaries: List[Dict[str, Any]], fieldnames: Optional[List[str]] = None) -> None:
    """
    Save list of dict rows into CSV.
    Fix 7 bugfix: if rows have extra keys, build fieldnames as union of keys to avoid csv.DictWriter crash.
    This preserves ALL keys (no silent dropping).
    """
    ensure_dir(os.path.dirname(path))

    if fieldnames is None:
        keys = []
        seen = set()
        # stable order: keep insertion order across rows
        for row in summaries:
            for k in row.keys():
                if k not in seen:
                    seen.add(k)
                    keys.append(k)
        fieldnames = keys

    write_csv_rows(path, fieldnames, summaries)


def tag_shots(shots: Optional[int]) -> str:
    return "analytic" if shots is None else f"shots{int(shots)}"


def tag_sigma(sigma: float) -> str:
    return f"sigma{sigma:.2f}".replace(".", "")


def tag_n(n: int) -> str:
    return f"n{int(n)}"
