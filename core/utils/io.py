from __future__ import annotations

import os
import csv
from typing import Dict, Iterable, List, Any
from pathlib import Path


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def append_csv_row(path: str | Path, fieldnames: List[str], row: Dict[str, Any]) -> None:
    """
    Append a single row to CSV. Creates file and header if missing.

    IMPORTANT:
    - Filters out keys not present in fieldnames (to avoid ValueError)
    - Fills missing keys with empty string
    """
    p = Path(path)
    ensure_dir(str(p.parent) if str(p.parent) else ".")
    file_exists = p.exists()

    # keep only declared columns, fill missing with ""
    filtered = {k: row.get(k, "") for k in fieldnames}

    with open(p, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            w.writeheader()
        w.writerow(filtered)


def csv_contains_row(path: str | Path, key_fields: Dict[str, Any]) -> bool:
    """
    Return True if CSV exists and contains a row matching all key_fields (string compare).
    """
    p = Path(path)
    if not p.exists():
        return False

    with open(p, "r", newline="", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            ok = True
            for k, v in key_fields.items():
                if str(row.get(k)) != str(v):
                    ok = False
                    break
            if ok:
                return True
    return False

def write_csv_rows(path: str, fieldnames: List[str], rows: Iterable[Dict[str, Any]]) -> None:
    """Write list of dict rows to CSV."""
    ensure_dir(os.path.dirname(path) or ".")
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)

def append_csv(path: Path, fieldnames: list[str], row: dict):
    exists = path.exists()
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not exists:
            writer.writeheader()
        writer.writerow(row)

def csv_contains(path: Path, key_fields: dict) -> bool:
    if not path.exists():
        return False

    with path.open() as f:
        reader = csv.DictReader(f)
        for r in reader:
            if all(str(r[k]) == str(v) for k, v in key_fields.items()):
                return True
    return False


def read_csv_rows(path: str) -> List[Dict[str, str]]:
    """Read CSV into list of dict rows (all values as strings)."""
    with open(path, "r", newline="", encoding="utf-8") as f:
        r = csv.DictReader(f)
        return list(r)
