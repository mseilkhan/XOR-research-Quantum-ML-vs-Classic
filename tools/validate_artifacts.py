from __future__ import annotations

import json
import sys
from pathlib import Path
import glob
import argparse


ROOT = Path(__file__).resolve().parents[1]


def glob_exists(pattern: str) -> bool:
    matches = glob.glob(str(ROOT / pattern), recursive=True)
    return len(matches) > 0

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="manifest.json", help="Path to manifest JSON (relative to repo root)")
    args = ap.parse_args()

    manifest_path = ROOT / args.manifest
    if not manifest_path.exists():
        print(f"[FAIL] manifest not found at: {manifest_path}")
        return 2

    text = manifest_path.read_text(encoding="utf-8").strip()

    data = json.loads(manifest_path.read_text(encoding="utf-8"))
    artifacts = data.get("artifacts", [])

    missing_total = 0
    print("[INFO] Validating artifacts listed in manifest.json ...\n")

    for a in artifacts:
        aid = a.get("id", "<no-id>")
        pref = a.get("paper_ref", "")
        script = a.get("script", "")
        exp = a.get("expected_outputs", [])
        print(f"== {aid} ({pref}) ==")
        print(f"script: {script}")

        missing = []
        for p in exp:
            if not glob_exists(p):
                missing.append(p)

        if missing:
            missing_total += len(missing)
            print("[FAIL] Missing expected outputs:")
            for m in missing:
                print(f"  - {m}")
        else:
            print("[OK] All expected outputs present.")

        print()

    if missing_total > 0:
        print(f"[SUMMARY] FAIL — missing {missing_total} expected outputs.")
        return 1

    print("[SUMMARY] OK — all artifacts present.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())