#!/usr/bin/env python3
"""Run the ablation table (configs/aprime/ablations.yaml) and collect csv + markdown.

    python scripts/run_ablations.py --severity strong [--only 1 2 3a] [--dry-run] [--force]

Train rows: fine-tune -> evaluate best.pth on val and test.  Row "5" combines the best fusion
(rows 2, 2b, 2c) and the best edge weight (3a, 3b) by VAL full-frame AbsRel -- never by test.
Existing result files are reused unless --force.  Nothing is trained with --dry-run.
"""
from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path

import _bootstrap  # noqa: F401
import yaml

from models.aprime_runner import intrinsics, load_cfg
from utils.misc import REPO_ROOT

PY = sys.executable
COLS = [("full", "absrel"), ("full", "rmse"), ("full", "delta1"), ("stairs", "absrel"), ("stairs", "rmse"),
        ("stairs", "delta1"), ("edge", "absrel"), ("edge", "rmse"), ("edge", "delta1")]


def sh(cmd: list[str], dry: bool) -> None:
    print("$", " ".join(cmd))
    if not dry:
        subprocess.run(cmd, check=True, cwd=REPO_ROOT)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--severity", default="strong")
    ap.add_argument("--only", nargs="*")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--out", default="outputs/aprime")
    ap.add_argument("overrides", nargs="*", help="applied to every row, e.g. train.epochs=1")
    a = ap.parse_args()
    rows = yaml.safe_load((REPO_ROOT / "configs/aprime/ablations.yaml").read_text())["rows"]
    root = REPO_ROOT / a.out / a.severity
    ov = [f"degradation.severity={a.severity}", *a.overrides]
    res: dict[str, dict] = {}

    def result(row, split="test"):
        return root / row["id"] / f"{split}.json"

    def val_absrel(i):
        f = root / i / "val.json"
        return json.loads(f.read_text())["full"]["absrel"] if f.exists() else float("inf")

    for row in rows:
        if a.only and row["id"] not in a.only:
            continue
        cfg_path = f"configs/aprime/{row['config']}"
        extra = list(ov)
        if row.get("requires") == "intrinsics" and intrinsics(load_cfg(cfg_path, ov)) is None:
            print(f"[skip] row {row['id']}: needs intrinsics K (data.intrinsics), not set")
            continue
        if row.get("combine"):
            fus = min(("2", "2b", "2c"), key=val_absrel)
            edg = min(("3a", "3b"), key=val_absrel)
            fk = {"2": "stairnetv2_orig", "2b": "channel", "2c": "spatial"}[fus]
            ek = {"3a": "binary", "3b": "gaussian"}[edg]
            if val_absrel(fus) == float("inf") or val_absrel(edg) == float("inf"):
                print(f"[skip] row 5: rows 2*/3* have no val results yet")
                continue
            extra += [f"model.fusion={fk}", f"loss.edge_weight={ek}"]
            row = {**row, "name": f"all: fusion={fk}, edge_weight={ek}"}
        out_dir = root / row["id"]
        ckpt = out_dir / "best.pth"
        if row["kind"] == "train" and (a.force or not ckpt.exists()):
            sh([PY, "scripts/train_aprime.py", cfg_path, "--out", str(out_dir), *extra], a.dry_run)
        for split in ("val", "test"):
            f = result(row, split)
            if a.force or not f.exists():
                cmd = [PY, "scripts/eval_aprime.py", cfg_path, "--split", split, "--out", str(f), *extra]
                if row["kind"] == "train":
                    cmd += ["--ckpt", str(ckpt)]
                sh(cmd, a.dry_run)
        if result(row).exists():
            res[row["id"]] = {"name": row["name"], **json.loads(result(row).read_text())}

    if not res:
        return
    root.mkdir(parents=True, exist_ok=True)
    geo = sorted({k for r in res.values() for k in r if k.startswith("step_")})
    head = ["id", "name", "input_invalid_frac"] + [f"{z}_{m}" for z, m in COLS] + geo
    flat = [[i, r["name"], r["input_invalid_frac"]] + [r[z][m] for z, m in COLS] + [r.get(g, "") for g in geo]
            for i, r in res.items()]
    with (root / "ablation.csv").open("w", newline="") as f:
        csv.writer(f).writerows([head, *flat])
    fmt = lambda v: f"{v:.4f}" if isinstance(v, float) else str(v)  # noqa: E731
    md = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    md += ["| " + " | ".join(fmt(v) for v in r) + " |" for r in flat]
    note = ("\n\nGT = the Astra 2 sensor's own depth (no independent reference): numbers rank methods, "
            "they are not accuracy.  delta1 uses the 1.25 threshold.  Split by staircase; "
            f"severity={a.severity}; input_invalid_frac = mean share of invalid pixels in the HR input.\n")
    (root / "ablation.md").write_text("\n".join(md) + note)
    print("\n".join(md))


if __name__ == "__main__":
    main()
