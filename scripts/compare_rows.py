#!/usr/bin/env python3
"""Paired comparison of ablation rows against a reference row, bootstrapped over STAIRCASES.

    python scripts/compare_rows.py --ref 1 --rows 2 2b 2c 3a 3b 4 5 [--dir outputs/aprime/strong] [--n-boot 5000]

Needs ``test.json`` with ``per_frame`` (written by the current eval_aprime.py; re-run it on the existing
best.pth, see docs/aprime.md).  For every zone/metric: mean relative change vs the reference in % (negative =
better for absrel/rmse), with a 95 % bootstrap CI over staircases (frames of one staircase are correlated, so the
unit is the staircase, not the frame).  "sig" marks CIs that exclude 0.
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import _bootstrap  # noqa: F401
import numpy as np

from utils.misc import REPO_ROOT

ZM = [("full", "absrel"), ("full", "rmse"), ("stairs", "absrel"), ("stairs", "rmse"), ("edge", "absrel"), ("edge", "rmse")]


def load(d: Path, rid: str):
    r = json.loads((d / rid / "test.json").read_text())
    if "per_frame" not in r:
        raise SystemExit(f"{d / rid / 'test.json'} has no per_frame: re-run eval_aprime.py on best.pth first")
    return {f["id"]: f for f in r["per_frame"]}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", default="1")
    ap.add_argument("--rows", nargs="+", required=True)
    ap.add_argument("--dir", default="outputs/aprime/strong")
    ap.add_argument("--n-boot", type=int, default=5000)
    a = ap.parse_args()
    d = REPO_ROOT / a.dir
    ref = load(d, a.ref)
    rng = np.random.default_rng(0)
    head = ["row"] + [f"{z} {m}" for z, m in ZM]
    print("| " + " | ".join(head) + " |\n|" + "---|" * len(head))
    for rid in a.rows:
        cur = load(d, rid)
        ids = [i for i in ref if i in cur]
        line = [rid]
        for z, m in ZM:
            by = defaultdict(list)  # staircase -> (ref, cur) per-frame pairs (frames present in both zones)
            for i in ids:
                if z in ref[i] and z in cur[i]:
                    by[ref[i]["stairs_id"]].append((ref[i][z][m], cur[i][z][m]))
            st = list(by)
            if not st:
                line.append("n/a")
                continue
            r_ = np.array([np.mean([p[0] for p in by[s]]) for s in st])
            c_ = np.array([np.mean([p[1] for p in by[s]]) for s in st])
            est = 100 * (c_.mean() / r_.mean() - 1)
            boots = []
            for _ in range(a.n_boot):
                k = rng.integers(0, len(st), len(st))
                boots.append(100 * (c_[k].mean() / r_[k].mean() - 1))
            lo, hi = np.percentile(boots, [2.5, 97.5])
            line.append(f"{est:+.1f}% [{lo:+.1f}, {hi:+.1f}]{' sig' if lo > 0 or hi < 0 else ''}")
        print("| " + " | ".join(line) + " |")
    print(f"\nreference = row {a.ref}; {len(set(f['stairs_id'] for f in ref.values()))} staircases; negative = better (absrel/rmse).")


if __name__ == "__main__":
    main()
