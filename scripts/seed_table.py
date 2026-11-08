#!/usr/bin/env python3
"""Mean +- std over seeds of test.json metrics, on all test frames and on the strict stair-like ones.

    python scripts/seed_table.py --subset mask_quality.json \
        1=outputs/aprime_e30/$X/1/test.json,outputs/aprime_e30_s1/$X/1/test.json,... M0=a.json,b.json,c.json

Each ``LABEL=`` has the test.json of one run per seed (any order).  Prints per-run values, mean +- std, and the relative
change of every label versus the first one (mean over seeds, and per seed when the number of seeds matches).
"""
from __future__ import annotations

import argparse
import json

import numpy as np

ZONES = [("full", "absrel"), ("full", "rmse"), ("stairs", "absrel"), ("edge", "absrel")]


def frames(path, keep):
    return [p for p in json.load(open(path))["per_frame"] if keep(p["id"])]


def mean(fr, z, m):
    return float(np.nanmean([p[z][m] for p in fr]))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+", help="LABEL=a.json,b.json,...")
    ap.add_argument("--subset", required=True, help="mask_quality.json")
    ap.add_argument("--flag", default="stair_like_strict")
    a = ap.parse_args()
    q = json.load(open(a.subset))
    sets = {"all test frames": lambda i: True, f"{a.flag} frames": lambda i: bool(q.get(i, {}).get(a.flag))}
    runs = {lab: files.split(",") for lab, files in (r.split("=", 1) for r in a.runs)}
    ref = next(iter(runs))
    for sname, keep in sets.items():
        print(f"\n## {sname}  (n={len(frames(runs[ref][0], keep))})")
        vals = {lab: np.array([[mean(frames(f, keep), z, m) for z, m in ZONES] for f in files]) for lab, files in runs.items()}
        print("| row | seeds | " + " | ".join(f"{z} {m}" for z, m in ZONES) + " |\n|---|---|" + "---|" * len(ZONES))
        for lab, v in vals.items():
            print(f"| {lab} | {len(v)} | " + " | ".join(f"{v[:, j].mean():.4f} ± {v[:, j].std(ddof=1) if len(v) > 1 else float('nan'):.4f}" for j in range(len(ZONES))) + " |")
        print("\nrelative to", ref, "(negative = better; mean over seeds / per seed):")
        for lab, v in vals.items():
            if lab == ref:
                continue
            rel = v.mean(0) / vals[ref].mean(0) - 1
            per = (v / vals[ref] - 1) * 100 if len(v) == len(vals[ref]) else None
            print(f"  {lab}: " + "  ".join(f"{z} {m} {100 * rel[j]:+.1f}%" + (f" [{', '.join(f'{x:+.1f}' for x in per[:, j])}]" if per is not None else "") for j, (z, m) in enumerate(ZONES)))


if __name__ == "__main__":
    main()
