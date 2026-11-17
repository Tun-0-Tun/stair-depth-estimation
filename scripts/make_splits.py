#!/usr/bin/env python3
"""Split ``ours`` by STAIRCASE (not by frame) and save the lists.

    python scripts/make_splits.py --out data/splits/ours_seed0.json --seed 0
    python scripts/make_splits.py --extend data/splits/ours_seed0.json --out data/splits/ours_v2_seed0.json   # more data, old test kept

Needs shots.xlsx (stairs id per shot).  Fractions 0.7 / 0.1 / 0.2 of the staircases, shuffled with
a fixed seed.  Re-running with the same seed and data gives the same file; a new Dataset V1.0
gets a new split file (never edit the old one -- results refer to it).
"""
from __future__ import annotations

import argparse
import json
from collections import Counter

import _bootstrap  # noqa: F401
import numpy as np

from data.degradation import IdentityDegradation
from data.loaders import build_dataset
from utils.misc import REPO_ROOT


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="ours/sample")
    ap.add_argument("--out", default="data/splits/ours_seed0.json")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--frac", type=float, nargs=3, default=(0.7, 0.1, 0.2))
    ap.add_argument("--extend", help="old split json: its staircases keep their split, only new ones are added")
    a = ap.parse_args()
    ds = build_dataset({"name": "ours", "loader": "ours", "root": a.root, "crop": None, "target_size": None},
                       degradation=IdentityDegradation())
    shots = Counter(r["meta"].get("stairs_id") for r in ds.records)
    if None in shots:
        raise SystemExit("shots.xlsx missing or unreadable (needs openpyxl): cannot split by staircase")
    ids = sorted(shots)
    rng = np.random.default_rng(a.seed)
    if a.extend:  # keep every old assignment, deal only the NEW staircases so the totals approach the fractions
        old = json.load(open(REPO_ROOT / a.extend))
        split = {k: [int(i) for i in old[k]] for k in ("test", "val", "train")}
        gone = sorted(set().union(*split.values()) - set(ids))
        if gone:
            raise SystemExit(f"staircases of {a.extend} missing from the data: {gone}")
        new = [int(i) for i in rng.permutation(sorted(set(ids) - set().union(*split.values())))]
        n_val, n_test = round(a.frac[1] * len(ids)), round(a.frac[2] * len(ids))
        k_test, k_val = max(0, n_test - len(split["test"])), max(0, n_val - len(split["val"]))
        split["test"] += new[:k_test]
        split["val"] += new[k_test : k_test + k_val]
        split["train"] += new[k_test + k_val :]
        split = {k: sorted(v) for k, v in split.items()}
    else:
        ids = [int(i) for i in rng.permutation(ids)]
        n_val, n_test = round(a.frac[1] * len(ids)), round(a.frac[2] * len(ids))
        split = {"test": sorted(ids[:n_test]), "val": sorted(ids[n_test : n_test + n_val]),
                 "train": sorted(ids[n_test + n_val :])}
    out = {"seed": a.seed, "root": a.root, "fractions": list(a.frac), "extends": a.extend, **split,
           "n_stairs": {k: len(v) for k, v in split.items()},
           "n_shots": {k: sum(shots[i] for i in v) for k, v in split.items()}}
    p = REPO_ROOT / a.out
    p.write_text(json.dumps(out, indent=1))
    print(json.dumps({k: out[k] for k in ("n_stairs", "n_shots")}), "->", p)


if __name__ == "__main__":
    main()
