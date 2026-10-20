#!/usr/bin/env python3
"""Split ``ours`` by STAIRCASE (not by frame) and save the lists.

    python scripts/make_splits.py --out data/splits/ours_seed0.json --seed 0

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
    a = ap.parse_args()
    ds = build_dataset({"name": "ours", "loader": "ours", "root": a.root, "crop": None, "target_size": None},
                       degradation=IdentityDegradation())
    shots = Counter(r["meta"].get("stairs_id") for r in ds.records)
    if None in shots:
        raise SystemExit("shots.xlsx missing or unreadable (needs openpyxl): cannot split by staircase")
    ids = sorted(shots)
    rng = np.random.default_rng(a.seed)
    ids = [int(i) for i in rng.permutation(ids)]
    n_val, n_test = round(a.frac[1] * len(ids)), round(a.frac[2] * len(ids))
    split = {"test": sorted(ids[:n_test]), "val": sorted(ids[n_test : n_test + n_val]),
             "train": sorted(ids[n_test + n_val :])}
    out = {"seed": a.seed, "root": a.root, "fractions": list(a.frac), **split,
           "n_stairs": {k: len(v) for k, v in split.items()},
           "n_shots": {k: sum(shots[i] for i in v) for k, v in split.items()}}
    p = REPO_ROOT / a.out
    p.write_text(json.dumps(out, indent=1))
    print(json.dumps({k: out[k] for k in ("n_stairs", "n_shots")}), "->", p)


if __name__ == "__main__":
    main()
