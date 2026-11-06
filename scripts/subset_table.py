#!/usr/bin/env python3
"""Absolute metrics of every ablation row (and the baselines) on a subset of test frames.

    python scripts/subset_table.py --dir $P --subset $P/mask_quality.json --flag stair_like

The subset comes from mask_quality.json (frames chosen by the GT geometry, never by a model's output).  Metrics are
the per-frame values stored in test.json (``per_frame``), averaged over the frames of the subset (image pooling, NaN
frames of an empty zone skipped) -- the same convention as ablation.md.  Prints markdown; also the number of frames.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import _bootstrap  # noqa: F401
import numpy as np

from utils.misc import REPO_ROOT

ROWS = ["0", "DuCos", "DEPTHOR", "1", "2", "2b", "2c", "3a", "3b", "4", "5", "6"]
COLS = [("full", "absrel"), ("full", "rmse"), ("full", "delta1"), ("stairs", "absrel"), ("edge", "absrel")]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--subset", required=True)
    ap.add_argument("--flag", default="stair_like")
    ap.add_argument("--rows", nargs="*", default=ROWS)
    ap.add_argument("--invert", action="store_true", help="the complement (frames WITHOUT the flag)")
    a = ap.parse_args()
    d = REPO_ROOT / a.dir if not Path(a.dir).is_absolute() else Path(a.dir)
    q = json.loads(Path(a.subset).read_text())
    keep = {i for i, r in q.items() if bool(r.get(a.flag)) != a.invert}
    head = ["row", "frames"] + [f"{z} {m}" for z, m in COLS]
    print(f"frames {'without' if a.invert else 'with'} {a.flag}: {len(keep)} of {len(q)}\n")
    print("| " + " | ".join(head) + " |\n|" + "---|" * len(head))
    for rid in a.rows:
        f = d / rid / "test.json"
        if not f.exists():
            continue
        pf = [x for x in json.loads(f.read_text()).get("per_frame", []) if x["id"] in keep]
        cells = []
        for z, m in COLS:
            v = np.array([x[z][m] for x in pf if z in x and x[z].get(m) is not None], float)
            v = v[np.isfinite(v)]
            cells.append(f"{v.mean():.4f}" if v.size else "n/a")
        print(f"| {rid} | {len(pf)} | " + " | ".join(cells) + " |")


if __name__ == "__main__":
    main()
