#!/usr/bin/env python3
"""Pick two test frames for a figure from the per-frame metrics already in test.json (no model is run).

    python scripts/pick_frames.py --dir outputs/aprime_e30/$X --gate mask_quality.json

Only frames with ``stair_like_strict`` (a real staircase by the GT geometry).  Rows: --ours (ours), --wave (fine-tuned
WAVE base), --baselines (fine-tuned DuCos / DEPTHOR); per-frame full-frame AbsRel from ``per_frame`` of each ``test.json``.
  typical : the frame whose mean AbsRel over all methods is closest to the median over the gated frames
  failure : the largest gap  max(baselines) - ours  (a baseline clearly errs, ours does not)
Prints the best 5 of each (so you can swap one), and the make_viz.py command for the first of each.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def per_frame(p: Path) -> dict[str, float]:
    return {f["id"]: f["full"]["absrel"] for f in json.loads(p.read_text())["per_frame"]}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True, help="outputs/aprime_e30/$X (one folder per row with test.json)")
    ap.add_argument("--gate", default="mask_quality.json")
    ap.add_argument("--ours", default="M0")
    ap.add_argument("--wave", default="1")
    ap.add_argument("--baselines", nargs="+", default=["DuCos_ft", "DEPTHOR_ft"])
    ap.add_argument("--top", type=int, default=5, help="how many frames to list per kind")
    ap.add_argument("--both", action="store_true", help="failure = EVERY baseline is worse than ours (gap = min over baselines), "
                    "not just the worst one")
    ap.add_argument("--ours-max", type=float, help="failure: keep only frames where ours AbsRel <= this")
    ap.add_argument("--bdir", help="folder of the --baselines rows when it differs from --dir (e.g. the zero-shot DuCos / DEPTHOR)")
    ap.add_argument("--no-gate", action="store_true", help="all test frames, not only strict stair-like ones")
    ap.add_argument("--ratio", action="store_true", help="rank by baseline/ours AbsRel instead of the difference")
    a = ap.parse_args()
    rows = [a.ours, a.wave, *a.baselines]
    where = {r: Path(a.bdir or a.dir) if r in a.baselines else Path(a.dir) for r in rows}
    missing = [f"{where[r]}/{r}" for r in rows if not (where[r] / r / "test.json").exists()]
    if missing:
        raise SystemExit(f"no test.json for {missing}")
    pf = {r: per_frame(where[r] / r / "test.json") for r in rows}
    ids = set.intersection(*(set(v) for v in pf.values()))
    if not a.no_gate:
        ids &= {k for k, v in json.loads(Path(a.gate).read_text()).items() if v["stair_like_strict"]}
    ids = sorted(ids)
    print(f"{len(ids)} frames ({'all' if a.no_gate else 'strict stair-like'}) present in all of {rows}")
    mean = {i: np.mean([pf[r][i] for r in rows]) for i in ids}
    med = np.median(list(mean.values()))
    typical = sorted(ids, key=lambda i: abs(mean[i] - med))[: a.top]
    agg = min if a.both else max
    gap = {i: agg(pf[r][i] for r in a.baselines) - pf[a.ours][i] for i in ids}
    if a.ratio:
        gap = {i: agg(pf[r][i] for r in a.baselines) / pf[a.ours][i] for i in ids}
    cand = [i for i in ids if a.ours_max is None or pf[a.ours][i] <= a.ours_max]
    fail = sorted(cand, key=lambda i: -gap[i])[: a.top]
    for name, lst in (("typical", typical), ("failure", fail)):
        print(f"\n{name}:")
        for i in lst:
            print(f"  {i:>6}  gap={gap[i]:+.4f}  " + "  ".join(f"{r}={pf[r][i]:.4f}" for r in rows))
    first_fail = next(i for i in fail if i != typical[0])  # two different frames
    print(f"\n--ids {typical[0]} {first_fail}")
    print("\nall failure ids:", " ".join(fail))


if __name__ == "__main__":
    main()
