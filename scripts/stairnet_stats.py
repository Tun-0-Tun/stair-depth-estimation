#!/usr/bin/env python3
"""How much of each frame is "stairs" according to the cached StairNet masks (a proxy -- there is no ground truth).

    python scripts/stairnet_stats.py [--cache ours/stairnet_cache]

Prints the distribution of the stair share of the frame (tread | riser pixels) before and after
``clean_masks``, how many frames each ``min_stair_frac`` threshold would call "no stairs", and the
frames to LOOK at (smallest shares, and those near the threshold) so the threshold can be set by eye.
"""
from __future__ import annotations

import argparse
from collections import defaultdict

import _bootstrap  # noqa: F401
import numpy as np

from data.stairnet.cache import cache_dir, load_masks
from data.stairnet.edges import DEFAULT_CLEAN, clean_masks
from data.degradation import IdentityDegradation
from data.loaders import build_dataset
from utils.misc import REPO_ROOT
import yaml


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="ours/stairnet_cache")
    ap.add_argument("--thresholds", type=float, nargs="*", default=[0.02, 0.05, 0.08, 0.12, 0.2])
    a = ap.parse_args()
    cfg = yaml.safe_load((REPO_ROOT / "configs/dataset/ours.yaml").read_text())
    ds = build_dataset({k: cfg[k] for k in ("name", "loader", "root", "crop", "target_size")}, degradation=IdentityDegradation())
    d = cache_dir(a.cache)
    ids, raw, clean_nofloor, by_stairs = [], [], [], defaultdict(list)
    for r in ds.records:
        sid = f"s{r['num']}"
        m = load_masks(d, sid)
        if m is None:
            continue
        ids.append(sid)
        raw.append(float((m["cls"] > 0).mean()))
        # cleaned share WITHOUT the frame-level drop, so the threshold can be studied
        clean_nofloor.append(float((clean_masks(m["cls"], {"min_stair_frac": 0.0}) > 0).mean()))
        by_stairs[r["meta"].get("stairs_id")].append(clean_nofloor[-1])
    raw, cl = np.array(raw), np.array(clean_nofloor)
    print(f"{len(ids)} cached frames; defaults: {DEFAULT_CLEAN}")
    for name, x in (("raw share", raw), ("share after clean (no frame drop)", cl)):
        print(f"{name}: " + "  ".join(f"p{p}={np.percentile(x, p) * 100:.1f}%" for p in (5, 10, 25, 50, 75, 90)))
    h, e = np.histogram(cl, bins=[0, .01, .02, .05, .08, .12, .2, .3, .5, 1.0])
    print("histogram of the cleaned share:")
    for c, lo, hi in zip(h, e[:-1], e[1:]):
        print(f"  {lo * 100:5.0f}-{hi * 100:3.0f}% {'#' * int(60 * c / max(h.max(), 1))} {c}")
    print("frames that would be called 'no stairs':")
    for t in a.thresholds:
        print(f"  min_stair_frac={t:<5} -> {(cl < t).sum():4d} frames ({100 * (cl < t).mean():.1f}%)")
    order = np.argsort(cl)
    print("\n20 smallest shares (open these in outputs/stairnet_viz, or run viz on them):")
    print("  " + ", ".join(f"{ids[i]}({cl[i] * 100:.1f}%)" for i in order[:20]))
    near = order[np.argsort(np.abs(cl[order] - 0.05))[:20]]
    print("20 frames closest to 5%:")
    print("  " + ", ".join(f"{ids[i]}({cl[i] * 100:.1f}%)" for i in near))
    print("\nstaircases with the lowest median share (a whole staircase that is 'small' is suspicious):")
    for sid, v in sorted(by_stairs.items(), key=lambda kv: np.median(kv[1]))[:8]:
        print(f"  stairs {sid}: median {np.median(v) * 100:.1f}% over {len(v)} shots")


if __name__ == "__main__":
    main()
