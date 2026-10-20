#!/usr/bin/env python3
"""Defect statistics of the REAL Astra 2 depth in ``ours`` + what each degradation severity does.

    python scripts/astra_defect_stats.py [--limit 200] [--severities light medium strong] [--out outputs/astra_defect_stats.json]

Real Astra (the GT maps): share of invalid pixels (distribution over frames); where they sit
(within 5 px of a depth edge, on bright/flat RGB regions, per distance bin of the nearest valid
depth); hole-component sizes.  Then, per severity, the invalid share of the degraded HR input
and its ratio to the real one (target for ``strong``: 2-3x).  Tune
``data/degradation/astra_hard.py:SEVERITY`` / configs until the ratios fit, and record them.
"""
from __future__ import annotations

import argparse
import json

import _bootstrap  # noqa: F401
import numpy as np
import yaml
from scipy import ndimage

from data.degradation import AstraHardDegradation, IdentityDegradation
from data.degradation.active_stereo_astra2 import _depth_edges
from data.loaders import build_dataset
from utils.misc import REPO_ROOT, sparse_to_dense_nn


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--severities", nargs="*", default=["light", "medium", "strong"])
    ap.add_argument("--out", default="outputs/astra_defect_stats.json")
    a = ap.parse_args()
    cfg = yaml.safe_load((REPO_ROOT / "configs/dataset/ours.yaml").read_text())
    ds = build_dataset({k: cfg[k] for k in ("name", "loader", "root", "crop", "target_size", "min_depth", "max_depth")},
                       degradation=IdentityDegradation())
    n = min(len(ds), a.limit or len(ds))
    idx = np.linspace(0, len(ds) - 1, n).astype(int)
    inv, near_edge, area_near_edge, bright, flat, comp, bins = [], [], [], [], [], [], np.zeros((4, 2))
    edges = [0.6, 2.0, 4.0, 8.0, 1e9]
    for i in idx:
        s = ds[i]
        gt, m = s["gt_depth"], s["mask"]
        hole = ~m
        inv.append(hole.mean())
        if hole.any() and m.any():
            d = sparse_to_dense_nn(np.where(m, gt, 0))
            band = ndimage.binary_dilation(_depth_edges(d, d > 0, 0.05), iterations=5)
            near_edge.append((hole & band).sum() / hole.sum()); area_near_edge.append(band.mean())
            g = s["rgb"].mean(-1)
            std = np.sqrt(np.maximum(ndimage.uniform_filter(g**2, 7) - ndimage.uniform_filter(g, 7) ** 2, 0))
            bright.append((hole & (g > 0.92)).sum() / hole.sum()); flat.append((hole & (std < 0.012)).sum() / hole.sum())
            lab, k = ndimage.label(hole)
            if k:
                comp += list(ndimage.sum(hole, lab, range(1, k + 1)))
            for b in range(4):
                sel = (d >= edges[b]) & (d < edges[b + 1])
                bins[b] += [(hole & sel).sum(), sel.sum()]
    q = lambda x: {p: float(np.percentile(x, p)) for p in (10, 50, 90)}  # noqa: E731
    real = {"n_frames": int(n), "invalid_frac_mean": float(np.mean(inv)), "invalid_frac_pcts": q(inv),
            "holes_within_5px_of_depth_edge": float(np.mean(near_edge)), "(area share of that band)": float(np.mean(area_near_edge)),
            "holes_on_bright_rgb": float(np.mean(bright)), "holes_on_flat_rgb": float(np.mean(flat)),
            "hole_area_px_pcts": q(comp) if comp else None,
            "hole_rate_by_nearest_depth_bin_m": {f"{edges[b]}-{edges[b + 1]}": float(bins[b, 0] / max(bins[b, 1], 1)) for b in range(4)}}
    sev = {}
    for name in a.severities:
        deg = AstraHardDegradation(severity=name, scale=8)
        fr = []
        for i in idx[:: max(1, len(idx) // 40)]:
            s = ds[i]
            o = deg(s["gt_depth"], seed=int(i), context={"rgb": s["rgb"], "sample_id": s["meta"]["sample_id"]})
            fr.append(o["meta"]["input_invalid_frac"])
        sev[name] = {"input_invalid_frac": float(np.mean(fr)), "ratio_to_real": float(np.mean(fr) / max(real["invalid_frac_mean"], 1e-9))}
    out = {"real_astra": real, "severities": sev}
    print(json.dumps(out, indent=1))
    p = REPO_ROOT / a.out
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
