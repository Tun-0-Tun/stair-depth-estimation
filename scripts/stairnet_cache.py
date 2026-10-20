#!/usr/bin/env python3
"""Offline StairNet-DepthOut pass over the RGB frames of ``ours`` -> masks cached next to the data.

    python scripts/stairnet_cache.py [--weights weights/stairnet/depth_out/best.pth] [--device cuda]

Runs on the loader's frame (crop + target_size from configs/dataset/ours.yaml), so the cached maps
are exactly 528x688-aligned with the depth; StairNet's own padding/resize is undone inside
``StairNetSeg.predict``.  Depth is never used.  Output: ``$STAIR_DATA_ROOT/<out>/<sample_id>.npz``.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import _bootstrap  # noqa: F401
import yaml
from tqdm import tqdm

from data.degradation import IdentityDegradation
from data.loaders import build_dataset
from data.stairnet.cache import cache_dir, save_masks
from data.stairnet.infer import StairNetSeg
from utils.misc import REPO_ROOT, data_root, resolve_device


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default="weights/stairnet/depth_out/best.pth")
    ap.add_argument("--width-factor", type=float, default=1.0)
    ap.add_argument("--out", default="ours/stairnet_cache", help="relative to STAIR_DATA_ROOT; $STAIRNET_CACHE_DIR overrides")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--tread-class", type=int, default=1)
    ap.add_argument("--riser-class", type=int, default=2)
    a = ap.parse_args()
    cfg = yaml.safe_load((REPO_ROOT / "configs/dataset/ours.yaml").read_text())
    ds = build_dataset({k: cfg[k] for k in ("name", "loader", "root", "crop", "target_size")},
                       degradation=IdentityDegradation())
    dev = resolve_device(a.device)
    net = StairNetSeg(a.weights, a.width_factor, dev, a.tread_class, a.riser_class)
    out = cache_dir(a.out)
    out.mkdir(parents=True, exist_ok=True)  # fail now, not after the first forward pass
    for i in tqdm(range(len(ds))):
        s = ds[i]
        r = net.predict(s["rgb"])
        save_masks(out, s["meta"]["sample_id"], r["cls"], r["conf"], r["lines"])
    (out / "meta.json").write_text(json.dumps({
        "weights": a.weights, "width_factor": a.width_factor, "crop": cfg["crop"],
        "target_size": cfg["target_size"], "class_map_net_to_canonical": {str(k): v for k, v in net.class_map.items()},
        "n": len(ds), "note": "canonical ids: 0 background, 1 tread, 2 riser. tread/riser mapping must be checked with stairnet_viz.py"}, indent=1))
    print("wrote", len(ds), "frames to", out)


if __name__ == "__main__":
    main()
