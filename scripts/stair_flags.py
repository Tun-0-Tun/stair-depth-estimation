#!/usr/bin/env python3
"""Which frames of ANOTHER dataset (VOID, ...) have a trustworthy staircase?  Same test as ``mask_quality.py``.

    python scripts/stair_flags.py --dataset void_stairs [--stride 5] [--out stair_flags_void.json]

StairNet masks (same cleaning as training) -> geometry check against the dense GT depth with the dataset's own
intrinsics ``meta["intrinsics"]`` (so the dataset needs K: VOID has it, MinJiang does not).  Writes one json,
``{sample_id: {..., stair_like, stair_like_strict}}``; the stair loss is later switched on only for strict frames.
Native resolution, nothing is resized.  Depth of the frame is used only to JUDGE the mask, never as input.
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict

import _bootstrap  # noqa: F401
import numpy as np
from PIL import Image, ImageDraw
from tqdm import tqdm

from data.degradation import IdentityDegradation
from data.loaders import build_dataset
from data.stairnet.cache import cache_dir, save_masks
from data.stairnet.edges import build_edge_maps, clean_masks
from data.stairnet.infer import StairNetSeg
from mask_quality import frame_quality
from models.aprime_runner import load_cfg
from utils.config import load_config
from utils.misc import REPO_ROOT, resolve_device


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="void_stairs")
    ap.add_argument("--stride", type=int, default=5, help="every N-th frame (neighbours are near-duplicates)")
    ap.add_argument("--weights", default="weights/stairnet/depth_out/best.pth")
    ap.add_argument("--config", default="configs/aprime/a1_validity.yaml", help="only for mask cleaning / thresholds")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--out", default=None)
    ap.add_argument("--target-size", type=int, nargs=2, default=None, metavar=("H", "W"),
                    help="resize frames like training does (e.g. 528 688); K is scaled from --native-size")
    ap.add_argument("--native-size", type=int, nargs=2, default=(480, 640), metavar=("H", "W"))
    ap.add_argument("--cache", default=None,
                    help="also write the raw StairNet masks (relative to $STAIR_DATA_ROOT, $STAIRNET_CACHE_DIR wins), e.g. void_stairs/stairnet_cache")
    ap.add_argument("--viz", type=int, default=0, help="save an overlay of every N-th processed frame (0 = off)")
    a = ap.parse_args()
    cfg = load_cfg(a.config, [])
    dcfg = dict(load_config("dataset", a.dataset))
    if "frame_stride" in dcfg:
        dcfg["frame_stride"] = a.stride
    if a.target_size:
        dcfg["target_size"] = list(a.target_size)
    ds = build_dataset(dcfg, degradation=IdentityDegradation())
    net = StairNetSeg(a.weights, 1.0, resolve_device(a.device), 2, 1)
    thr, band = float(cfg["data"].get("conf_thr", 0.7)), float(cfg["loss"].get("edge_band_px", 6))
    res, seq = {}, defaultdict(list)
    for i in tqdm(range(len(ds))):
        s = ds[i]
        k = s["meta"].get("intrinsics")
        if k is None:
            raise SystemExit(f"{a.dataset} gives no intrinsics (meta['intrinsics']): the geometry check needs K")
        if a.target_size:
            (h0, w0), (h1, w1) = a.native_size, a.target_size
            k = (k[0] * w1 / w0, k[1] * h1 / h0, k[2] * w1 / w0, k[3] * h1 / h0)
        r = net.predict(s["rgb"])
        if a.cache:
            save_masks(cache_dir(a.cache), s["meta"]["sample_id"], r["cls"], r["conf"], r.get("lines"))
        cls = clean_masks(r["cls"], cfg["data"].get("mask_clean"))
        m = build_edge_maps(cls, r["conf"], "binary", band, thr, bool(cfg["loss"].get("stair_bg_boundary", True)))
        ok = r["conf"] >= thr
        m["tread"], m["riser"] = (cls == 1) & ok, (cls == 2) & ok
        gt = np.asarray(s["gt_depth"], np.float32)
        q = frame_quality(gt, gt > 0, m, tuple(k))
        res[s["meta"]["sample_id"]] = q
        if a.viz and i % a.viz == 0:
            img = (np.asarray(s["rgb"]) * 255).astype(np.uint8)
            ov = img.copy()
            ov[m["tread"]] = (230, 40, 40)
            ov[m["riser"]] = (40, 200, 40)
            im = Image.fromarray((0.5 * img + 0.5 * ov).astype(np.uint8))
            ImageDraw.Draw(im).text((6, 6), f"{s['meta']['sample_id']} strict={q['stair_like_strict']} lv={q['n_levels']} "
                                    f"ti={q['tread_inlier']:.2f} ri={q['riser_inlier']:.2f}", fill=(255, 255, 0))
            vd = REPO_ROOT / "outputs" / f"stair_flags_{a.dataset}"
            vd.mkdir(parents=True, exist_ok=True)
            im.save(vd / (s["meta"]["sample_id"].replace("/", "_") + ".png"))
        seq[s["meta"].get("sequence", "-")].append(q)
    n = len(res)
    print(f"{n} frames of {a.dataset} (stride {a.stride})")
    for name in ("stair_like", "stair_like_strict"):
        print(f"  {name}: {sum(q[name] for q in res.values())} of {n}")
    for sq, qs in sorted(seq.items()):
        print(f"  {sq}: strict {sum(q['stair_like_strict'] for q in qs)} of {len(qs)}, "
              f"median stair share {np.median([q['stair_share'] for q in qs]):.2f}")
    out = a.out or f"stair_flags_{a.dataset}.json"
    (REPO_ROOT / out).write_text(json.dumps(res, indent=1))
    print("->", out)


if __name__ == "__main__":
    main()
