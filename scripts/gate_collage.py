#!/usr/bin/env python3
"""Collage of what the stair terms will SEE in training: StairNet masks on the frames the gate lets through (and a few it blocks).

    STAIRNET_CACHE_DIR=~/stairnet_cache python scripts/gate_collage.py [--config configs/aprime/m2_mix_gated.yaml] [--n 24] [--off 8]

Uses the training code itself (``APrimeDataset.masks``: cache -> clean_masks -> confidence -> gate), so the picture is the mask
the loss gets.  Red = tread, green = riser, yellow = edge band (drawn from the raw cleaned masks, even where training uses only the edge weight); the caption is  id | tread heights | tread_inlier | riser_inlier.
One sheet per source in ``outputs/gate_collage/``: ``<source>_gated.png`` (stair terms ON) and ``<source>_blocked.png`` (OFF).
"""
from __future__ import annotations

import argparse
from pathlib import Path

import _bootstrap  # noqa: F401
import numpy as np
from PIL import Image, ImageDraw

from data.aprime_dataset import APrimeDataset, build_extra, load_split
from data.degradation import IdentityDegradation
from models.aprime_runner import load_cfg
from utils.misc import REPO_ROOT

COLORS = {"tread": (230, 40, 40), "riser": (40, 200, 40), "band": (250, 220, 0)}


def tile(s, m, label, size=(344, 264)) -> Image.Image:
    img = (np.asarray(s["rgb"]) * 255).astype(np.uint8)
    ov = img.astype(np.float32)
    for k, w in (("tread", 0.5), ("riser", 0.5), ("band", 0.8)):
        ov[m[k]] = (1 - w) * ov[m[k]] + w * np.asarray(COLORS[k], np.float32)
    im = Image.fromarray(np.clip(ov, 0, 255).astype(np.uint8)).resize(size)
    ImageDraw.Draw(im).rectangle((0, 0, size[0], 12), fill=(0, 0, 0))
    ImageDraw.Draw(im).text((3, 0), label, fill=(255, 255, 0))
    return im


def sheet(tiles, cols, path) -> None:
    if not tiles:
        print("  (nothing to draw)", path.name)
        return
    w, h = tiles[0].size
    rows = -(-len(tiles) // cols)
    out = Image.new("RGB", (cols * w, rows * h))
    for i, t in enumerate(tiles):
        out.paste(t, ((i % cols) * w, (i // cols) * h))
    out.save(path)
    print("->", path, f"({len(tiles)} tiles)")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("overrides", nargs="*")
    ap.add_argument("--config", default="configs/aprime/m2_mix_gated.yaml")
    ap.add_argument("--n", type=int, default=24, help="gated frames per source")
    ap.add_argument("--off", type=int, default=8, help="blocked frames per source")
    ap.add_argument("--cols", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    cfg = load_cfg(a.config, ["degradation.preset=sequence_bicubic_x8_fill_lateral_projector_axial", *a.overrides])
    sources = {"ours": (APrimeDataset(cfg, "train", train=True), cfg["data"].get("stair_gate_file"))}
    for src in cfg["data"].get("extra") or []:
        if src.get("stair_terms", "none") != "none":
            sources[src["dataset"]] = (build_extra(cfg, src), src.get("flags"))
    out = REPO_ROOT / "outputs" / "gate_collage"
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(a.seed)
    for name, (ds, stats_file) in sources.items():
        ds.base.degradation = IdentityDegradation()  # only RGB / depth shape are needed, skip the sensor chain
        stats = load_split(stats_file) if stats_file else {}
        ids = [ds.base.record_id(r) for r in ds.base.records]
        on = [i for i, sid in enumerate(ids) if ds.gate is None or ds.gate.get(sid, False)]
        off = [i for i in range(len(ids)) if i not in set(on)]
        print(f"{name}: {len(on)} of {len(ids)} frames pass the gate")
        for tag, pool, k in (("gated", on, a.n), ("blocked", off, a.off)):
            pick = sorted(rng.choice(pool, min(k, len(pool)), replace=False)) if pool else []
            tiles, share = [], []
            for i in pick:
                s = ds.base[i]
                # blocked frames: show what StairNet WOULD have said, so ask the cache directly (gate off for this call)
                gate, terms, ds.gate, ds.stair_terms = ds.gate, ds.stair_terms, None, "all"  # draw the raw cleaned masks
                m = ds.masks(s)
                ds.gate, ds.stair_terms = gate, terms
                share.append([float(m[k].mean()) for k in ("tread", "riser", "band")])
                st = stats.get(ids[i], {})
                lab = f"{ids[i]} | lv {st.get('n_levels', '?')} ti {st.get('tread_inlier', float('nan')):.2f} ri {st.get('riser_inlier', float('nan')):.2f}"
                tiles.append(tile(s, m, lab))
            if share:
                t_, r_, b_ = np.mean(share, 0) * 100
                print(f"  {tag}: mean share of frame  tread {t_:.1f}%  riser {r_:.1f}%  band {b_:.1f}%")
            sheet(tiles, a.cols, out / f"{name}_{tag}.png")


if __name__ == "__main__":
    main()
