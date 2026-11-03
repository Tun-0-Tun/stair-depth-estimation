#!/usr/bin/env python3
"""What does Hypersim say about the stairs in our selected frames?  (look before building masks)

    python scripts/hypersim_stair_probe.py [--split data/splits/hypersim/stairs.json] [--n 8] [--ids ai_001_003/cam_00/0012 ...]

Per frame writes ``outputs/hypersim_probe/<scene>_<cam>_<frame>.png``: RGB | semantic | instance (numbers = instance
ids) | normal_world: red = near-horizontal (|nz| > 0.9, tread candidates), green = near-vertical (|nz| < 0.25, riser
candidates).  Also prints, once, whether world Z is "up" (floor normals should point +1, ceiling -1).
Needs normal_world / semantic / semantic_instance next to depth_meters (see docs/datasets.md).
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import _bootstrap  # noqa: F401
import h5py
import numpy as np
from PIL import Image, ImageDraw

from utils.misc import REPO_ROOT

FLOOR, CEILING = 2, 22  # NYU40


def read(path) -> np.ndarray:
    with h5py.File(path) as f:
        return f["dataset"][:]


def colours(lbl: np.ndarray) -> np.ndarray:
    rng = np.random.default_rng(0)
    pal = rng.integers(40, 255, (int(lbl.max()) + 2, 3), dtype=np.uint8)
    out = pal[np.clip(lbl, 0, None)]
    out[lbl < 0] = 0
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="data/splits/hypersim/stairs.json")
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--ids", nargs="*")
    a = ap.parse_args()
    root = Path(os.environ["STAIR_DATA_ROOT"]) / "hypersim"
    d = json.load(open(REPO_ROOT / a.split))
    ids = a.ids or (d if isinstance(d, list) else next(v for v in d.values() if isinstance(v, list)))[:: max(1, 150 // a.n)][: a.n]
    out = REPO_ROOT / "outputs" / "hypersim_probe"
    out.mkdir(parents=True, exist_ok=True)
    fl, ce = [], []
    for i in ids:
        scene, cam, fr = i.split("/")
        g = root / scene / "images" / f"scene_{cam}_geometry_hdf5"
        nz = read(g / f"frame.{fr}.normal_world.hdf5").astype(np.float32)[..., 2]
        sem = read(g / f"frame.{fr}.semantic.hdf5").astype(np.int32)
        ins = read(g / f"frame.{fr}.semantic_instance.hdf5").astype(np.int32)
        fl.append(nz[sem == FLOOR]); ce.append(nz[sem == CEILING])
        rgb = np.asarray(Image.open(root / scene / "images" / f"scene_{cam}_final_preview" / f"frame.{fr}.color.jpg"))
        nrm = np.zeros(nz.shape + (3,), np.uint8)
        nrm[np.abs(nz) > 0.9] = (230, 40, 40)
        nrm[np.abs(nz) < 0.25] = (40, 200, 40)
        nrm[np.isnan(nz)] = 0
        panels = [Image.fromarray(x).resize((512, 384)) for x in (rgb, colours(sem), colours(ins), nrm)]
        draw = ImageDraw.Draw(panels[2])
        for k in np.unique(ins[ins >= 0]):
            m = ins == k
            if m.mean() < 0.01:
                continue
            ys, xs = np.nonzero(m)
            draw.text((xs.mean() / 2, ys.mean() / 2), str(k), fill=(255, 255, 255))
        sheet = Image.new("RGB", (2048, 384))
        for j, p in enumerate(panels):
            sheet.paste(p, (512 * j, 0))
        sheet.save(out / f"{scene}_{cam}_{fr}.png")
        print("->", out / f"{scene}_{cam}_{fr}.png")
    fl, ce = np.concatenate(fl), np.concatenate(ce)
    print(f"median nz: floor {np.median(fl):.3f} (want +1), ceiling {np.median(ce) if ce.size else float('nan'):.3f} (want -1)")


if __name__ == "__main__":
    main()
