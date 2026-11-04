#!/usr/bin/env python3
"""Raw Astra 2 ``.bag`` -> the PNGs of ``ours/sample``, and a check that we reproduce the shots we already have.

  validate  -- take a bag of an EXISTING shot and compare with its PNGs in ``ours/sample``:
       uv run --with lz4 python scripts/bag_to_shot.py validate --bag "<raw/other/Orbbec ...bag>" --shot 1
       * raw depth:    which depth frame of the bag is the one in s<id>_depth_raw.png, and is it bit-identical
       * colour:       which colour frame matches s<id>_rgb.png (and how far the YUYV->RGB conversion is off)
       * alignment:    every variant (extrinsics direction x distortion x z) against s<id>_depth_aligned_to_rgb.png
  convert   -- write s<id>_rgb / _depth_aligned_to_rgb / _mask_aligned_valid / _depth_raw / _mask_raw_valid:
       uv run --with lz4 python scripts/bag_to_shot.py convert --bag X.bag --out DIR --shot 903 [--frame mid] \
           [--direction d2c --dist --z-mode color]

``convert`` is only to be used with the variant and the frame rule that ``validate`` confirmed on old shots.
"""
from __future__ import annotations

import argparse
import itertools
import os
from pathlib import Path

import _bootstrap  # noqa: F401
import numpy as np
from PIL import Image

from utils.astra_bag import align_depth, decode_color, decode_depth, load_bag, nearest


def _png(path, a) -> None:
    Image.fromarray(a).save(path)


def validate(a) -> None:
    sample = Path(a.sample or Path(os.environ["STAIR_DATA_ROOT"]) / "ours" / "sample")
    sid = f"s{a.shot}"
    raw_ref = np.asarray(Image.open(sample / f"{sid}_depth_raw.png"))
    rgb_ref = np.asarray(Image.open(sample / f"{sid}_rgb.png").convert("RGB")).astype(np.int16)
    al_ref = np.asarray(Image.open(sample / f"{sid}_depth_aligned_to_rgb.png"))
    prof, fr = load_bag(a.bag)
    print(f"bag: {len(fr['color'])} colour / {len(fr['depth'])} depth frames; "
          f"colour {prof['color']['width']}x{prof['color']['height']}, depth {prof['depth']['width']}x{prof['depth']['height']}")

    eq = [float((decode_depth(img) == raw_ref).mean()) for _, img in fr["depth"]]
    di = int(np.argmax(eq))
    print(f"[raw depth]  best frame {di} of {len(eq)}: {eq[di] * 100:.2f}% pixels identical to s{a.shot}_depth_raw "
          f"(others: max {max(e for k, e in enumerate(eq) if k != di) * 100:.2f}%)")
    diff = [float(np.abs(decode_color(img).astype(np.int16) - rgb_ref).mean()) for _, img in fr["color"]]
    ci = int(np.argmin(diff))
    nn = nearest(fr["color"], fr["depth"][di][0])
    print(f"[colour]     best frame {ci}: mean |dRGB| = {diff[ci]:.2f} levels (next best {sorted(diff)[1]:.2f}); "
          f"nearest in time to the depth frame is {nn} ({'same' if nn == ci else 'DIFFERENT'})")
    print(f"[frame rule] depth frame {di}/{len(eq)} (middle = {len(eq) // 2}), colour frame {ci}/{len(diff)}")

    depth = decode_depth(fr["depth"][di][1])
    print(f"[alignment] vs s{a.shot}_depth_aligned_to_rgb  (ref valid {(al_ref > 0).mean() * 100:.1f}%)")
    print("  direction  dist  z      | valid IoU | within 1/2/5 mm of ref (both valid) | median |d| mm")
    for dr, dist, zm in itertools.product(("d2c", "c2d"), (False, True), ("color", "depth")):
        al = align_depth(depth, prof["depth"], prof["color"], dr, dist, zm)
        both, either = (al > 0) & (al_ref > 0), (al > 0) | (al_ref > 0)
        d = np.abs(al[both].astype(np.int32) - al_ref[both].astype(np.int32))
        print(f"  {dr}      {str(dist):5} {zm:6} | {both.sum() / max(either.sum(), 1):9.3f} | "
              f"{(d <= 1).mean():.3f} / {(d <= 2).mean():.3f} / {(d <= 5).mean():.3f}"
              f"                   | {np.median(d) if d.size else float('nan'):.1f}")


def convert(a) -> None:
    prof, fr = load_bag(a.bag)
    n = len(fr["depth"])
    di = n // 2 if a.frame == "mid" else n - 1 if a.frame == "last" else int(a.frame)
    ci = nearest(fr["color"], fr["depth"][di][0])
    depth = decode_depth(fr["depth"][di][1])
    al = align_depth(depth, prof["depth"], prof["color"], a.direction, a.dist, a.z_mode)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    sid = f"s{a.shot}"
    _png(out / f"{sid}_rgb.png", decode_color(fr["color"][ci][1]))
    _png(out / f"{sid}_depth_raw.png", depth)
    _png(out / f"{sid}_depth_aligned_to_rgb.png", al)
    _png(out / f"{sid}_mask_raw_valid.png", ((depth > 0) * 255).astype(np.uint8))
    _png(out / f"{sid}_mask_aligned_valid.png", ((al > 0) * 255).astype(np.uint8))
    print(f"{sid}: depth frame {di}/{n}, colour frame {ci}, aligned valid {(al > 0).mean() * 100:.1f}% -> {out}")


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    v = sub.add_parser("validate")
    v.add_argument("--bag", required=True)
    v.add_argument("--shot", type=int, required=True)
    v.add_argument("--sample", default=None, help="folder with the sample PNGs (default $STAIR_DATA_ROOT/ours/sample)")
    c = sub.add_parser("convert")
    c.add_argument("--bag", required=True)
    c.add_argument("--out", required=True)
    c.add_argument("--shot", type=int, required=True)
    c.add_argument("--frame", default="mid", help="depth frame: mid | last | index")
    c.add_argument("--direction", default="d2c", choices=("d2c", "c2d"))
    c.add_argument("--dist", action="store_true")
    c.add_argument("--z-mode", default="color", choices=("color", "depth"))
    a = ap.parse_args()
    validate(a) if a.cmd == "validate" else convert(a)


if __name__ == "__main__":
    main()
