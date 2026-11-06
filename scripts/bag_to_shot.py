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

from utils import rosbag_min as rb
from utils.astra_bag import align_depth, decode_color, decode_depth, load_bag, nearest


def decode_disparity_param(bag) -> dict | None:
    info = rb.read_bag(bag)
    m = info["first"].get("/cam/disparityParam")
    return rb.parse_ob_disparity(m) if m else None


def _png(path, a) -> None:
    Image.fromarray(a).save(path)


def _ref(sample: Path, shot: int, suffix: str) -> Path:
    """``s<id>_<suffix>.png`` of shot ``shot``, whatever the zero padding of the id is."""
    hit = [p for p in sample.glob(f"s*_{suffix}.png") if int(p.name[1:].split("_")[0]) == shot]
    if not hit:
        raise SystemExit(f"no s*_{suffix}.png for shot {shot} in {sample}")
    return hit[0]


def validate(a) -> None:
    sample = Path(a.sample or Path(os.environ["STAIR_DATA_ROOT"]) / "ours" / "sample")
    sid = f"s{a.shot}"
    raw_ref = np.asarray(Image.open(_ref(sample, a.shot, "depth_raw")))
    rgb_ref = np.asarray(Image.open(_ref(sample, a.shot, "rgb")).convert("RGB")).astype(np.int16)
    al_ref = np.asarray(Image.open(_ref(sample, a.shot, "depth_aligned_to_rgb")))
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


def probe(a) -> None:
    """How do the bag's raw depth values relate to the sample's s<id>_depth_raw?  (scale? offset? inverse = disparity?)"""
    sample = Path(a.sample or Path(os.environ["STAIR_DATA_ROOT"]) / "ours" / "sample")
    ref = np.asarray(Image.open(_ref(sample, a.shot, "depth_raw"))).astype(np.float64)
    rgb_ref = np.asarray(Image.open(_ref(sample, a.shot, "rgb")).convert("RGB")).astype(np.int16)
    prof, fr = load_bag(a.bag)
    iou = []
    for _, img in fr["depth"]:
        d = decode_depth(img)
        iou.append(float(((d > 0) & (ref > 0)).sum() / max(((d > 0) | (ref > 0)).sum(), 1)))
    di = int(np.argmax(iou))
    ci = int(np.argmin([np.abs(decode_color(img).astype(np.int16) - rgb_ref).mean() for _, img in fr["color"]]))
    print(f"valid-mask IoU per depth frame (max {max(iou):.3f} at {di}): " + " ".join(f"{x:.2f}" for x in iou))
    print(f"colour frame {ci}; depth frame nearest in time to it: {nearest(fr['depth'], fr['color'][ci][0])}")
    for k in sorted({di, nearest(fr["depth"], fr["color"][ci][0])}):
        v = decode_depth(fr["depth"][k][1]).astype(np.float64)
        both = (v > 0) & (ref > 0)
        x, y = v[both], ref[both]
        print(f"-- depth frame {k}: both valid {both.sum()} px | bag raw min/med/max {x.min():.0f}/{np.median(x):.0f}/{x.max():.0f} "
              f"| sample raw {y.min():.0f}/{np.median(y):.0f}/{y.max():.0f}")
        r = y / x
        print(f"   ratio sample/bag: median {np.median(r):.4f}, IQR {np.percentile(r, 25):.4f}..{np.percentile(r, 75):.4f}")
        print(f"   product sample*bag: median {np.median(x * y):.0f}, rel. IQR {(np.percentile(x * y, 75) - np.percentile(x * y, 25)) / np.median(x * y):.3f}")
        ok = (x > 1) & (x < 16383)  # 1 and 2^14-1 look like markers, not measurements
        for lo, hi in ((0.5, 99.5),):
            print(f"   bag raw percentiles 1/5/25/50/75/95/99: {np.percentile(x, [1, 5, 25, 50, 75, 95, 99]).round(0)}; "
                  f"share ==16383: {(x == 16383).mean():.3f}, ==1: {(x == 1).mean():.3f}")
        xi, yi = x[ok], y[ok]
        A = np.stack([xi, np.ones_like(xi)], 1)
        c, *_ = np.linalg.lstsq(A, 1.0 / yi, rcond=None)  # structured light: 1/Z linear in disparity
        z = 1.0 / (A @ c)
        err = z - yi
        print(f"   fit 1/sample = a*bag + b: a={c[0]:.6g} b={c[1]:.6g}  -> depth rms {np.sqrt((err ** 2).mean()):.1f} mm, "
              f"median |err| {np.median(np.abs(err)):.1f} mm, within 5 mm {(np.abs(err) <= 5).mean():.3f} (on {ok.sum()} px)")
        dp = decode_disparity_param(a.bag)
        if dp:
            print(f"   camera disparity param: {dp};  1/(fx*baseline) = {1 / (dp['fx'] * dp['baseline']):.6g}, 1/zpd = {1 / dp['zpd']:.6g}")
        for name, xx in (("bag", x), ("1/bag", 1.0 / x)):
            A = np.stack([xx, np.ones_like(xx)], 1)
            c, *_ = np.linalg.lstsq(A, y, rcond=None)
            res = y - A @ c
            print(f"   fit sample = a*{name} + b: a={c[0]:.5g} b={c[1]:.4g}  R2={1 - res.var() / y.var():.4f}  rms {np.sqrt((res ** 2).mean()):.1f} mm")


def look(a) -> None:
    """Is the bag's depth frame the same picture as s<id>_depth_raw?  Correlation under flips + a side-by-side PNG."""
    sample = Path(a.sample or Path(os.environ["STAIR_DATA_ROOT"]) / "ours" / "sample")
    ref = np.asarray(Image.open(_ref(sample, a.shot, "depth_raw"))).astype(np.float64)
    prof, fr = load_bag(a.bag)
    k = len(fr["depth"]) // 2
    img = fr["depth"][k][1]
    v = decode_depth(img).astype(np.float64)
    print(f"bag depth frame {k}: {img['width']}x{img['height']} encoding={img['encoding']!r} step={img['step']} "
          f"bytes={len(img['data'])} big_endian={img['big_endian']}; sample raw {ref.shape[::-1]}")
    sub = (slice(None, None, 8), slice(None, None, 8))
    for name, t in (("identity", v), ("flip lr", v[:, ::-1]), ("flip ud", v[::-1]), ("rot180", v[::-1, ::-1])):
        x, y = t[sub].ravel(), ref[sub].ravel()
        ok = (x > 0) & (y > 0)
        print(f"  {name:9} Pearson corr(bag, sample) = {np.corrcoef(x[ok], y[ok])[0, 1]:+.3f}   corr(1/bag, sample) = "
              f"{np.corrcoef(1.0 / x[ok], y[ok])[0, 1]:+.3f}")
    def norm(a_):
        lo, hi = np.percentile(a_[a_ > 0], [1, 99])
        return (np.clip((a_ - lo) / (hi - lo), 0, 1) * 255).astype(np.uint8)
    rgb = decode_color(fr["color"][k][1])
    pan = [np.asarray(Image.fromarray(norm(x)).resize((640, 480))) for x in (v, ref)]
    pan = [np.stack([p_] * 3, -1) for p_ in pan] + [np.asarray(Image.fromarray(rgb).resize((640, 480)))]
    out = Path(os.path.expanduser(a.out))
    Image.fromarray(np.concatenate(pan, 1)).save(out)
    print("->", out, "(bag raw depth | sample depth_raw | colour)")


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
    pr = sub.add_parser("probe")
    pr.add_argument("--bag", required=True)
    pr.add_argument("--shot", type=int, required=True)
    pr.add_argument("--sample", default=None)
    lk = sub.add_parser("look")
    lk.add_argument("--bag", required=True)
    lk.add_argument("--shot", type=int, required=True)
    lk.add_argument("--sample", default=None)
    lk.add_argument("--out", default="~/look.png")
    c = sub.add_parser("convert")
    c.add_argument("--bag", required=True)
    c.add_argument("--out", required=True)
    c.add_argument("--shot", type=int, required=True)
    c.add_argument("--frame", default="mid", help="depth frame: mid | last | index")
    c.add_argument("--direction", default="d2c", choices=("d2c", "c2d"))
    c.add_argument("--dist", action="store_true")
    c.add_argument("--z-mode", default="color", choices=("color", "depth"))
    a = ap.parse_args()
    {"validate": validate, "probe": probe, "look": look, "convert": convert}[a.cmd](a)


if __name__ == "__main__":
    main()
