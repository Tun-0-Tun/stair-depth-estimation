#!/usr/bin/env python3
"""Forgetting check: a checkpoint (or a baseline) on NYUv2 under OUR sensor chain, full-frame metrics only (no stairs).

    python scripts/eval_nyu.py CONFIG [--ckpt best.pth] [--limit N] [--out r.json] [--stride 5] [k.sub=v ...]
    python scripts/eval_nyu.py configs/aprime/r0_wave_zeroshot.yaml --out nyu_wave0.json     # reference

NYUv2 (``nyu_depth_v2_labeled.mat``, 1449 frames, test only) goes through the same ``data.extra`` machinery as the training
mixes: 480x640 -> 528x688, the preset of the config's ``degradation.preset``, normals from the GT.  Pass the same preset
for every model you compare.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import _bootstrap  # noqa: F401

from data.aprime_dataset import build_extra
from models.aprime_runner import baseline_predictor, build_aprime, evaluate, load_cfg, predict_aprime
from utils.misc import resolve_device, set_seed

# NYUv2 Kinect K at 640x480 scaled to the 688x528 working frame
NYU_K = [518.86 * 688 / 640, 519.47 * 528 / 480, 325.58 * 688 / 640, 253.74 * 528 / 480]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("overrides", nargs="*")
    ap.add_argument("--ckpt")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--stride", type=int, default=1, help="every N-th frame")
    ap.add_argument("--out")
    a = ap.parse_args()
    cfg = load_cfg(a.config, a.overrides)
    set_seed(int(cfg.get("seed", 0)))
    device = resolve_device(cfg.get("device", "auto"))
    src = {"dataset": "nyuv2", "target_size": [528, 688], "native_size": [480, 640], "normals_K": NYU_K,
           "stair_terms": "none", "loader_cfg": {"split": "test", "force_normals": True}}
    ds = build_extra(cfg, src, train=False)
    ds.with_raw = True
    if a.stride > 1:
        ds.base.records = ds.base.records[:: a.stride]
    if cfg.get("method") == "baseline":
        predict, _ = baseline_predictor(cfg, device)
    else:
        if a.ckpt:
            cfg["model"]["resume"] = a.ckpt
        net = build_aprime(cfg, device).eval()
        predict = lambda it: predict_aprime(net, it, device)  # noqa: E731
    res = evaluate(cfg, predict, ds, a.limit)
    res.update(config=cfg["name"], dataset="nyuv2", preset=cfg["degradation"].get("preset"), ckpt=a.ckpt)
    print(json.dumps({k: res[k] for k in ("full", "n_frames", "seconds", "preset", "config", "ckpt")}, indent=1))
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
