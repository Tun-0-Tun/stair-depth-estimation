#!/usr/bin/env python3
"""Evaluate A' (a checkpoint) or a baseline adapter on one split, same data/degradation/metrics.

    python scripts/eval_aprime.py CONFIG [--ckpt best.pth] [--split test] [--out result.json] [k.sub=v ...]

Zones: full frame, stair mask (tread|riser), edge band.  delta1 is the standard 1.25 threshold.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import _bootstrap  # noqa: F401
import torch

from data.aprime_dataset import APrimeDataset
from models.aprime_runner import (baseline_predictor, build_aprime, evaluate, load_cfg, predict_aprime)
from utils.misc import resolve_device, set_seed


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("overrides", nargs="*")
    ap.add_argument("--ckpt")
    ap.add_argument("--split", default="test")
    ap.add_argument("--out")
    ap.add_argument("--limit", type=int)
    a = ap.parse_args()
    cfg = load_cfg(a.config, a.overrides)
    set_seed(int(cfg.get("seed", 0)))
    device = resolve_device(cfg.get("device", "auto"))
    ds = APrimeDataset(cfg, a.split, train=False, with_raw=True)  # test corruption is deterministic
    if cfg.get("method") == "baseline":
        predict, _ = baseline_predictor(cfg, device)
    else:
        if a.ckpt:
            cfg["model"]["resume"] = a.ckpt
        net = build_aprime(cfg, device).eval()
        predict = lambda it: predict_aprime(net, it, device)  # noqa: E731
    res = evaluate(cfg, predict, ds, a.limit)
    res.update(config=cfg["name"], split=a.split, severity=cfg["degradation"]["severity"], ckpt=a.ckpt)
    print(json.dumps(res, indent=1))
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
