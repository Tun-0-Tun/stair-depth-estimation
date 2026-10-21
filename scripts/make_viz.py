#!/usr/bin/env python3
"""Qualitative figure: WAVE / DuCos / DEPTHOR (and optionally our model) on test staircases, same degradation as training.

    python scripts/make_viz.py [--n 3] [--ids s12 s203 s640] [--methods wave ducos depthor] [--ours outputs/aprime/strong/3b/best.pth] [--out outputs/viz]

One PNG per column (rgb, input, gt, <methods...>), rows = frames, all cells 688x528 with a fixed per-row depth scale
(turbo, near = blue), black = no measurement.  ``viz_meta.json`` has the chosen frames, the per-row depth scale and
the per-frame full-frame AbsRel of every method (the numbers to print under the images).
Frames: unless --ids is given, the test staircases with the largest stair share of the frame, one frame each
(the one closest to that staircase's median share) -- deterministic, no cherry-picking by error.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import _bootstrap  # noqa: F401
import cv2
import numpy as np
from tqdm import tqdm

from data.aprime_dataset import APrimeDataset
from metrics.depth_metrics import MetricConfig, compute_depth_metrics
from models.aprime_runner import baseline_predictor, build_aprime, load_cfg, predict_aprime
from utils.misc import REPO_ROOT, resolve_device, set_seed

CFG = {"wave": "r0_wave_zeroshot.yaml", "ducos": "r_ducos.yaml", "depthor": "r_depthor.yaml"}
BG = (241, 245, 246)  # BGR of #F6F5F1


def colorize(d, lo, hi, valid=None):
    x = np.clip((np.nan_to_num(d) - lo) / max(hi - lo, 1e-6), 0, 1)
    img = cv2.applyColorMap((x * 255).astype(np.uint8), cv2.COLORMAP_TURBO)
    bad = ~(np.nan_to_num(d) > 0) if valid is None else ~valid
    img[bad] = (0, 0, 0)
    return img


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=3)
    ap.add_argument("--ids", nargs="*")
    ap.add_argument("--methods", nargs="*", default=["wave", "ducos", "depthor"])
    ap.add_argument("--ours", help="best.pth of an A' row; adds a column 'ours'")
    ap.add_argument("--ours-config", default="configs/aprime/a3b_edge_gaussian.yaml")
    ap.add_argument("--severity", default="strong")
    ap.add_argument("--out", default="outputs/viz")
    ap.add_argument("--gap", type=int, default=12)
    ap.add_argument("--set", nargs="*", default=[], help="config overrides key.sub=value applied to every config")
    a = ap.parse_args()
    set_seed(0)
    dev = resolve_device("auto")
    base = load_cfg("configs/aprime/a1_validity.yaml", [f"degradation.severity={a.severity}", *a.set])
    ds = APrimeDataset(base, "test", train=False, with_raw=True)

    if a.ids:
        ids = {r["id"]: i for i, r in enumerate(ds.base.records)}
        picks = [ids[s] for s in a.ids]
    else:  # largest stair share of the frame, one frame per staircase
        share = {}
        for i in tqdm(range(len(ds)), desc="stair share"):
            m = ds[i]["masks"]
            share.setdefault(ds.base.records[i]["meta"].get("stairs_id"), []).append((i, float((m["tread"] | m["riser"]).mean())))
        best = sorted(share.items(), key=lambda kv: -np.mean([s for _, s in kv[1]]))[: a.n]
        picks = [min(v, key=lambda t: abs(t[1] - np.median([s for _, s in v])))[0] for _, v in best]
    items = [ds[i] for i in picks]
    print("frames:", [it["raw"]["meta"]["sample_id"] for it in items])

    preds: dict[str, list[np.ndarray]] = {}
    for m in a.methods:
        cfg = load_cfg(f"configs/aprime/{CFG[m]}", [f"degradation.severity={a.severity}", *a.set])
        try:
            predict, _ = baseline_predictor(cfg, dev)
        except Exception as e:  # noqa: BLE001
            print(f"[skip] {m}: {type(e).__name__}: {e}")
            continue
        preds[m] = [np.asarray(predict(it), np.float32) for it in tqdm(items, desc=m)]
    if a.ours:
        cfg = load_cfg(a.ours_config, [f"degradation.severity={a.severity}", f"model.resume={a.ours}", *a.set])
        net = build_aprime(cfg, dev).eval()
        preds["ours"] = [predict_aprime(net, it, dev) for it in tqdm(items, desc="ours")]

    mc = MetricConfig(min_depth=float(base["data"]["min_depth"]), max_depth=float(base["data"]["max_depth"]))
    cols: dict[str, list[np.ndarray]] = {k: [] for k in ["rgb", "input", "gt", *preds]}
    meta = {"frames": [], "scale_m": [], "absrel": {k: [] for k in preds}, "severity": a.severity}
    for r, it in enumerate(items):
        raw = it["raw"]
        gt, valid = raw["gt_depth"], raw["mask"]
        lo, hi = np.percentile(gt[valid], [2, 98])
        meta["frames"].append({"id": raw["meta"]["sample_id"], "stairs_id": raw["meta"].get("stairs_id"),
                               "input_invalid_frac": float((raw["sparse_depth"] <= 0).mean())})
        meta["scale_m"].append([float(lo), float(hi)])
        cols["rgb"].append((raw["rgb"][..., ::-1] * 255).astype(np.uint8))
        cols["input"].append(colorize(raw["sparse_depth"], lo, hi))
        cols["gt"].append(colorize(gt, lo, hi, valid))
        for k in preds:
            p = preds[k][r]
            cols[k].append(colorize(np.clip(p, mc.min_depth, mc.max_depth), lo, hi, np.ones_like(valid)))
            meta["absrel"][k].append(float(compute_depth_metrics(p, gt, valid, mc)["absrel"]))
    out = REPO_ROOT / a.out
    out.mkdir(parents=True, exist_ok=True)
    for k, rows in cols.items():
        h, w = rows[0].shape[:2]
        canvas = np.full((len(rows) * h + (len(rows) - 1) * a.gap, w, 3), BG, np.uint8)
        for r, img in enumerate(rows):
            canvas[r * (h + a.gap) : r * (h + a.gap) + h] = img
        cv2.imwrite(str(out / f"{k}.png"), canvas)
    (out / "viz_meta.json").write_text(json.dumps(meta, indent=1))
    print(json.dumps(meta, indent=1), "\n->", out)


if __name__ == "__main__":
    main()
