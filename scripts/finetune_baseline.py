#!/usr/bin/env python3
"""Fine-tune DuCos / DEPTHOR on OUR data under OUR sensor chain, then evaluate like every other row (same test, metrics).

    python scripts/finetune_baseline.py configs/aprime/r_ducos.yaml --out outputs/aprime_e30/$X/DuCos_ft degradation.preset=$X
    python scripts/finetune_baseline.py configs/aprime/r_depthor.yaml --out outputs/aprime_e30/$X/DEPTHOR_ft degradation.preset=$X
    python scripts/finetune_baseline.py configs/aprime/r_depthor_blocks.yaml --out .../DEPTHOR_blocks_ft ...

The model, its input preparation and the pre/post-processing are the adapter's own (``baselines/``): the same code path that gives the
zero-shot rows, with gradients.  Loss: masked L1 in metres on the GT-valid pixels.  Model selection: val AbsRel (ours val) after every
epoch; ``best.pth`` is loadable by the adapter (``baseline.overrides.checkpoint=...``).  Writes ``test.json`` (+``val.json``,
``train_log.jsonl``, ``config.json``) like ``train_aprime.py``.  Flags: --epochs (10) --lr (2e-5) --accum (4) --smoke N (N steps).
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import _bootstrap  # noqa: F401
import numpy as np
import torch
import torch.nn.functional as F

from data.aprime_dataset import APrimeDataset
from models.aprime_runner import baseline_predictor, evaluate, load_cfg
from utils.misc import resize_depth, resize_rgb, resize_sparse, resolve_device, set_seed


def forward_ducos(ad, raw, dev):
    rgb, dep = raw["rgb"].astype(np.float32), raw["lr_depth"].astype(np.float32)
    h, w = rgb.shape[:2]
    if dep.shape != (h, w):
        dep = resize_depth(dep, (h, w), "bicubic")
    lo, hi = float(dep.min()), float(dep.max())
    if hi - lo < 1e-6:
        return None
    r0, r1 = float(rgb.min()), float(rgb.max())
    rgb_n = rgb if r1 - r0 < 1e-6 else (rgb - r0) / (r1 - r0)
    rgb_t = torch.from_numpy(rgb_n.transpose(2, 0, 1).copy())[None].to(dev).float()
    dep_t = torch.from_numpy(((dep - lo) / (hi - lo)).astype(np.float32))[None, None].to(dev)
    pred = ad.model({"rgb": rgb_t, "dep": dep_t})["pred"]
    return (pred * (hi - lo) + lo).clamp(ad.min_depth, ad.max_depth).reshape(1, 1, h, w)


def forward_depthor(ad, raw, dev):
    rgb = raw["rgb"].astype(np.float32)
    h, w = rgb.shape[:2]
    th, tw = ad.native_size
    image = torch.from_numpy(resize_rgb(rgb, (th, tw)).transpose(2, 0, 1).copy())[None].to(dev).float()
    sparse = torch.from_numpy(resize_sparse(raw["sparse_depth"].astype(np.float32), (th, tw)))[None, None].to(dev)
    _, final = ad.model({"image": image, "sparse_depth": sparse})
    return F.interpolate(final.float().clamp(ad.min_depth, ad.max_depth), (h, w), mode="bilinear", align_corners=False)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("overrides", nargs="*")
    ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--accum", type=int, default=4, help="samples per optimizer step (batch size 1 each)")
    ap.add_argument("--smoke", type=int, default=0)
    a = ap.parse_args()
    cfg = load_cfg(a.config, a.overrides)
    set_seed(int(cfg.get("seed", 0)))
    dev = resolve_device(cfg.get("device", "auto"))
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "config.json").write_text(json.dumps({**cfg, "finetune": vars(a)}, indent=1, default=str))
    predict, ad = baseline_predictor(cfg, dev)
    fwd = {"ducos": forward_ducos, "depthor": forward_depthor}[cfg["baseline"]["model"]]
    key = "state_dict" if cfg["baseline"]["model"] == "ducos" else "model"
    params = [p for p in ad.model.parameters() if p.requires_grad]
    print(f"[ft] {cfg['baseline']['model']}: {sum(p.numel() for p in params) / 1e6:.1f}M trainable parameters")
    ds = APrimeDataset(cfg, "train", train=True, with_raw=True, cache=None)
    ds.flip = False
    val = APrimeDataset(cfg, "val", with_raw=True)
    if a.smoke:
        ds.base.records, val.base.records = ds.base.records[: a.smoke], val.base.records[:2]
    opt = torch.optim.AdamW(params, lr=a.lr, weight_decay=1e-4)
    n_steps = max(1, a.epochs * len(ds) // a.accum)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=n_steps)
    log, best, step = (out / "train_log.jsonl").open("a"), float("inf"), 0
    for ep in range(a.epochs):
        ds.set_epoch(ep)
        ad.model.train()
        t0, acc, run = time.perf_counter(), 0, []
        for i in np.random.default_rng(ep).permutation(len(ds)):
            it = ds[int(i)]
            raw = predict.prepare(it)
            pred = fwd(ad, raw, dev)
            if pred is None:
                continue
            gt = torch.from_numpy(raw["gt_depth"].astype(np.float32))[None, None].to(dev)
            valid = torch.from_numpy(np.asarray(raw["mask"], bool))[None, None].to(dev)
            loss = (pred - gt).abs()[valid].mean() / a.accum
            loss.backward()
            run.append(float(loss) * a.accum)
            acc += 1
            if acc == a.accum:
                torch.nn.utils.clip_grad_norm_(params, 1.0)
                opt.step(); opt.zero_grad(set_to_none=True); sched.step()
                acc, step = 0, step + 1
                if step % 20 == 0:
                    print(f"step={step} epoch={ep} l1={np.mean(run[-20 * a.accum:]):.4f}", flush=True)
        ad.model.eval()
        with torch.no_grad():
            res = evaluate(cfg, predict, val, None)
        log.write(json.dumps({"epoch": ep, "train_l1": float(np.mean(run)), "val": res, "seconds": time.perf_counter() - t0}) + "\n"); log.flush()
        print(f"[ft] epoch {ep}: train L1 {np.mean(run):.4f}  val absrel {res['full']['absrel']:.4f}  {time.perf_counter() - t0:.0f}s", flush=True)
        if res["full"]["absrel"] < best:
            best = res["full"]["absrel"]
            torch.save({key: ad.model.state_dict(), "epoch": ep}, out / "best.pth")
            (out / "val.json").write_text(json.dumps(res, indent=1))
    ckpt = out / "best.pth"
    state = torch.load(ckpt, map_location="cpu")[key]
    ad.model.load_state_dict({k.replace("module.", "", 1): v for k, v in state.items()}, strict=False)
    ad.model.eval()
    test = APrimeDataset(cfg, "test", with_raw=True)
    with torch.no_grad():
        res = evaluate(cfg, predict, test, 2 if a.smoke else None)
    res.update(config=Path(a.config).stem + "_ft", split="test", ckpt=str(ckpt))
    (out / "test.json").write_text(json.dumps(res, indent=1))
    print("[ft] test", json.dumps({k: res[k] for k in ("full", "stairs", "edge")}))


if __name__ == "__main__":
    main()
