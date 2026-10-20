"""Config loading, model building, training and evaluation for A' (and baselines on the same protocol).

Everything is driven by ``configs/aprime/*.yaml`` (+ ``key.sub=value`` overrides on the CLI).
Evaluation scores every method -- A' variants and the baseline adapters -- on the same frames
with the repo's metric code, in three zones: whole frame, stair mask (tread | riser), edge band.
"""

from __future__ import annotations

import copy
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

from data.aprime_dataset import APrimeDataset
from metrics.depth_metrics import DepthMetricAccumulator, MetricConfig
from models.aprime_losses import AprimeLoss, LossConfig
from utils.misc import REPO_ROOT, resolve_device, set_seed

ZONES = ("full", "stairs", "edge")


# ------------------------------------------------------------------ config
def _merge(a: dict, b: dict) -> dict:
    out = copy.deepcopy(a)
    for k, v in b.items():
        out[k] = _merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out


def load_cfg(path: str | Path, overrides: list[str] = ()) -> dict:
    p = Path(path)
    p = p if p.is_absolute() or p.exists() else REPO_ROOT / p
    cfg = yaml.safe_load(p.read_text()) or {}
    if "base" in cfg:  # one level of inheritance, relative to this file
        base = load_cfg(p.parent / cfg.pop("base"))
        base.pop("name", None)
        cfg = _merge(base, cfg)
    for o in overrides:
        key, _, val = o.partition("=")
        node = cfg
        *parents, last = key.split(".")
        for q in parents:
            node = node.setdefault(q, {})
        node[last] = yaml.safe_load(val)
    ds = REPO_ROOT / "configs" / "dataset" / f"{cfg.get('dataset', 'ours')}.yaml"
    cfg["data"] = _merge(yaml.safe_load(ds.read_text()), cfg.get("data", {}))
    cfg.setdefault("name", p.stem)
    return cfg


def intrinsics(cfg: dict, shape=None):
    """``None`` (unknown) | ``(fx, fy, cx, cy)`` | undistorted ray map ``[2,H,W]`` if ``data.camera.use_distortion``.

    ``data.intrinsics`` is the K of the cropped / resized frame; ``data.camera`` holds the full-resolution
    stream (K, size, Orbbec distortion) from which the ray map is built.
    """
    d = cfg["data"]
    k = d.get("intrinsics")
    if not k:
        return None
    K = tuple(float(k[x]) for x in ("fx", "fy", "cx", "cy")) if isinstance(k, dict) else tuple(map(float, k))
    cam = d.get("camera") or {}
    if cam.get("use_distortion") and cam.get("distortion_ob"):
        from utils.camera import frame_rays

        ts = d.get("target_size") or (None if not d.get("crop") else None)
        hw = tuple(shape) if shape else tuple(ts) if ts else None
        if hw is None:
            raise ValueError("distortion needs target_size (or an explicit frame shape)")
        key = (hw, tuple(cam["K"]), tuple(cam["distortion_ob"]), str(d.get("crop")), str(d.get("target_size")))
        if key not in _RAYS:
            _RAYS[key] = frame_rays(hw, cam["K"], cam["distortion_ob"], d.get("crop"), d.get("target_size"), cam.get("size", (1920, 1080)))
        return _RAYS[key]
    return K


_RAYS: dict = {}


# ------------------------------------------------------------------- model
def build_aprime(cfg: dict, device: str):
    from models.wave_aprime import WaveAPrime, load_wave_checkpoint

    m = cfg["model"]
    net = WaveAPrime(
        scale=int(m["scale"]), num_feats=int(m.get("num_feats", 48)),
        use_validity_channel=bool(m.get("use_validity_channel", True)),
        fusion=m.get("fusion", "none"), fusion_levels=tuple(m.get("fusion_levels", (3, 4))),
    )
    if m.get("checkpoint"):
        ck = Path(m["checkpoint"])
        ck = ck if ck.is_absolute() else REPO_ROOT / ck
        load_wave_checkpoint(net, ck)
    if m.get("resume"):
        net.load_state_dict(torch.load(m["resume"], map_location="cpu")["model"])
    return net.to(device)


def build_loss(cfg: dict) -> AprimeLoss:
    l = cfg["loss"]
    K = intrinsics(cfg)
    lc = LossConfig(
        alpha=float(l.get("alpha", 0.0)) if l.get("edge_weight", "none") != "none" else 0.0,
        beta=float(l.get("beta", 0.0)) if l.get("use_planar_loss", False) else 0.0,
        planar_warmup_steps=int(l.get("planar_warmup_steps", 1000)),
        huber_delta=float(l.get("huber_delta", 0.1)), normal_source=l.get("normal_source", "gt"),
        min_tread_px=int(l.get("min_tread_px", 500)), lambda_grad=float(l.get("lambda_grad", 0.1)),
        gate_cos_tread=float(l.get("gate_cos_tread", 0.85)), gate_cos_riser=float(l.get("gate_cos_riser", 0.5)),
        min_inlier_frac=float(l.get("min_inlier_frac", 0.5)),
    )
    return AprimeLoss(lc, K)


def to_dev(batch: dict, device: str) -> dict:
    return {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items() if k not in ("raw", "masks")}


# ---------------------------------------------------------------- training
def train(cfg: dict, out_dir: Path, smoke: int = 0) -> Path:
    t = cfg["train"]
    set_seed(int(cfg.get("seed", 0)))
    device = resolve_device(cfg.get("device", "auto"))
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "config.json").write_text(json.dumps(cfg, indent=1, default=str))
    ds = APrimeDataset(cfg, "train", train=True)
    val = APrimeDataset(cfg, "val", with_raw=True)
    if smoke:
        ds.base.records, val.base.records = ds.base.records[:2], val.base.records[:1]
    bs = int(t.get("batch_size", 2))
    if cfg["model"].get("fusion") == "stairnetv2_orig" and bs < 2:
        raise ValueError("fusion=stairnetv2_orig uses BatchNorm on pooled features: batch_size >= 2")
    gen = torch.Generator().manual_seed(int(cfg.get("seed", 0)))
    loader = DataLoader(ds, bs, shuffle=not smoke, drop_last=len(ds) >= bs, generator=gen,
                        num_workers=int(t.get("workers", 0)), pin_memory=device == "cuda")
    net, loss_fn = build_aprime(cfg, device), build_loss(cfg)
    new = [p for n, p in net.named_parameters() if p.requires_grad and (".fusion." in n or "init_depth_projection" in n)]
    new_ids = {id(p) for p in new}
    old = [p for p in net.parameters() if p.requires_grad and id(p) not in new_ids]
    opt = torch.optim.AdamW([{"params": old, "lr": float(t["lr"])},
                             {"params": new, "lr": float(t.get("lr_new", t["lr"]))}], weight_decay=float(t.get("wd", 1e-4)))
    epochs = int(t["epochs"])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, epochs * max(1, len(loader))))
    amp = device == "cuda" and t.get("amp", "bf16") == "bf16"
    log, step, best = (out_dir / "train_log.jsonl").open("a"), 0, float("inf")
    first = last = None
    batch0 = None
    for ep in range(epochs if not smoke else 10**9):
        ds.set_epoch(ep)
        net.train()
        for batch in loader:
            if smoke:
                batch0 = batch0 or batch
                batch = batch0  # overfit one batch: the loss must go down
            b = to_dev(batch, device)
            with torch.autocast("cuda", torch.bfloat16, enabled=amp):
                pred = net(b["rgb"], b["lr"], b["lr_valid"])
            losses = loss_fn(pred.float(), b, step)
            opt.zero_grad(set_to_none=True)
            losses["total"].backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), float(t.get("clip", 1.0)))
            opt.step(); sched.step(); step += 1
            row = {"step": step, "epoch": ep, **{k: float(v.detach()) for k, v in losses.items()}}
            assert np.isfinite(row["total"]), f"non-finite loss at step {step}: {row}"
            log.write(json.dumps(row) + "\n"); log.flush()
            first = row["total"] if first is None else first
            last = row["total"]
            if step % 20 == 0 or smoke:
                print(" ".join(f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}" for k, v in row.items()))
            if smoke and step >= smoke:
                assert last < first, f"smoke: loss did not decrease ({first:.4f} -> {last:.4f})"
                print(f"[smoke] OK: all losses finite, total {first:.4f} -> {last:.4f}")
                return out_dir
        if not smoke and (ep + 1) % int(t.get("val_every", 1)) == 0:
            res = evaluate(cfg, lambda it: predict_aprime(net, it, device), val)
            print(f"[val] epoch {ep}: full absrel={res['full']['absrel']:.4f} rmse={res['full']['rmse']:.4f}")
            log.write(json.dumps({"epoch": ep, "val": res}) + "\n"); log.flush()
            torch.save({"model": net.state_dict(), "cfg": cfg, "epoch": ep}, out_dir / "last.pth")
            if res["full"]["absrel"] < best:
                best = res["full"]["absrel"]
                torch.save({"model": net.state_dict(), "cfg": cfg, "epoch": ep}, out_dir / "best.pth")
    return out_dir


# -------------------------------------------------------------- evaluation
@torch.no_grad()
def predict_aprime(net, item: dict, device: str) -> np.ndarray:
    b = to_dev({k: v[None] for k, v in item.items() if torch.is_tensor(v)}, device)
    pred_n = net(b["rgb"], b["lr"], b["lr_valid"])
    return (pred_n * (b["dmax"] - b["dmin"]) + b["dmin"])[0, 0].float().cpu().numpy()


def baseline_predictor(cfg: dict, device: str):
    """Adapter from ``baselines/`` fed with the same degraded input (its own input_modality)."""
    from baselines.base import build_baseline

    spec = cfg["baseline"]
    mcfg = yaml.safe_load((REPO_ROOT / "configs" / "model" / f"{spec['model']}.yaml").read_text())
    mcfg.update(spec.get("overrides", {}))
    d = cfg["data"]
    model = build_baseline(mcfg, device=device, min_depth=float(mcfg.get("min_depth", d["min_depth"])),
                           max_depth=float(mcfg.get("max_depth", d["max_depth"])))
    model.load(mcfg.get("checkpoint"))
    return lambda it: model.predict_sample(it["raw"]), model


def evaluate(cfg: dict, predict, ds: APrimeDataset, limit: int | None = None) -> dict[str, Any]:
    d = cfg["data"]
    mc = MetricConfig(min_depth=float(d["min_depth"]), max_depth=float(d["max_depth"]), crop="none")
    acc = {z: DepthMetricAccumulator(mc, pooling="image") for z in ZONES}
    K, geo, inv = intrinsics(cfg), [], []
    has_masks = ds.cache is not None
    n = len(ds) if limit is None else min(limit, len(ds))
    if not has_masks:
        print("[eval] no StairNet masks: stair/edge zones are NaN")
    t0 = time.perf_counter()
    for i in range(n):
        it = ds[i]
        pred, raw, m = predict(it), it["raw"], it["masks"]
        gt, valid = raw["gt_depth"], raw["mask"]
        acc["full"].update(pred, gt, valid)
        if has_masks:
            acc["stairs"].update(pred, gt, valid & (m["tread"] | m["riser"]))
            acc["edge"].update(pred, gt, valid & m["band"])
            if K is not None:
                from metrics.stair_geometry import step_geometry_error

                g = step_geometry_error(np.clip(pred, mc.min_depth, mc.max_depth), gt, m["tread"], K)
                if g:
                    geo.append(g)
        inv.append(float(it["input_invalid_frac"]))
    out: dict[str, Any] = {}
    for z in ZONES:
        r = acc[z].compute()
        out[z] = {k: float(r[k]) for k in ("absrel", "rmse", "delta1")} | {"n": int(r.get("n_samples", 0))}
    out["input_invalid_frac"] = float(np.mean(inv))
    out["n_frames"], out["seconds"] = n, time.perf_counter() - t0
    if geo:
        for k in geo[0]:
            out[f"step_{k}"] = float(np.nanmedian(np.concatenate([g[k] for g in geo])))  # median: robust to merged / missing regions
    return out
