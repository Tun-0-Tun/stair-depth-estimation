#!/usr/bin/env python3
"""Train our model (``models/pipeline.py``).

Usage::

    python scripts/train.py experiment=ours_stairs train.epochs=50
    python scripts/train.py --dataset hypersim --degradation astra2_uncalibrated

The loop is intentionally plain -- one file, no framework -- because the
research risk in this project is in the *data* (does the Astra 2 degradation
model transfer?) and in the *metrics* (are we comparing like with like?), not
in the training loop.  When we need multi-GPU, swap this for ``accelerate`` and
keep everything else.

Validation uses the same ``metrics.depth_metrics`` code as the benchmark, so a
training curve and a benchmark row are directly comparable.
"""

from __future__ import annotations

import sys
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import _bootstrap  # noqa: F401
import numpy as np
import run_baseline  # reuse the identical config composition
import torch
from torch.utils.data import DataLoader

from data.degradation import build_degradation
from data.loaders import build_dataset
from metrics.depth_metrics import DepthMetricAccumulator, MetricConfig, format_metrics
from models.pipeline import StairDepthNet, StairDepthNetConfig
from utils.misc import REPO_ROOT, resolve_device, set_seed
from utils.results import RunRecorder, get_tracker


def masked_l1_l2(pred: torch.Tensor, gt: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """L1 + L2 on valid pixels only, the loss both baselines train with.

    Averaging over *valid* pixels rather than all pixels matters: with 90% holes
    a mean over the full frame silently scales the gradient by 0.1 and the model
    appears to train ten times slower.
    """
    m = mask.float()
    n = m.sum().clamp_min(1.0)
    diff = (pred - gt) * m
    return diff.abs().sum() / n + (diff**2).sum() / n


def collate(batch: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
    def stack(key, transform=lambda x: x):
        return torch.from_numpy(np.stack([transform(b[key]) for b in batch]))

    return {
        "rgb": stack("rgb", lambda x: x.transpose(2, 0, 1)).float(),
        "gt": stack("gt_depth")[:, None].float(),
        "sparse": stack("sparse_depth")[:, None].float(),
        "mask": stack("mask")[:, None],
    }


def build_loader(cfg, split: str, degradation, batch_size: int, shuffle: bool, workers: int):
    ds_cfg = dict(cfg.dataset)
    ds_cfg["split"] = split
    dataset = build_dataset(ds_cfg, degradation=degradation)
    return dataset, DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=workers,
        collate_fn=collate,
        drop_last=shuffle,
    )


def main(argv: Sequence[str] | None = None) -> int:
    args, overrides = run_baseline.parse_args(list(sys.argv[1:] if argv is None else argv))
    cfg = run_baseline.compose(args, overrides)

    train_cfg = dict(cfg.get("train", {}))
    epochs = int(train_cfg.get("epochs", 20))
    batch_size = int(train_cfg.get("batch_size", 4))
    lr = float(train_cfg.get("lr", 1e-3))
    workers = int(train_cfg.get("workers", 0))
    val_every = int(train_cfg.get("val_every", 1))

    set_seed(int(cfg.get("seed", 0)))
    device = resolve_device(args.device or cfg.get("device", "auto"))

    degradation = build_degradation(dict(cfg.degradation) if "degradation" in cfg else None)
    train_ds, train_loader = build_loader(cfg, "train", degradation, batch_size, True, workers)
    val_ds, val_loader = build_loader(cfg, "val", degradation, 1, False, workers)
    print(f"[data] train={len(train_ds)} val={len(val_ds)} device={device}")

    net_cfg = StairDepthNetConfig(
        **{
            **dict(cfg.model.get("net", {})),
            "min_depth": float(cfg.eval.get("min_depth") or train_ds.min_depth),
            "max_depth": float(cfg.eval.get("max_depth") or train_ds.max_depth),
        }
    )
    model = StairDepthNet(net_cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[model] StairDepthNet, {n_params / 1e6:.2f}M parameters")

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, epochs))

    recorder = RunRecorder(
        experiment=f"train_{cfg.get('experiment_name', 'adhoc')}",
        config=cfg.to_dict(),
        tracker=get_tracker(dict(cfg.get("tracking", {})), "train"),
    )
    ckpt_dir = Path(train_cfg.get("checkpoint_dir", REPO_ROOT / "checkpoints" / "ours"))
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    metric_cfg = MetricConfig(
        min_depth=net_cfg.min_depth,
        max_depth=net_cfg.max_depth,
        crop=str(cfg.eval.get("crop", "none")),
    )
    best = float("inf")

    for epoch in range(1, epochs + 1):
        model.train()
        t0, total, seen = time.perf_counter(), 0.0, 0
        for batch in train_loader:
            rgb = batch["rgb"].to(device)
            sparse = batch["sparse"].to(device)
            gt = batch["gt"].to(device)
            mask = batch["mask"].to(device)

            out = model(rgb, sparse)
            loss = masked_l1_l2(out["pred"], gt, mask)
            if "coarse" in out:  # deep supervision on the pre-refinement output
                loss = loss + 0.3 * masked_l1_l2(out["coarse"], gt, mask)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()

            total += float(loss) * rgb.shape[0]
            seen += rgb.shape[0]
        sched.step()
        train_loss = total / max(seen, 1)
        print(f"[epoch {epoch}/{epochs}] loss={train_loss:.4f} ({time.perf_counter() - t0:.1f}s)")

        if epoch % val_every == 0 and len(val_ds):
            acc = DepthMetricAccumulator(metric_cfg)
            model.eval()
            with torch.no_grad():
                for batch in val_loader:
                    out = model(batch["rgb"].to(device), batch["sparse"].to(device))
                    pred = out["pred"].squeeze().cpu().numpy()
                    acc.update(pred, batch["gt"].squeeze().numpy(), batch["mask"].squeeze().numpy())
            summary = acc.compute()
            print(f"           val  {format_metrics(summary)}")
            if recorder.tracker:
                recorder.tracker.log({"epoch": epoch, "train_loss": train_loss, **summary})
            if summary["rmse"] < best:
                best = summary["rmse"]
                torch.save(
                    {"model": model.state_dict(), "config": net_cfg.__dict__, "epoch": epoch},
                    ckpt_dir / "best.pt",
                )
                print(f"           new best rmse={best:.4f} -> {ckpt_dir / 'best.pt'}")

    torch.save({"model": model.state_dict(), "config": net_cfg.__dict__}, ckpt_dir / "last.pt")
    recorder.finalize({"rmse": best, "n_samples": len(val_ds)}, notes="training run")
    print(f"done. best val rmse={best:.4f}; checkpoints in {ckpt_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
