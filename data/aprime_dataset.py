"""Training / eval samples for A': ``ours`` + degradation + cached StairNet masks, as tensors.

Input recipe (identical for every method, see ``degradation/astra_hard.py``): HR degraded depth
(with holes) -> ``masked_downsample`` by ``scale`` -> LR depth + LR validity.  Holes are filled
(``fill: nearest|zero``) and the validity is a separate channel.  Normalisation = per-frame
min-max over the VALID LR pixels (never the GT), same as the WAVE adapter; ``gt_n`` uses the
same (min, max).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset

from data.degradation import build_degradation, masked_downsample
from data.loaders import build_dataset
from data.stairnet.cache import load_masks
from data.stairnet.edges import build_edge_maps
from utils.misc import REPO_ROOT, data_root, sparse_to_dense_nn

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], np.float32)


def load_split(path: str | Path) -> dict[str, Any]:
    p = Path(path)
    return json.loads((p if p.is_absolute() else REPO_ROOT / p).read_text())


def resolve_cache(cfg: dict) -> str | None:
    c = cfg.get("stair_cache")
    if not c or cfg.get("mask_source", "stairnet_official") == "none":
        return None
    p = Path(c)
    return str(p if p.is_absolute() else data_root() / p)


def build_ours(cfg: dict, split: str, train: bool = False):
    """``BaseDepthDataset`` for one split of the stairs-level split file, degradation attached."""
    d = dict(cfg["data"])
    deg = dict(cfg["degradation"])
    deg["scale"] = int(cfg["model"]["scale"])
    deg["stochastic"] = bool(train)
    deg.setdefault("stair_cache_dir", resolve_cache(d))
    ds_cfg = {k: d[k] for k in ("name", "loader", "root", "crop", "target_size", "min_depth", "max_depth") if k in d}
    ds_cfg["split"] = split
    ds_cfg["stairs"] = load_split(d["split_file"])[split]
    ds_cfg["seed"] = int(cfg.get("seed", 0))
    ds_cfg["lr_scale"] = int(cfg["model"]["scale"])
    return build_dataset(ds_cfg, degradation=build_degradation(deg))


class APrimeDataset(Dataset):
    def __init__(self, cfg: dict, split: str, train: bool = False, with_raw: bool = False) -> None:
        self.cfg, self.train, self.with_raw = cfg, train, with_raw
        self.base = build_ours(cfg, split, train)
        d, l = cfg["data"], cfg["loss"]
        self.scale = int(cfg["model"]["scale"])
        self.cache = resolve_cache(d)
        self.conf_thr = float(d.get("conf_thr", 0.7))
        self.fill = d.get("fill", "nearest")
        self.edge_mode = l.get("edge_weight", "none")
        self.band_px = float(l.get("edge_band_px", 6))
        self.stair_bg = bool(l.get("stair_bg_boundary", True))
        self.flip = train and not l.get("use_planar_loss", False)  # hflip would move cx
        self.lr_min_frac = float(cfg["degradation"].get("lr_min_frac", 0.5))
        if self.cache is None and (self.edge_mode != "none" or l.get("use_planar_loss")):
            raise ValueError("edge_weight / planar loss need StairNet masks: set data.stair_cache (mask_source != none)")

    def set_epoch(self, epoch: int) -> None:
        self.base.degradation.epoch = epoch

    def __len__(self) -> int:
        return len(self.base)

    def masks(self, sample: dict) -> dict[str, np.ndarray]:
        shape = sample["gt_depth"].shape
        if self.cache is None:
            z = np.zeros(shape, bool)
            return {"E": z.astype(np.float32), "band": z, "ignore": z, "tread": z, "riser": z}
        m = load_masks(self.cache, sample["meta"]["sample_id"], shape)
        if m is None:
            raise FileNotFoundError(f"no StairNet cache for {sample['meta']['sample_id']} in {self.cache}")
        e = build_edge_maps(m["cls"], m["conf"], self.edge_mode if self.edge_mode != "none" else "binary",
                            self.band_px, self.conf_thr, self.stair_bg)
        if self.edge_mode == "none":
            e["E"] = np.zeros_like(e["E"])
        ok = m["conf"] >= self.conf_thr
        return {**e, "tread": (m["cls"] == 1) & ok, "riser": (m["cls"] == 2) & ok}

    def __getitem__(self, i: int) -> dict[str, Any]:
        s = self.base[i]
        lr, lr_valid = masked_downsample(s["sparse_depth"], self.scale, self.lr_min_frac)
        if lr_valid.any():
            dmin, dmax = float(lr[lr_valid].min()), float(lr[lr_valid].max())
        if not lr_valid.any() or dmax - dmin < 1e-6:  # degenerate frame: fall back to the sensor range
            dmin, dmax = float(self.base.min_depth), float(self.base.max_depth)
        lr_f = sparse_to_dense_nn(lr) if self.fill == "nearest" and lr_valid.any() else lr
        lr_n = np.where(lr_f > 0, np.clip((lr_f - dmin) / (dmax - dmin), 0, 1), 0).astype(np.float32)
        gt_n = ((s["gt_depth"] - dmin) / (dmax - dmin)).astype(np.float32)
        rgb = ((s["rgb"] - IMAGENET_MEAN) / IMAGENET_STD).astype(np.float32)
        m = self.masks(s)
        t = {
            "rgb": rgb.transpose(2, 0, 1), "lr": lr_n[None], "lr_valid": lr_valid[None].astype(np.float32),
            "gt_n": gt_n[None], "valid": s["mask"][None], "E": m["E"][None], "band": m["band"][None],
            "ignore": m["ignore"][None], "tread": m["tread"][None], "riser": m["riser"][None],
        }
        if self.flip and np.random.default_rng([i, self.base.degradation.epoch]).random() < 0.5:
            t = {k: np.ascontiguousarray(v[..., ::-1]) for k, v in t.items()}
        out = {k: torch.from_numpy(v) for k, v in t.items()}
        out["dmin"], out["dmax"] = torch.tensor([[[dmin]]]), torch.tensor([[[dmax]]])
        out["input_invalid_frac"] = torch.tensor(float((s["sparse_depth"] <= 0).mean()))
        if self.with_raw:
            out["raw"] = s
            out["masks"] = m
        return out
