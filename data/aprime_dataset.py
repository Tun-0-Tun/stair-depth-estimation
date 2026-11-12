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
from data.stairnet.edges import build_edge_maps, clean_masks
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
    from data.stairnet.cache import cache_dir

    return str(cache_dir(c))


def build_ours(cfg: dict, split: str, train: bool = False):
    """``BaseDepthDataset`` for one split of the stairs-level split file, degradation attached.

    ``degradation.preset: <name>`` takes the degradation from ``configs/degradation/<name>.yaml`` (the sensor chains
    of data/degradation, e.g. ``sequence_bicubic_x8_lateral_projector_axial``) and ignores the other keys of the block;
    otherwise the block itself is the (legacy) ``astra_hard`` config.
    """
    d = dict(cfg["data"])
    deg = dict(cfg["degradation"])
    preset = (deg.get("train_preset") if train else None) or deg.get("preset")  # train_preset: training-only randomised chain
    if preset:
        from utils.config import load_config

        deg = dict(load_config("degradation", preset))
    else:
        deg["scale"] = int(cfg["model"]["scale"])
        deg["stochastic"] = bool(train)
        deg.setdefault("stair_cache_dir", resolve_cache(d))
        deg.setdefault("mask_clean", d.get("mask_clean"))
    ds_cfg = {k: d[k] for k in ("name", "loader", "root", "crop", "target_size", "min_depth", "max_depth") if k in d}
    ds_cfg["split"] = split
    ds_cfg["stairs"] = load_split(d["split_file"])[split]
    ds_cfg["seed"] = int(cfg.get("seed", 0))
    ds_cfg["lr_scale"] = int(cfg["model"]["scale"])
    extra = {}
    degradation = build_degradation(deg)
    if getattr(degradation, "uses_normals", False):  # normals are not in `ours`: estimate them from the clean GT
        from data.norm import build_normal_map
        from utils.config import load_config

        nm = dict(load_config("normals", cfg.get("normals", "gt_cross")))
        K = cfg["data"].get("intrinsics")  # frame K of the crop/resize (configs/dataset/ours.yaml)
        if K:
            nm.update(zip(("fx", "fy", "cx", "cy"), map(float, K), strict=True))
        extra["normals"] = build_normal_map(nm)
    return build_dataset(ds_cfg, degradation=degradation, **extra)


def load_gate(path: str | Path) -> dict[str, bool]:
    """``{sample_id: bool}`` from a mask_quality.py / stair_flags.py json: where the stair terms may act (strict stair-like)."""
    return {k: bool(v["stair_like_strict"]) for k, v in load_split(path).items()}


def gt_edge_map(gt: np.ndarray, valid: np.ndarray, rel: float = 0.1, sigma: float = 3.0) -> np.ndarray:
    """exp(-d^2 / 2 sigma^2), d = distance to the nearest GT depth jump (relative jump > ``rel`` between valid neighbours).

    Needs no StairNet: it marks every depth discontinuity (thin railings, bars, stair nosings alike)."""
    from scipy import ndimage

    d = np.where(valid, gt, np.nan).astype(np.float32)
    j = np.zeros(gt.shape, bool)
    with np.errstate(invalid="ignore"):  # a NaN (invalid neighbour) compares False
        dx = np.abs(d[:, 1:] - d[:, :-1]) / np.fmin(d[:, 1:], d[:, :-1]) > rel
        dy = np.abs(d[1:] - d[:-1]) / np.fmin(d[1:], d[:-1]) > rel
    j[:, 1:] |= dx; j[:, :-1] |= dx; j[1:] |= dy; j[:-1] |= dy
    if not j.any():
        return np.zeros(gt.shape, np.float32)
    return np.exp(-(ndimage.distance_transform_edt(~j) ** 2) / (2 * sigma**2)).astype(np.float32)


class APrimeDataset(Dataset):
    """``base`` / ``cache`` / ``gate`` / ``stair_terms`` default to the ``ours`` recipe; ``build_extra`` sets them for other sources.

    ``gate`` {sample_id: bool}: the stair terms (edge weight, planar masks) act only on frames flagged True, every other
    frame is trained with the plain reconstruction loss.  ``stair_terms``: all | edge (no planar masks: another K) | none.
    """

    def __init__(self, cfg: dict, split: str, train: bool = False, with_raw: bool = False, base=None, cache="auto",
                 gate: dict[str, bool] | None = None, stair_terms: str = "all") -> None:
        self.cfg, self.train, self.with_raw, self._epoch = cfg, train, with_raw, 0
        self.base = build_ours(cfg, split, train) if base is None else base
        d, l = cfg["data"], cfg["loss"]
        self.scale = int(cfg["model"]["scale"])
        self.cache = resolve_cache(d) if cache == "auto" else cache
        self.stair_terms = stair_terms
        if gate is None and base is None and train and d.get("stair_gate_file"):
            gate = load_gate(d["stair_gate_file"])
        self.gate = gate
        self.conf_thr = float(d.get("conf_thr", 0.7))
        self.fill = d.get("fill", "nearest")
        self.edge_mode = l.get("edge_weight", "none")
        self.band_px = float(l.get("edge_band_px", 6))
        self.gt_edge = (float(l.get("gt_edge_rel", 0.1)), float(l.get("gt_edge_sigma_px", 3.0))) if float(l.get("gt_edge_alpha", 0.0)) > 0 else None
        self.stair_bg = bool(l.get("stair_bg_boundary", True))
        self.flip = train and not l.get("use_planar_loss", False)  # hflip would move cx
        self.lr_min_frac = float(cfg["degradation"].get("lr_min_frac", 0.5))
        if self.cache is None and stair_terms != "none" and (self.edge_mode != "none" or l.get("use_planar_loss")):
            raise ValueError("edge_weight / planar loss need StairNet masks: set data.stair_cache (mask_source != none)")

    def set_epoch(self, epoch: int) -> None:
        if hasattr(self.base.degradation, "epoch"):  # astra_hard mixes the epoch into its own seed
            self.base.degradation.epoch = epoch
        elif self.train:  # sensor chains: a new but reproducible corruption every epoch via the loader seed
            self.base.seed = int(self.cfg.get("seed", 0)) + 7919 * epoch
        self._epoch = epoch

    def __len__(self) -> int:
        return len(self.base)

    def masks(self, sample: dict) -> dict[str, np.ndarray]:
        shape = sample["gt_depth"].shape
        off = self.stair_terms == "none" or (self.gate is not None and not self.gate.get(str(sample["meta"]["sample_id"]), False))
        if self.cache is None or off:
            z = np.zeros(shape, bool)
            return {"E": z.astype(np.float32), "band": z, "ignore": z, "tread": z, "riser": z}
        m = load_masks(self.cache, sample["meta"]["sample_id"], shape)
        if m is None:
            raise FileNotFoundError(f"no StairNet cache for {sample['meta']['sample_id']} in {self.cache}")
        m["cls"] = clean_masks(m["cls"], self.cfg["data"].get("mask_clean"))
        e = build_edge_maps(m["cls"], m["conf"], self.edge_mode if self.edge_mode != "none" else "binary",
                            self.band_px, self.conf_thr, self.stair_bg)
        if self.edge_mode == "none":
            e["E"] = np.zeros_like(e["E"])
        ok = m["conf"] >= self.conf_thr
        tread, riser = (m["cls"] == 1) & ok, (m["cls"] == 2) & ok
        if self.stair_terms == "edge":  # planar loss uses the camera K of `ours`; another camera -> edge weight only
            tread, riser = np.zeros_like(tread), np.zeros_like(riser)
        return {**e, "tread": tread, "riser": riser}

    def __getitem__(self, i: int) -> dict[str, Any]:
        s = self.base[i]
        if s["sparse_depth"].shape != s["gt_depth"].shape:  # chain that already works at low resolution: use its map as is
            lr = np.asarray(s["lr_depth"], np.float32)
            lr_valid = lr > 0
        else:
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
            "Egt": (gt_edge_map(s["gt_depth"], s["mask"], *self.gt_edge) if self.gt_edge else np.zeros_like(m["E"]))[None],
        }
        if self.flip and np.random.default_rng([i, self._epoch]).random() < 0.5:
            t = {k: np.ascontiguousarray(v[..., ::-1]) for k, v in t.items()}
        out = {k: torch.from_numpy(v) for k, v in t.items()}
        out["dmin"], out["dmax"] = torch.tensor([[[dmin]]]), torch.tensor([[[dmax]]])
        out["input_invalid_frac"] = torch.tensor(float((s["sparse_depth"] <= 0).mean()))  # share of missing pixels of the input map
        if self.with_raw:
            out["raw"] = s
            out["masks"] = m
        return out


def build_extra(cfg: dict, src: dict, train: bool = True) -> APrimeDataset:
    """One extra training source (``data.extra`` entry), degraded by the SAME chain preset as ``ours``.

    src: dataset (configs/dataset/<name>), target_size, split_file?, flags? (stair_flags.py json: keep only these ids and
    gate the stair terms by ``stair_like_strict``), stair_cache, stair_terms (all|edge|none), max_frames?, native_size
    (for the camera K that the noise model's normals need), normals_K? (when the dataset has no K).
    """
    from data.degradation import IdentityDegradation
    from data.norm import build_normal_map
    from data.stairnet.cache import cache_dir
    from utils.config import load_config

    preset = (cfg["degradation"].get("train_preset") if train else None) or cfg["degradation"].get("preset")
    if not preset:
        raise ValueError("data.extra needs degradation.preset (a sensor chain): the legacy astra_hard degrades HR frames of ours only")
    ds_cfg = {k: v for k, v in dict(load_config("dataset", src["dataset"])).items() if k != "split"}
    # force_degradation: a dataset that ships its own sensor input (VOID: ~1500 VIO points) must not use it, only the GT
    ds_cfg.update(split="train", seed=int(cfg.get("seed", 0)), lr_scale=int(cfg["model"]["scale"]), force_degradation=True)
    for k in ("target_size", "split_file"):
        if src.get(k):
            ds_cfg[k] = src[k]
    ds_cfg.update(src.get("loader_cfg") or {})  # any other loader argument (e.g. roots, crop)
    degradation = build_degradation(dict(load_config("degradation", preset)))
    extra = {}
    if getattr(degradation, "uses_normals", False):
        nm = dict(load_config("normals", cfg.get("normals", "gt_cross")))
        K = build_dataset(ds_cfg, degradation=IdentityDegradation())[0]["meta"].get("intrinsics")
        if K and src.get("target_size"):  # the dataset's K belongs to the native frame, the working frame is resized
            (h0, w0), (h1, w1) = src["native_size"], src["target_size"]
            K = [K[0] * w1 / w0, K[1] * h1 / h0, K[2] * w1 / w0, K[3] * h1 / h0]
        K = K or src.get("normals_K")  # e.g. Hypersim: no K in the loader, give one for the working frame
        if K:
            nm.update(zip(("fx", "fy", "cx", "cy"), map(float, K), strict=True))
        extra["normals"] = build_normal_map(nm)
    base = build_dataset(ds_cfg, degradation=degradation, **extra)
    gate = None
    if src.get("flags"):
        gate = load_gate(src["flags"])
        base.records = [r for r in base.records if base.record_id(r) in gate]
    if src.get("max_frames") and len(base.records) > int(src["max_frames"]):
        base.records = base.records[:: -(-len(base.records) // int(src["max_frames"]))]
    terms = src.get("stair_terms", "none")
    cache = None if terms == "none" else str(cache_dir(src.get("stair_cache", f"{src['dataset']}/stairnet_cache")))
    return APrimeDataset(cfg, "train", train, base=base, cache=cache, gate=gate, stair_terms=terms)


class MixedDataset(torch.utils.data.ConcatDataset):
    def set_epoch(self, epoch: int) -> None:
        for d in self.datasets:
            d.set_epoch(epoch)


def build_train(cfg: dict):
    """``ours`` train, plus the ``data.extra`` sources when the config lists any."""
    extras = cfg["data"].get("extra") or []
    ours = [APrimeDataset(cfg, "train", train=True)] if cfg["data"].get("train_ours", True) else []  # False: extras only (NYU reference)
    if not ours and not extras:
        raise ValueError("data.train_ours=false needs data.extra")
    return MixedDataset([*ours, *(build_extra(cfg, s) for s in extras)]) if extras else ours[0]
