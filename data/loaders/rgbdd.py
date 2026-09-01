"""RGB-D-D -- real low-resolution ToF (192x144) with a high-resolution reference.

One of DuCos's headline real-world benchmarks (``test_RealRGBDD.py``), so it is
where we cross-check our re-implementation of their evaluation against the
published table.

Layout (unchanged from the official release, and what
``third_party/ducos/data/rgbdd_dataloader.py`` expects)::

    <root>/{Train,Test}/
        RGBDD_RGB/<name>_RGB.jpg          # 512 x 384
        RGBDD_GT/<name>_HR_gt.png         # uint16 mm, 512 x 384   -> GT
        RGBDD_LR/<name>_LR_fill_depth.png # uint16 mm, 192 x 144   -> input

``downsample="real"`` uses the genuine low-resolution ToF frame (the honest
setting).  ``downsample="sync"`` ignores it and bicubically downsamples the GT
by ``scale`` instead -- the synthetic protocol most DSR papers report; we keep
it only to reproduce published numbers.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from data.loaders.base import BaseDepthDataset, DatasetInfo, register_dataset
from utils.misc import downsample_depth

__all__ = ["RGBDDDataset"]

_SPLIT_DIRS = {"train": "Train", "test": "Test", "val": "Test"}


@register_dataset("rgbdd")
class RGBDDDataset(BaseDepthDataset):
    info = DatasetInfo(
        name="rgbdd",
        modality="lr",
        min_depth=0.1,
        max_depth=10.0,
        depth_unit_per_metre=1000.0,
        native_size=(384, 512),
        sensor="Huawei P30 Pro ToF (192x144) + RGB; Lucid Helios reference",
        url="https://github.com/lingzhi96/RGB-D-D-Dataset",
        notes="Access requires emailing the authors for the download link.",
    )

    def __init__(
        self,
        root: str | Path,
        split: str = "test",
        downsample: str = "real",
        scale: int = 4,
        **kwargs: Any,
    ) -> None:
        if downsample not in ("real", "sync"):
            raise ValueError(f"downsample must be 'real' or 'sync', got {downsample!r}")
        self.downsample = downsample
        self.scale = int(scale)
        super().__init__(root=root, split=split, lr_scale=int(scale), **kwargs)

    def _build_index(self) -> Sequence[Any]:
        base = self.root / _SPLIT_DIRS.get(self.split, self.split)
        rgb_dir = base / "RGBDD_RGB"
        if not rgb_dir.is_dir():
            if self.strict:
                raise FileNotFoundError(
                    f"[rgbdd] {rgb_dir} not found. Expected "
                    "<root>/{Train,Test}/{RGBDD_RGB,RGBDD_GT,RGBDD_LR} -- see docs/datasets.md."
                )
            return []
        records = []
        for rgb in sorted(rgb_dir.glob("*_RGB.*")):
            name = rgb.name[: -len("_RGB" + rgb.suffix)]
            gt = base / "RGBDD_GT" / f"{name}_HR_gt.png"
            lr = base / "RGBDD_LR" / f"{name}_LR_fill_depth.png"
            if not gt.exists():
                continue
            records.append({"id": name, "rgb": rgb, "gt": gt, "lr": lr if lr.exists() else None})
        return records

    def _load_raw(self, record: Mapping[str, Any]) -> Mapping[str, Any]:
        from PIL import Image

        scale_m = self.info.depth_unit_per_metre
        rgb = np.asarray(Image.open(record["rgb"]).convert("RGB"), dtype=np.float32) / 255.0
        gt = np.asarray(Image.open(record["gt"]), dtype=np.float32) / scale_m

        if self.downsample == "real":
            if record["lr"] is None:
                raise FileNotFoundError(
                    f"[rgbdd] downsample='real' but no LR frame for {record['id']}; "
                    "either the release is incomplete or use downsample='sync'."
                )
            lr = np.asarray(Image.open(record["lr"]), dtype=np.float32) / scale_m
        else:
            lr = downsample_depth(gt, self.scale, mode="bicubic")

        return {
            "rgb": rgb,
            "gt_depth": gt,
            "lr_depth": lr.astype(np.float32),
            "sample_id": record["id"],
            "rgb_path": str(record["rgb"]),
            "depth_path": str(record["gt"]),
            "meta": {"downsample": self.downsample, "scale": self.scale},
        }
