"""Our own Orbbec Astra 2 captures of staircases (``ours/sample`` on the server).

Layout: one flat folder, six PNGs per shot plus a table::

    <root>/s<id>_rgb.png                     1920x1080 uint8 RGB
    <root>/s<id>_depth_aligned_to_rgb.png    1080x1920 uint16 millimetres, 0 = none, in the RGB frame
    <root>/s<id>_depth_raw.png               1200x1600 uint16 mm, the sensor's own frame (not used)
    <root>/s<id>_mask_aligned_valid.png      255 where the aligned depth is valid
    <root>/s<id>_mask_raw_valid.png          same for the raw depth (not used)
    <root>/s<id>_depth_colorized.png         preview, NOT depth (not used)
    <root>/shots.xlsx                        shot id, stairs id, view, illumination, occlusion, comments

**There is no independent ground truth here**, only the sensor's own depth, so this loader is
``gt_only`` with the sensor depth standing in as GT, exactly like MinJiang: a degradation
(``bicubic_x4``, ``random_sparse_500``, ...) makes the input, and metrics measure how well a method
recovers the sensor's depth.  Numbers are optimistic and are for ranking methods against each other
on real structured-light stairs; they are not accuracy.

The depth covers a central 4:3 part of the 16:9 RGB (the rest of the aligned map is 0), so by
default the frame is cropped to that part: ``crop = (y0, y1, x0, x1)`` in RGB pixels, chosen from the
valid-region box of every 10th shot (y 6-52 .. 1068-1075, x 238-378 .. 1632-1659).  Combine with
``target_size`` (e.g. ``[528, 688]``, exactly half) to keep the methods' run time sane.

Per-shot metadata from ``shots.xlsx`` (``stairs_id``, ``view``, ``illumination``, ``occlusion``,
``comments``) lands in ``meta``; it needs ``openpyxl`` and is skipped with a warning when missing.
``frame_stride=N`` takes every N-th shot (an even spread over all stairs, unlike ``max_samples``,
which takes the first ones).  ``stairs=[...]`` keeps only those stairs, which is how to split without putting one staircase in
both halves.
"""

from __future__ import annotations

import re
import warnings
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from data.loaders.base import BaseDepthDataset, DatasetInfo, register_dataset

__all__ = ["OursDataset"]

_RGB = re.compile(r"^s(\d+)_rgb\.png$")
#: shots.xlsx column -> meta key
_COLUMNS = {
    "stairs id": "stairs_id",
    "view category": "view",
    "illumination": "illumination",
    "occlusion": "occlusion",
    "comments": "comments",
}


def _read_table(path: Path) -> dict[int, dict[str, Any]]:
    """``shot id`` -> metadata, or ``{}`` (with a warning) if the table cannot be read."""
    if not path.exists():
        return {}
    try:
        import pandas as pd

        df = pd.read_excel(path)
    except Exception as exc:  # no openpyxl, a corrupt file, ... never worth losing the images over
        warnings.warn(
            f"[ours] cannot read {path.name}: {exc}; per-shot metadata is skipped", stacklevel=2
        )
        return {}
    out: dict[int, dict[str, Any]] = {}
    for row in df.to_dict("records"):
        meta = {}
        for col, key in _COLUMNS.items():
            v = row.get(col)
            meta[key] = None if v is None or (isinstance(v, float) and np.isnan(v)) else v
        if meta["stairs_id"] is not None:
            meta["stairs_id"] = int(meta["stairs_id"])
        out[int(row["shot id"])] = meta
    return out


@register_dataset("ours")
class OursDataset(BaseDepthDataset):
    info = DatasetInfo(
        name="ours",
        modality="gt_only",
        min_depth=0.6,
        max_depth=8.0,
        depth_unit_per_metre=1000.0,
        native_size=(1080, 1920),
        sensor="Orbbec Astra 2 (structured light), depth aligned to the RGB camera",
        notes="GT is the sensor's own depth, not an independent reference.",
    )

    def __init__(
        self,
        root: str | Path = "ours/sample",
        split: str = "test",
        stairs: Sequence[int] | None = None,
        crop: Sequence[int] | None = (12, 1068, 256, 1632),
        frame_stride: int = 1,
        **kwargs: Any,
    ) -> None:
        self.frame_stride = max(1, int(frame_stride))
        self.stairs = None if stairs is None else {int(s) for s in stairs}
        self.crop = None if crop is None else tuple(int(c) for c in crop)
        super().__init__(root=root, split=split, **kwargs)

    def _build_index(self) -> Sequence[Any]:
        table = _read_table(self.root / "shots.xlsx")
        if self.stairs is not None and not table:
            raise RuntimeError(
                "[ours] stairs=[...] needs shots.xlsx (and openpyxl) to know which shot is on "
                "which staircase"
            )
        records: list[dict[str, Any]] = []
        for rgb in self.root.glob("s*_rgb.png"):
            m = _RGB.match(rgb.name)
            if not m:
                continue
            sid = int(m.group(1))
            depth = self.root / f"s{sid}_depth_aligned_to_rgb.png"
            mask = self.root / f"s{sid}_mask_aligned_valid.png"
            if not (depth.exists() and mask.exists()):
                continue
            meta = table.get(sid, {})
            if self.stairs is not None and meta.get("stairs_id") not in self.stairs:
                continue
            records.append(
                {
                    "id": f"s{sid}",
                    "num": sid,
                    "rgb": rgb,
                    "depth": depth,
                    "mask": mask,
                    "meta": meta,
                }
            )
        records.sort(key=lambda r: r["num"])
        return records[:: self.frame_stride]

    def _load_raw(self, record: Mapping[str, Any]) -> Mapping[str, Any]:
        from PIL import Image

        rgb = np.asarray(Image.open(record["rgb"]).convert("RGB"), dtype=np.float32) / 255.0
        depth = np.asarray(Image.open(record["depth"]), dtype=np.float32)
        mask = np.asarray(Image.open(record["mask"])) > 0
        if depth.ndim != 2 or depth.shape != rgb.shape[:2] or mask.shape != depth.shape:
            raise ValueError(
                f"[ours] {record['id']}: rgb {rgb.shape}, depth {depth.shape}, mask {mask.shape} "
                "must all be the same frame; use depth_aligned_to_rgb, not depth_raw"
            )
        depth /= self.info.depth_unit_per_metre
        if self.crop is not None:
            y0, y1, x0, x1 = self.crop
            rgb, depth, mask = rgb[y0:y1, x0:x1], depth[y0:y1, x0:x1], mask[y0:y1, x0:x1]

        return {
            "rgb": np.ascontiguousarray(rgb),
            "gt_depth": np.ascontiguousarray(depth),
            "mask": np.ascontiguousarray(mask),
            "sample_id": record["id"],
            "rgb_path": str(record["rgb"]),
            "depth_path": str(record["depth"]),
            "meta": {**record["meta"], "gt_source": "sensor (not an independent reference)"},
        }
