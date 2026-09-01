"""ZJU-L5 -- real VL53L5CX dToF + RGB, released with Deltar (ECCV 2022).

The primary real-world benchmark for DEPTHOR / DEPTHOR++, so it is the dataset
our cross-check against their published numbers runs on.

On-disk layout (identical to what ``third_party/depthor`` expects)::

    <root>/
        data.json                 # {"train": [...], "test": [{"filename": ...}]}
        theater/<timestamp>.h5
        lab1/ cafe1/ cafe2/

Each HDF5 holds:

======================  ==========================================================
``rgb``                 (480, 640, 3) uint8
``depth``               (480, 640) float32, metres, stereo-derived reference
``hist_data``           (64, 2) float32 -- per-zone (mean, variance) of the dToF
                        histogram peak
``fr``                  (64, 4) -- each zone's pixel rectangle (y0, x0, y1, x1)
``mask``                (64,) -- per-zone validity
======================  ==========================================================

Sparse-depth construction reproduces
``third_party/depthor/src/utils/dataloader.py::dtof_to_sparse_depth`` exactly:
one pixel per valid zone, at the zone rectangle's centre, carrying the zone's
mean depth.  ``footprint="zone"`` instead writes the whole zone rectangle, which
is what a method expecting a dense-ish patch input wants; the DEPTHOR
comparison must use ``footprint="center"``.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from data.loaders.base import BaseDepthDataset, DatasetInfo, register_dataset

__all__ = ["ZJUL5Dataset"]


@register_dataset("zju_l5")
class ZJUL5Dataset(BaseDepthDataset):
    info = DatasetInfo(
        name="zju_l5",
        modality="sparse",
        min_depth=1e-3,
        max_depth=10.0,
        native_size=(480, 640),
        sensor="ST VL53L5CX 8x8 dToF + RGB, stereo reference depth",
        url="https://github.com/zju3dv/deltar",
        notes="Real dToF. The DEPTHOR/DEPTHOR++ headline benchmark.",
    )

    def __init__(
        self,
        root: str | Path,
        split: str = "test",
        index_file: str = "data.json",
        footprint: str = "center",
        **kwargs: Any,
    ) -> None:
        if footprint not in ("center", "zone"):
            raise ValueError(f"footprint must be 'center' or 'zone', got {footprint!r}")
        self.index_file = index_file
        self.footprint = footprint
        super().__init__(root=root, split=split, **kwargs)

    def _build_index(self) -> Sequence[Any]:
        index_path = self.root / self.index_file
        if not index_path.exists():
            if self.strict:
                raise FileNotFoundError(
                    f"[zju_l5] {index_path} not found. Download ZJU-L5 from the Deltar repo "
                    "and keep its original layout -- see docs/datasets.md."
                )
            return []
        payload = json.loads(index_path.read_text(encoding="utf-8"))
        entries = payload.get(self.split) or payload.get("test") or []
        return [
            {"id": e["filename"], "path": self.root / e["filename"]}
            for e in entries
            if (self.root / e["filename"]).exists() or not self.strict
        ]

    def _load_raw(self, record: Mapping[str, Any]) -> Mapping[str, Any]:
        try:
            import h5py
        except ImportError as exc:  # pragma: no cover
            raise ImportError("zju_l5 needs h5py: pip install h5py") from exc

        with h5py.File(record["path"], "r") as f:
            rgb = np.array(f["rgb"])
            depth = np.array(f["depth"], dtype=np.float32)
            hist = np.array(f["hist_data"], dtype=np.float32)
            fr = np.array(f["fr"])
            zone_mask = np.array(f["mask"])

        rgb = rgb.astype(np.float32) / 255.0
        depth = np.nan_to_num(depth, nan=0.0, posinf=0.0, neginf=0.0)
        h, w = depth.shape[:2]

        sparse = np.zeros((h, w), dtype=np.float32)
        means = hist[:, 0]
        valid = np.asarray(zone_mask).reshape(-1) > 0
        rects = np.asarray(fr).reshape(-1, 4)
        for k in np.flatnonzero(valid):
            y0, x0, y1, x1 = rects[k]
            if self.footprint == "center":
                cy = int(np.clip((y0 + y1) // 2, 0, h - 1))
                cx = int(np.clip((x0 + x1) // 2, 0, w - 1))
                sparse[cy, cx] = means[k]
            else:
                ya, yb = int(np.clip(y0, 0, h)), int(np.clip(y1, 0, h))
                xa, xb = int(np.clip(x0, 0, w)), int(np.clip(x1, 0, w))
                sparse[ya:yb, xa:xb] = means[k]

        return {
            "rgb": rgb,
            "gt_depth": depth,
            "sparse_depth": sparse,
            "sample_id": record["id"],
            "rgb_path": str(record["path"]),
            "depth_path": str(record["path"]),
            "meta": {
                "n_zones_valid": int(valid.sum()),
                "footprint": self.footprint,
                "dtof_variance_mean": float(hist[valid, 1].mean()) if valid.any() else 0.0,
            },
        }
