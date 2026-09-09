"""MinJiang -- dual-view RGB-D of obstacles a robot meets, including staircases.

Real consumer RGB-D on real stairs, two synchronised viewpoints per frame.
Useful as an in-domain stair set; weak as a benchmark, because the depth *is*
the sensor output, not an independent reference (see the caveat below).

Layout (unchanged from the release)::

    <root>/{STAIRS,ELEVATOR}/
        CAM1_RGB/<timestamp>_rgb_cam1.png          # 640x480 uint8
        CAM1_GDEP/<timestamp>_grey_depth_cam1.png  # 640x480 uint16 millimetres
        CAM1_DEP/<timestamp>_depth_cam1.png        # colourised preview, NOT depth
        CAM2_*/                                    # the second viewpoint

Two things to get right, both of which silently produce wrong numbers:

* **``_DEP`` is not depth.**  It is an 8-bit colourised visualisation. The real
  measurement is ``_GDEP`` ("grey depth"), uint16 millimetres. This loader only
  ever reads ``_GDEP``.
* **The depth is the ground truth *and* the input.**  There is no independent
  reference, so a completion method evaluated here is scored against the very
  sensor it is meant to fix; holes in the sensor output are simply excluded from
  the mask. That makes RMSE optimistic. Use MinJiang for training and for
  qualitative stair results, and say so next to any number from it.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from data.loaders.base import BaseDepthDataset, DatasetInfo, register_dataset

__all__ = ["MinJiangDataset"]


@register_dataset("minjiang")
class MinJiangDataset(BaseDepthDataset):
    info = DatasetInfo(
        name="minjiang",
        modality="gt_only",
        min_depth=0.3,
        max_depth=8.0,
        depth_unit_per_metre=1000.0,
        native_size=(480, 640),
        sensor="consumer RGB-D, two viewpoints (CAM1 / CAM2)",
        url="https://github.com/Zhoujiahuan1005/MinJiang-Dataset",
        notes="Depth is the sensor output, not an independent reference. _DEP is a preview, use _GDEP.",
    )

    def __init__(
        self,
        root: str | Path = "MinJiang-Dataset",
        split: str = "test",
        scenes: Sequence[str] = ("STAIRS",),
        cameras: Sequence[str] = ("CAM1",),
        **kwargs: Any,
    ) -> None:
        self.scenes = list(scenes)
        self.cameras = list(cameras)
        super().__init__(root=root, split=split, **kwargs)

    def _build_index(self) -> Sequence[Any]:
        records: list[dict[str, Any]] = []
        for scene in self.scenes:
            for cam in self.cameras:
                rgb_dir = self.root / scene / f"{cam}_RGB"
                dep_dir = self.root / scene / f"{cam}_GDEP"
                if not rgb_dir.is_dir():
                    continue
                for rgb in sorted(rgb_dir.glob("*.png")):
                    # 1693272485.544243_rgb_cam1.png -> 1693272485.544243_grey_depth_cam1.png
                    stamp = rgb.name.split("_rgb_")[0]
                    dep = dep_dir / f"{stamp}_grey_depth_{cam.lower()}.png"
                    if not dep.exists():
                        continue
                    records.append(
                        {
                            "id": f"{scene}/{cam}/{stamp}",
                            "scene": scene,
                            "camera": cam,
                            "rgb": rgb,
                            "depth": dep,
                        }
                    )

        if not records and self.strict:
            raise FileNotFoundError(
                f"[minjiang] no frames under {self.root} for scenes={self.scenes} "
                f"cameras={self.cameras}.\n"
                "Expected <root>/STAIRS/CAM1_RGB/*.png with matching CAM1_GDEP/*.png -- "
                "see docs/datasets.md."
            )
        return records

    def _load_raw(self, record: Mapping[str, Any]) -> Mapping[str, Any]:
        from PIL import Image

        rgb = np.asarray(Image.open(record["rgb"]).convert("RGB"), dtype=np.float32) / 255.0
        depth = np.asarray(Image.open(record["depth"]), dtype=np.float32)
        if depth.ndim != 2:
            raise ValueError(
                f"[minjiang] {record['depth']} is not a single-channel 16-bit image. "
                "Are you pointing at CAM*_DEP (the colourised preview) instead of CAM*_GDEP?"
            )
        depth /= self.info.depth_unit_per_metre

        return {
            "rgb": rgb,
            "gt_depth": depth,
            "sample_id": record["id"],
            "rgb_path": str(record["rgb"]),
            "depth_path": str(record["depth"]),
            "meta": {
                "scene": record["scene"],
                "camera": record["camera"],
                "gt_source": "sensor (not an independent reference)",
            },
        }
