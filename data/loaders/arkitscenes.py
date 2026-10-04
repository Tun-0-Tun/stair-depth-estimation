"""ARKitScenes, depth-upsampling subset -- a *real* low-res sensor with laser GT.

Baruch et al., "ARKitScenes", NeurIPS 2021 Datasets track.
https://github.com/apple/ARKitScenes  (DATA.md, section "depth upsampling")

Not stairs, but the only dataset here where the SR input is a real phone sensor
(iPad LiDAR, 256x192) and the reference is independent of it (Faro laser scan
rendered into the RGB frame), so nothing is simulated on either side.

Layout (unchanged from the release)::

    <root>/data/upsampling/{Training,Validation}/<video_id>/
        wide/<video_id>_<ts>.png           # 1920x1440 RGB, uint8
        lowres_depth/<video_id>_<ts>.png   # 256x192 uint16 mm, ARKit LiDAR depth
        confidence/<video_id>_<ts>.png     # 256x192 uint8, 0/1/2 = low/med/high
        highres_depth/<video_id>_<ts>.png  # 1920x1440 uint16 mm, laser GT, 0 = none

The resolution ratio is 7.5, not an integer: ``meta["lr_scale"]`` rounds it to
8, and the SR adapters resample to their own grid anyway.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from data.loaders.base import BaseDepthDataset, DatasetInfo, register_dataset

__all__ = ["ARKitScenesDataset"]

_SPLIT_DIRS = {"train": "Training", "val": "Validation", "test": "Validation"}


@register_dataset("arkitscenes")
class ARKitScenesDataset(BaseDepthDataset):
    info = DatasetInfo(
        name="arkitscenes",
        modality="lr",
        min_depth=0.05,
        max_depth=10.0,
        depth_unit_per_metre=1000.0,
        native_size=(1440, 1920),
        sensor="iPad Pro LiDAR (ARKit depth, 256x192); GT from a Faro laser scanner",
        url="https://github.com/apple/ARKitScenes",
        notes="Real LR input, independent laser GT. Official upsampling split has no test set: test = Validation.",
    )

    def __init__(
        self,
        root: str | Path = "ARKitScenes",
        split: str = "test",
        frame_stride: int = 1,
        **kwargs: Any,
    ) -> None:
        self.frame_stride = max(1, int(frame_stride))
        super().__init__(root=root, split=split, **kwargs)

    def _build_index(self) -> Sequence[Any]:
        split_dir = self.root / "data" / "upsampling" / _SPLIT_DIRS[self.split]
        records: list[dict[str, Any]] = []
        for video in sorted(p for p in split_dir.glob("*") if p.is_dir()):
            for gt in sorted((video / "highres_depth").glob("*.png"))[:: self.frame_stride]:
                rgb, lr = video / "wide" / gt.name, video / "lowres_depth" / gt.name
                if rgb.exists() and lr.exists():
                    records.append(
                        {
                            "id": f"{video.name}/{gt.stem}",
                            "video": video.name,
                            "rgb": rgb,
                            "lr": lr,
                            "gt": gt,
                        }
                    )

        if not records and self.strict:
            raise FileNotFoundError(
                f"[arkitscenes] no frames under {split_dir}.\n"
                "Expected <root>/data/upsampling/Validation/<video>/{wide,lowres_depth,highres_depth}/*.png "
                "-- see docs/datasets.md."
            )
        return records

    def _load_raw(self, record: Mapping[str, Any]) -> Mapping[str, Any]:
        from PIL import Image

        mm = self.info.depth_unit_per_metre
        return {
            "rgb": np.asarray(Image.open(record["rgb"]).convert("RGB"), dtype=np.float32) / 255.0,
            "gt_depth": np.asarray(Image.open(record["gt"]), dtype=np.float32) / mm,
            "lr_depth": np.asarray(Image.open(record["lr"]), dtype=np.float32) / mm,
            "sample_id": record["id"],
            "rgb_path": str(record["rgb"]),
            "depth_path": str(record["gt"]),
            "meta": {"scene": record["video"], "gt_source": "Faro laser scan (independent)"},
        }
