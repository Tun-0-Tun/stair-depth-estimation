"""ARKitScenes -- ``depth_upsampling`` subset.

Layout produced by ``download_data.py --dataset arkitscenes`` (which wraps
Apple's ``download_data.py`` from https://github.com/apple/ARKitScenes)::

    <root>/upsampling/<Training|Validation>/<video_id>/
        wide/<video_id>_<timestamp>.png          # RGB, 1920x1440
        highres_depth/<video_id>_<timestamp>.png # uint16 mm, 1920x1440  -> GT
        lowres_depth/<video_id>_<timestamp>.png  # uint16 mm,  256x192   -> input
        confidence/<video_id>_<timestamp>.png    # 0/1/2 ARKit confidence
        wide_intrinsics/<video_id>_<timestamp>.pincam

This dataset is the closest public analogue of our target setup: a real
low-resolution consumer depth sensor (Apple's lidar) with a real high-resolution
reference, i.e. a *native* LR input rather than a simulated one.  ``lowres_depth``
becomes ``lr_depth`` and no degradation model is involved.

``confidence`` is exposed two ways: ``min_confidence`` drops low-confidence LR
pixels from the input (0 = keep everything, 2 = only high confidence), and the
raw map is returned in ``meta`` for analysis.
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
        min_depth=0.1,
        max_depth=10.0,
        depth_unit_per_metre=1000.0,
        native_size=(1440, 1920),
        sensor="Apple lidar (dToF) 256x192 + FaceTime HD RGB",
        url="https://github.com/apple/ARKitScenes",
        notes="Native low-resolution depth input; highres_depth is the GT.",
    )

    def __init__(
        self,
        root: str | Path,
        split: str = "val",
        video_ids: Sequence[str] | None = None,
        min_confidence: int = 0,
        frame_stride: int = 1,
        target_size: Sequence[int] | None = (480, 640),
        **kwargs: Any,
    ) -> None:
        self.video_ids = list(video_ids) if video_ids else None
        self.min_confidence = int(min_confidence)
        self.frame_stride = max(1, int(frame_stride))
        super().__init__(root=root, split=split, target_size=target_size, **kwargs)

    def _split_root(self) -> Path:
        sub = _SPLIT_DIRS.get(self.split, self.split)
        for candidate in (self.root / "upsampling" / sub, self.root / sub, self.root):
            if candidate.exists():
                return candidate
        return self.root / "upsampling" / sub

    def _build_index(self) -> Sequence[Any]:
        base = self._split_root()
        records: list[dict[str, Any]] = []
        video_dirs = sorted(p for p in base.glob("*") if (p / "lowres_depth").is_dir())
        if self.video_ids:
            wanted = set(self.video_ids)
            video_dirs = [p for p in video_dirs if p.name in wanted]
        for vid in video_dirs:
            frames = sorted((vid / "lowres_depth").glob("*.png"))[:: self.frame_stride]
            for lr in frames:
                stem = lr.stem
                hr = vid / "highres_depth" / f"{stem}.png"
                rgb = vid / "wide" / f"{stem}.png"
                if not rgb.exists():
                    rgb = vid / "wide" / f"{stem}.jpg"
                if not (hr.exists() and rgb.exists()):
                    continue  # ARKitScenes has unmatched frames; skip silently, count below
                records.append(
                    {
                        "id": f"{vid.name}/{stem}",
                        "video": vid.name,
                        "rgb": rgb,
                        "hr": hr,
                        "lr": lr,
                        "conf": vid / "confidence" / f"{stem}.png",
                    }
                )
        if not records and self.strict:
            raise FileNotFoundError(
                f"[arkitscenes] no matched frames under {base}. Expected "
                "<root>/upsampling/<Training|Validation>/<video_id>/{wide,lowres_depth,"
                "highres_depth}/ -- see docs/datasets.md."
            )
        return records

    def _load_raw(self, record: Mapping[str, Any]) -> Mapping[str, Any]:
        from PIL import Image

        rgb = np.asarray(Image.open(record["rgb"]).convert("RGB"), dtype=np.float32) / 255.0
        scale = self.info.depth_unit_per_metre
        hr = np.asarray(Image.open(record["hr"]), dtype=np.float32) / scale
        lr = np.asarray(Image.open(record["lr"]), dtype=np.float32) / scale

        if rgb.shape[:2] != hr.shape[:2]:
            from utils.misc import resize_rgb

            rgb = resize_rgb(rgb, hr.shape[:2])

        meta: dict[str, Any] = {"video": record["video"]}
        conf_path = record["conf"]
        if self.min_confidence > 0 and conf_path.exists():
            conf = np.asarray(Image.open(conf_path), dtype=np.uint8)
            lr = np.where(conf >= self.min_confidence, lr, 0.0).astype(np.float32)
            meta["min_confidence"] = self.min_confidence

        return {
            "rgb": rgb,
            "gt_depth": hr,
            "lr_depth": lr,
            "sample_id": record["id"],
            "rgb_path": str(record["rgb"]),
            "depth_path": str(record["hr"]),
            "meta": meta,
        }
