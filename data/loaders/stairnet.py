"""The StairNet stair datasets (Wang et al., Sensors 2023 / Visual Computer 2024).

Two folders on the shared drive, one family of files:

``RGB-D_stair_dataset/RGB-D stair dataset/``   -> ``rgbd_stair`` (test split only)
    test/images/color_<n>.png             640x480 RGB
    test/depthes/Depth_<n>.png            480x640 uint8, depth **normalised per frame**
    test/extrinsicses/Extrinsics_<n>.txt  one line: ``dmin_mm dmax_mm gx gy gz``
    test/{plys,pcds}/                     point clouds, unused
    {train,val}/                          512x512 crops, same as below + segmentations

``Stair_dataset_with_depth_maps/data/``      -> ``stairnet_rel``
    {train,val}/images/<name>.jpg         512x512 RGB
    {train,val}/depthes/<name>.png        512x512 uint8 (grey in RGB), no range stored
    {train,val}/labels/<name>.txt         stair edge lines, ``cls x1 y1 x2 y2``

.. warning::
   **Only ``rgbd_stair`` is metric.**  Its 8-bit depth is decoded with the
   per-frame range from the extrinsics file: ``d = dmin + v/255 * (dmax - dmin)``
   mm, so the quantisation step is ``(dmax - dmin)/255`` -- about 18 mm at the
   median frame.  The last three numbers of that file are a gravity vector
   (|g| = 9.0-9.8 on all 154 frames), which is how we know the first two are
   not rotation.  A few frames carry a stray far pixel (dmax up to 65535 mm),
   which makes the step useless; ``max_range_mm`` drops them.  The camera is
   the GT, as on MinJiang, so numbers are optimistic.

   ``stairnet_rel`` has no range anywhere: depth is uint8/255, a normalised
   scale, exactly like Lu/Middlebury.  Its RMSE is not in metres -- use it for
   SiLog and for stair-edge error analysis via ``labels``, never in a metric
   table next to VOID or ZJU-L5.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from data.loaders.base import BaseDepthDataset, DatasetInfo, register_dataset

__all__ = ["RGBDStairDataset", "StairNetRelDataset", "decode_normalised_depth"]


def decode_normalised_depth(v: np.ndarray, dmin_mm: float, dmax_mm: float) -> np.ndarray:
    """uint8 per-frame-normalised depth -> metres; 0 stays 0 (no measurement)."""
    v = v.astype(np.float32)
    d = (dmin_mm + v / 255.0 * (dmax_mm - dmin_mm)) / 1000.0
    return np.where(v > 0, d, 0.0).astype(np.float32)


@register_dataset("rgbd_stair")
class RGBDStairDataset(BaseDepthDataset):
    info = DatasetInfo(
        name="rgbd_stair",
        modality="gt_only",
        min_depth=0.2,
        max_depth=10.0,
        native_size=(480, 640),
        sensor="RGB-D camera with IMU (StairNetV2 capture); depth quantised to 256 levels per frame",
        url="https://data.mendeley.com/datasets/p28ncjnvgk",
        notes="Test split only. Depth decoded from uint8 with the per-frame range; GT is the sensor.",
    )

    def __init__(
        self,
        root: str | Path = "RGB-D_stair_dataset/RGB-D stair dataset",
        split: str = "test",
        max_range_mm: float = 10000.0,
        **kwargs: Any,
    ) -> None:
        self.max_range_mm = float(max_range_mm)
        super().__init__(root=root, split=split, **kwargs)

    def _build_index(self) -> Sequence[Any]:
        if self.split != "test":
            raise ValueError(
                "[rgbd_stair] only split=test has a depth range; train/val are the "
                "non-metric 512x512 crops -- use dataset=stairnet_rel for those."
            )
        base = self.root / "test"
        records: list[dict[str, Any]] = []
        for rgb in sorted((base / "images").glob("color_*.png"), key=lambda p: int(p.stem[6:])):
            n = rgb.stem[6:]
            dep, ext = (
                base / "depthes" / f"Depth_{n}.png",
                base / "extrinsicses" / f"Extrinsics_{n}.txt",
            )
            if not (dep.exists() and ext.exists()):
                continue
            dmin, dmax = (float(x) for x in ext.read_text().split()[:2])
            if dmax - dmin > self.max_range_mm:
                continue  # ponytail: drops ~15/154 frames with a stray far pixel; lower max_range_mm for finer steps
            records.append({"id": n, "rgb": rgb, "depth": dep, "dmin": dmin, "dmax": dmax})

        if not records and self.strict:
            raise FileNotFoundError(
                f"[rgbd_stair] no frames under {base}. Expected test/images/color_<n>.png, "
                "test/depthes/Depth_<n>.png, test/extrinsicses/Extrinsics_<n>.txt."
            )
        return records

    def _load_raw(self, record: Mapping[str, Any]) -> Mapping[str, Any]:
        from PIL import Image

        v = np.asarray(Image.open(record["depth"]))
        if v.ndim == 3:
            v = v[..., 0]
        return {
            "rgb": np.asarray(Image.open(record["rgb"]).convert("RGB"), dtype=np.float32) / 255.0,
            "gt_depth": decode_normalised_depth(v, record["dmin"], record["dmax"]),
            "sample_id": record["id"],
            "rgb_path": str(record["rgb"]),
            "depth_path": str(record["depth"]),
            "meta": {
                "quant_step_mm": (record["dmax"] - record["dmin"]) / 255.0,
                "gt_source": "sensor, 8-bit quantised (not an independent reference)",
            },
        }


@register_dataset("stairnet_rel")
class StairNetRelDataset(BaseDepthDataset):
    info = DatasetInfo(
        name="stairnet_rel",
        modality="gt_only",
        min_depth=0.0,
        max_depth=1.01,  # [0, 1]; 1.01 keeps a saturated pixel in the mask
        depth_unit_per_metre=255.0,
        native_size=(512, 512),
        url="https://data.mendeley.com/datasets/6kffmjt7g2/1",
        notes="NOT metric: uint8/255 with no stored range. Stair edge labels in meta.",
    )

    def __init__(
        self,
        root: str | Path = "Stair_dataset_with_depth_maps/data",
        split: str = "val",
        **kwargs: Any,
    ) -> None:
        super().__init__(root=root, split=split, **kwargs)

    def _build_index(self) -> Sequence[Any]:
        base = self.root / ("val" if self.split == "test" else self.split)
        records = [
            {
                "id": img.stem,
                "rgb": img,
                "depth": base / "depthes" / f"{img.stem}.png",
                "labels": base / "labels" / f"{img.stem}.txt",
            }
            for img in sorted((base / "images").glob("*.jpg"))
            if (base / "depthes" / f"{img.stem}.png").exists()
        ]
        if not records and self.strict:
            raise FileNotFoundError(f"[stairnet_rel] no frames under {base}/images + depthes.")
        return records

    def _load_raw(self, record: Mapping[str, Any]) -> Mapping[str, Any]:
        from PIL import Image

        v = np.asarray(Image.open(record["depth"]), dtype=np.float32)
        if v.ndim == 3:
            v = v[..., 0]
        lines = (
            np.loadtxt(record["labels"], ndmin=2) if record["labels"].exists() else np.zeros((0, 5))
        )
        return {
            "rgb": np.asarray(Image.open(record["rgb"]).convert("RGB"), dtype=np.float32) / 255.0,
            "gt_depth": v / 255.0,
            "sample_id": record["id"],
            "rgb_path": str(record["rgb"]),
            "depth_path": str(record["depth"]),
            "meta": {
                "depth_unit": "normalised [0,1] (uint8/255) -- NOT metres",
                "stair_lines": lines.tolist(),
            },
        }
