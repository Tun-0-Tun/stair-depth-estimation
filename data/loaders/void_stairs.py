"""VOID -- sparse VIO points + dense ground truth, stair sequences only.

The closest public analogue of our task: a *real* sparse depth input (points
tracked by a visual-inertial odometry system, not a simulation) with dense
ground truth, on real staircases.  The ``stairs*`` sequences are the reason we
carry this dataset at all.

Layout (unchanged from the official release)::

    <root>/data/<sequence>/
        image/<timestamp>.png          # 640x480 RGB
        sparse_depth/<timestamp>.png   # uint16, ~1500 points, 0 elsewhere
        ground_truth/<timestamp>.png   # uint16, dense
        validity_map/<timestamp>.png   # uint16, 256 where sparse_depth is valid
        absolute_pose/<timestamp>.txt
        K.txt                          # 3x3 intrinsics, one row per line

Depth encoding is **uint16 / 256.0 -> metres** (VOID's own
``src/data_utils.py::load_depth``), not the millimetres most datasets use.  Get
that wrong and every depth is off by a factor of 3.9.

``root`` points at a release directory (``void_150``, ``void_500``,
``void_1500``); the number is the sparse point count.  ``void_1500`` is the
default and the densest.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from data.loaders.base import BaseDepthDataset, DatasetInfo, register_dataset

__all__ = ["VOIDStairsDataset"]

#: VOID stores depth as uint16 scaled by 256, not by 1000.
VOID_DEPTH_SCALE = 256.0


@register_dataset("void_stairs")
class VOIDStairsDataset(BaseDepthDataset):
    info = DatasetInfo(
        name="void_stairs",
        modality="sparse",
        min_depth=0.2,
        max_depth=10.0,
        depth_unit_per_metre=VOID_DEPTH_SCALE,
        native_size=(480, 640),
        sensor="Intel RealSense D435i: XIVO sparse VIO points + dense reference",
        url="https://github.com/alexklwong/void-dataset",
        notes="Sequences named stairs*. Depth is uint16/256 metres.",
    )

    def __init__(
        self,
        root: str | Path = "void_1500",
        split: str = "test",
        sequences: Sequence[str] | None = None,
        sequence_glob: str = "stairs*",
        frame_stride: int = 1,
        **kwargs: Any,
    ) -> None:
        self.sequences = list(sequences) if sequences else None
        self.sequence_glob = sequence_glob
        self.frame_stride = max(1, int(frame_stride))
        super().__init__(root=root, split=split, **kwargs)

    def _build_index(self) -> Sequence[Any]:
        base = self.root / "data" if (self.root / "data").is_dir() else self.root
        if self.sequences:
            seq_dirs = [base / s for s in self.sequences]
        else:
            seq_dirs = sorted(p for p in base.glob(self.sequence_glob) if (p / "image").is_dir())

        records: list[dict[str, Any]] = []
        for seq in seq_dirs:
            for img in sorted((seq / "image").glob("*.png"))[:: self.frame_stride]:
                gt = seq / "ground_truth" / img.name
                sp = seq / "sparse_depth" / img.name
                if not (gt.exists() and sp.exists()):
                    continue
                records.append(
                    {
                        "id": f"{seq.name}/{img.stem}",
                        "seq": seq.name,
                        "image": img,
                        "gt": gt,
                        "sparse": sp,
                        "validity": seq / "validity_map" / img.name,
                        "K": seq / "K.txt",
                    }
                )

        if not records and self.strict:
            found = sorted(p.name for p in base.glob("*") if p.is_dir())[:20]
            raise FileNotFoundError(
                f"[void_stairs] no frames under {base} matching {self.sequence_glob!r}.\n"
                f"Sequences present: {found}\n"
                "Expected <root>/data/<sequence>/{image,sparse_depth,ground_truth}/ -- "
                "see docs/datasets.md."
            )
        return records

    def _load_raw(self, record: Mapping[str, Any]) -> Mapping[str, Any]:
        from PIL import Image

        rgb = np.asarray(Image.open(record["image"]).convert("RGB"), dtype=np.float32) / 255.0
        gt = np.asarray(Image.open(record["gt"]), dtype=np.float32) / VOID_DEPTH_SCALE
        sparse = np.asarray(Image.open(record["sparse"]), dtype=np.float32) / VOID_DEPTH_SCALE

        meta: dict[str, Any] = {"sequence": record["seq"]}
        if record["K"].exists():
            k = np.loadtxt(record["K"])
            meta["intrinsics"] = [float(k[0, 0]), float(k[1, 1]), float(k[0, 2]), float(k[1, 2])]

        return {
            "rgb": rgb,
            "gt_depth": gt,
            "sparse_depth": sparse,
            "sample_id": record["id"],
            "rgb_path": str(record["image"]),
            "depth_path": str(record["gt"]),
            "meta": meta,
        }
