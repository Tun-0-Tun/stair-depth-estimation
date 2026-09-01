"""HAMMER -- Highly Accurate Multi-Modal Dataset for dense 3D scene Regression.

Why we care: HAMMER is the only public dataset that ships an **active stereo**
depth stream (RealSense D435) *and* a dToF stream (L515) *and* laser-accurate
ground truth for the same frames.  That makes it the one place where we can
measure the domain gap between the sensor DEPTHOR was built for (dToF) and the
sensor we must ship on (active stereo, Astra 2) without confounding it with a
change of scene.  DEPTHOR++ also reports on it, so it is a cross-check dataset.

Download: a single ~50 GB zip, no registration --
``http://www.campar.in.tum.de/public_datasets/2022_arxiv_jung/_dataset_processed.zip``
(see ``docs/datasets.md``).

.. warning::
   The exact folder/file naming inside the archive is **not documented** in the
   authors' README, so the patterns below are configurable rather than
   hard-coded.  After unpacking, run::

       python scripts/download_data.py --dataset hammer --inspect

   which prints the real tree, then set ``rgb_glob`` / ``gt_subdir`` /
   ``input_subdir`` in ``configs/dataset/hammer.yaml`` to match.  The loader
   raises a message naming the directories it actually found rather than
   silently returning zero samples.

Modalities (``input_sensor``):
``d435`` active stereo (our analogue of the Astra 2), ``l515`` dToF,
``tof`` Lucid Helios I-ToF.  All are provided already warped into the RGB frame
by the dataset authors.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from data.loaders.base import BaseDepthDataset, DatasetInfo, register_dataset

__all__ = ["HammerDataset"]

_TEST_SCENES = ("scene12", "scene13", "scene14")


@register_dataset("hammer")
class HammerDataset(BaseDepthDataset):
    info = DatasetInfo(
        name="hammer",
        modality="sparse",
        min_depth=0.1,
        max_depth=5.0,
        depth_unit_per_metre=1000.0,
        sensor="RealSense D435 (active stereo) / L515 (dToF) / Lucid Helios (I-ToF); laser GT",
        url="https://github.com/Junggy/HAMMER-dataset",
        notes="Scenes 2-11 train, 12-14 test. Depth PNGs are uint16 millimetres.",
    )

    def __init__(
        self,
        root: str | Path,
        split: str = "test",
        input_sensor: str = "d435",
        rgb_subdir: str = "rgb",
        gt_subdir: str = "gt",
        input_subdir: str | None = None,
        rgb_glob: str = "*.png",
        depth_unit_per_metre: float = 1000.0,
        scenes: Sequence[str] | None = None,
        **kwargs: Any,
    ) -> None:
        self.input_sensor = input_sensor
        self.rgb_subdir = rgb_subdir
        self.gt_subdir = gt_subdir
        self.input_subdir = input_subdir or input_sensor
        self.rgb_glob = rgb_glob
        self.depth_scale = float(depth_unit_per_metre)
        self.scenes = list(scenes) if scenes else None
        super().__init__(root=root, split=split, **kwargs)

    def _wanted_scene(self, seq_dir: Path) -> bool:
        name = seq_dir.name
        if self.scenes:
            return any(name.startswith(s) for s in self.scenes)
        is_test = any(name.startswith(s) for s in _TEST_SCENES)
        return is_test if self.split in ("test", "val") else not is_test

    def _build_index(self) -> Sequence[Any]:
        seq_dirs = sorted(p for p in self.root.rglob("*") if (p / self.rgb_subdir).is_dir())
        seq_dirs = [p for p in seq_dirs if self._wanted_scene(p)]

        records: list[dict[str, Any]] = []
        for seq in seq_dirs:
            for rgb in sorted((seq / self.rgb_subdir).glob(self.rgb_glob)):
                gt = seq / self.gt_subdir / rgb.name
                inp = seq / self.input_subdir / rgb.name
                if not gt.exists():
                    continue
                records.append(
                    {
                        "id": f"{seq.name}/{rgb.stem}",
                        "seq": seq.name,
                        "rgb": rgb,
                        "gt": gt,
                        "input": inp if inp.exists() else None,
                    }
                )

        if not records and self.strict:
            found = sorted({p.name for p in self.root.rglob("*") if p.is_dir()})[:25]
            raise FileNotFoundError(
                f"[hammer] no samples under {self.root} with rgb_subdir={self.rgb_subdir!r}, "
                f"gt_subdir={self.gt_subdir!r}, input_subdir={self.input_subdir!r}.\n"
                f"Directory names actually present: {found}\n"
                "Adjust configs/dataset/hammer.yaml to the real layout "
                "(the archive's structure is not documented upstream)."
            )
        return records

    def _load_raw(self, record: Mapping[str, Any]) -> Mapping[str, Any]:
        from PIL import Image

        rgb = np.asarray(Image.open(record["rgb"]).convert("RGB"), dtype=np.float32) / 255.0
        gt = np.asarray(Image.open(record["gt"]), dtype=np.float32) / self.depth_scale

        out: dict[str, Any] = {
            "rgb": rgb,
            "gt_depth": gt,
            "sample_id": record["id"],
            "rgb_path": str(record["rgb"]),
            "depth_path": str(record["gt"]),
            "meta": {"sequence": record["seq"], "input_sensor": self.input_sensor},
        }
        if record["input"] is not None:
            sparse = np.asarray(Image.open(record["input"]), dtype=np.float32) / self.depth_scale
            out["sparse_depth"] = sparse
        elif self.degradation is None and self.strict:
            raise FileNotFoundError(
                f"[hammer] no {self.input_sensor} frame for {record['id']} and no degradation "
                "model configured. Either fix input_subdir or attach a degradation."
            )
        return out
