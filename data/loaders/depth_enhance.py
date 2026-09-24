"""Lu and Middlebury -- the two classic guided depth-SR test sets.

Both ship in one archive, ``Depth_Enh.zip``, released with Lu et al., "Depth
Enhancement via Low-Rank Matrix Completion" (CVPR 2014):
https://web.cecs.pdx.edu/~fliu/project/depth-enhance/

    Depth_Enh/01_Middlebury_Dataset   30 pairs -> our  Middlebury/
    Depth_Enh/02_RGBZ_Dataset          9 pairs -> unused, not part of the benchmark
    Depth_Enh/03_RGBD_Dataset          6 pairs -> our  Lu/

The two directories we keep have the same flat format, so one loader reads
both; they differ only by ``root``, and the dataset name in ``meta`` is taken
from that directory.  Each scene ships three variants (``clean_*``, ``noisy_*``,
``output_*``); the DSR literature -- and DuCos's own dataloader,
``third_party/ducos/data/middlebury_dataloader.py`` -- uses ``output_*`` only.

.. warning::
   **Depth here is not metric.**  It is an 8-bit PNG that upstream divides by
   255, i.e. a normalised quantity with no physical scale.  Every other loader
   in this repo returns metres (rule 4 in CLAUDE.md); this one cannot, because
   the data has no metres in it.  ``meta["depth_unit"]`` says so on every
   sample.  An RMSE from these two sets lives on the [0, 1] scale -- DuCos
   multiplies it by 100 and calls it "centimetres" -- and is **not comparable**
   with a metric RMSE from VOID, ZJU-L5 or NYUv2.  Label it in every table.

.. note::
   Upstream crops both images to a multiple of 16 before anything else
   (``modcrop``) and evaluates on the cropped frame, so we reproduce it: without
   it we would score a slightly different region than every published number for
   these sets.  ``mod_crop: 0`` turns it off.

Filename trap: in ``03_RGBD_Dataset`` the RGB files are misspelled
``ouput_color`` (sic) upstream while the depth files are spelled correctly.  Fix
that when laying the data out (see ``docs/datasets.md``) rather than teaching
the loader the typo -- DuCos hard-codes ``output_color`` too, and would find
zero RGB frames in the archive as shipped.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np

from data.loaders.base import BaseDepthDataset, DatasetInfo, register_dataset

__all__ = ["DepthEnhanceDataset"]


# One class, two datasets: registered under both names so that every key in the
# registry has a configs/dataset/<key>.yaml, which the suite checks.
@register_dataset("lu")
@register_dataset("middlebury")
class DepthEnhanceDataset(BaseDepthDataset):
    info = DatasetInfo(
        name="depth_enhance",
        modality="gt_only",
        min_depth=0.0,
        max_depth=1.01,  # depth is [0, 1]; 1.01 so a saturated pixel stays in the mask
        depth_unit_per_metre=255.0,
        sensor="Middlebury structured light / ASUS Xtion Pro (Lu)",
        url="https://web.cecs.pdx.edu/~fliu/project/depth-enhance/",
        notes="GT only, and NOT metric: depth is uint8/255, a normalised scale.",
    )

    def __init__(
        self,
        root: str | Path,
        split: str = "test",
        rgb_token: str = "output_color",
        depth_token: str = "output_depth",
        mod_crop: int = 16,
        depth_scale: float = 255.0,
        **kwargs: Any,
    ) -> None:
        self.rgb_token = rgb_token
        self.depth_token = depth_token
        self.mod_crop = int(mod_crop)
        self.depth_scale = float(depth_scale)
        # One class, two datasets: name the samples after the directory so the
        # results table says "middlebury" / "lu" rather than "depth_enhance".
        self.info = replace(type(self).info, name=Path(root).name.lower())
        super().__init__(root=root, split=split, **kwargs)

    # ------------------------------------------------------------------

    def _build_index(self) -> Sequence[Any]:
        rgbs = sorted(self.root.glob(f"*{self.rgb_token}*"))
        deps = sorted(self.root.glob(f"*{self.depth_token}*"))

        if not deps and self.strict:
            raise FileNotFoundError(
                f"[{self.info.name}] no '*{self.depth_token}*' files under {self.root}. "
                "Expected a flat directory of <scene>_output_color.png / "
                "<scene>_output_depth.png -- see docs/datasets.md."
            )
        if len(rgbs) != len(deps):
            raise FileNotFoundError(
                f"[{self.info.name}] {len(rgbs)} '*{self.rgb_token}*' files but {len(deps)} "
                f"'*{self.depth_token}*' under {self.root}. In 03_RGBD_Dataset upstream "
                "misspells the RGB token as 'ouput_color'; rename on copy, see docs/datasets.md."
            )

        records: list[dict[str, Any]] = []
        for rgb, dep in zip(rgbs, deps, strict=True):
            # The two lists are sorted independently, exactly as upstream does it.
            # That is only correct while the names agree outside the token, so
            # check it here rather than silently pairing depth with a foreign RGB.
            stem_rgb = rgb.name.replace(self.rgb_token, "")
            stem_dep = dep.name.replace(self.depth_token, "")
            if stem_rgb.lower() != stem_dep.lower():
                raise FileNotFoundError(
                    f"[{self.info.name}] {rgb.name} and {dep.name} do not pair up "
                    f"({stem_rgb!r} vs {stem_dep!r}). Sorted-order pairing is unsafe here; "
                    "fix the file names."
                )
            records.append({"id": stem_dep.rstrip("_."), "rgb": rgb, "depth": dep})
        return records

    def _mod_crop(self, arr: np.ndarray) -> np.ndarray:
        if self.mod_crop <= 1:
            return arr
        h = arr.shape[0] - arr.shape[0] % self.mod_crop
        w = arr.shape[1] - arr.shape[1] % self.mod_crop
        return arr[:h, :w]

    def _load_raw(self, record: Mapping[str, Any]) -> Mapping[str, Any]:
        from PIL import Image

        rgb = np.asarray(Image.open(record["rgb"]).convert("RGB"), dtype=np.float32) / 255.0
        dep = np.asarray(Image.open(record["depth"]), dtype=np.float32)
        if dep.ndim == 3:  # some frames are stored as grey-in-RGB
            dep = dep[..., 0]

        if dep.shape != rgb.shape[:2]:
            raise ValueError(
                f"[{self.info.name}] {record['id']}: depth {dep.shape} does not match "
                f"rgb {rgb.shape[:2]}"
            )

        rgb = self._mod_crop(rgb)
        dep = self._mod_crop(dep) / self.depth_scale

        return {
            "rgb": np.ascontiguousarray(rgb),
            "gt_depth": np.ascontiguousarray(dep),
            "sample_id": record["id"],
            "rgb_path": str(record["rgb"]),
            "depth_path": str(record["depth"]),
            "meta": {"depth_unit": "normalised [0,1] (uint8/255) -- NOT metres"},
        }
