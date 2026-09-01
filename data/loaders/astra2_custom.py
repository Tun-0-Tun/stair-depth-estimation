"""Our own Orbbec Astra 2 stair captures.

STATUS: stub.  OWNER: Olesya (capture) + Maxim (loader).

This is the dataset the SOW is actually judged on, so its loader is the one that
must be most careful:

* **Reference depth.**  An Astra 2 frame cannot be its own ground truth.
  Decide and document the reference: multi-view stereo reconstruction, a laser
  scan, or a temporally averaged + hole-filled Astra 2 frame (weakest option,
  and it makes RMSE optimistic).  Whatever it is, ``meta["gt_source"]`` must
  say so on every sample, and docs/datasets.md must describe the procedure.
* **Alignment.**  RGB and depth come from different lenses; use the factory
  extrinsics from the Orbbec SDK and store the intrinsics per sample in
  ``meta`` -- the degradation model needs ``focal_px`` and ``baseline_m``.
* **Splits.**  Split by *capture session*, never by frame: consecutive frames of
  the same staircase are near-duplicates and a random frame split leaks the test
  set into training.
* **Raw data stays out of git.**  ``data/raw/astra2/`` is git-ignored; only the
  split json belongs in the repo.
"""

from __future__ import annotations

from data.loaders._stub import StubLoaderMixin
from data.loaders.base import BaseDepthDataset, DatasetInfo, register_dataset

__all__ = ["Astra2CustomDataset"]


@register_dataset("astra2_custom")
class Astra2CustomDataset(StubLoaderMixin, BaseDepthDataset):
    info = DatasetInfo(
        name="astra2_custom",
        modality="sparse",
        min_depth=0.6,
        max_depth=8.0,
        depth_unit_per_metre=1000.0,
        native_size=(480, 640),
        sensor="Orbbec Astra 2 (active stereo), uint16 mm depth",
        url="internal",
        notes="Reference depth source must be recorded per sample in meta['gt_source'].",
    )
    todo = (
        "Freeze the on-disk layout: <root>/<session>/{rgb,depth,gt}/<frame>.png + meta.json.",
        "Record intrinsics (fx, fy, cx, cy) and the IR baseline per session in meta.",
        "Decide and document the reference-depth procedure; set meta['gt_source'].",
        "Split by session into data/splits/astra2_custom/{train,val,test}.json and commit.",
    )
