"""Hypersim -- photorealistic synthetic indoor scenes with perfect depth.

STATUS: stub.  OWNER: Maxim.

Why it matters: this is DEPTHOR's *training* set.  Reproducing their training
run, and later training our own method, both go through this loader.  It is
also the cleanest source of GT for fitting the Astra 2 degradation model,
because there is no sensor noise to disentangle.

Gotchas the implementer must handle:

* Hypersim ships **distance to the camera centre** (``depth_meters`` is really
  a ray distance), not planar Z.  Convert with
  ``z = dist / sqrt(1 + ((u - cx)/f)^2 + ((v - cy)/f)^2)`` or every metric is
  systematically wrong towards the image corners.
* Depth is stored in ``*.hdf5`` under ``dataset`` in the
  ``_detail/.../depth_meters.hdf5`` style tree; tone-mapped RGB lives in
  ``images/scene_cam_XX_final_preview/frame.NNNN.tonemap.jpg``.
* Some frames are known-bad; ``third_party/depthor/assets/hypersim_train.txt``
  and ``hypersim_val.txt`` are the filtered lists DEPTHOR used -- start from
  those so our training split matches theirs, and freeze them into
  ``data/splits/hypersim/``.
"""

from __future__ import annotations

from data.loaders._stub import StubLoaderMixin
from data.loaders.base import BaseDepthDataset, DatasetInfo, register_dataset

__all__ = ["HypersimDataset"]


@register_dataset("hypersim")
class HypersimDataset(StubLoaderMixin, BaseDepthDataset):
    info = DatasetInfo(
        name="hypersim",
        modality="gt_only",
        min_depth=1e-3,
        max_depth=20.0,
        native_size=(768, 1024),
        sensor="rendered (Evermotion scenes)",
        url="https://github.com/apple/ml-hypersim",
        notes="Ray distance, not planar Z. DEPTHOR's training set.",
    )
    todo = (
        "Point root at <root>/scenes/ai_XXX_XXX/ and index frames from the "
        "DEPTHOR split lists in third_party/depthor/assets/.",
        "Read tonemapped RGB (jpg) and depth_meters.hdf5 per frame.",
        "Convert ray distance -> planar Z using the per-scene intrinsics.",
        "Copy the split lists into data/splits/hypersim/{train,val}.json and commit them.",
    )
