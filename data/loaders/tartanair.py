"""TartanAir -- large-scale photorealistic sequences (indoor + outdoor).

STATUS: stub.  OWNER: Maxim.

Why it matters: volume and motion diversity for pre-training, and it contains
stairwell environments, which is our target domain.  Depth is perfect, so it
pairs with the Astra 2 degradation model for synthetic supervision.

Gotchas:

* Depth is stored as ``*_depth.npy`` in **metres** but with ``inf`` for sky in
  outdoor scenes -- clamp to ``max_depth`` and mask, do not let ``inf`` through.
* The published depth is planar Z already; do not apply the Hypersim conversion.
* Environments are huge; index lazily and honour ``frame_stride``.
* Pick the stair-bearing environments first (e.g. ``office``, ``hospital``,
  ``japanesealley``) and record the choice in data/splits/tartanair/.
"""

from __future__ import annotations

from data.loaders._stub import StubLoaderMixin
from data.loaders.base import BaseDepthDataset, DatasetInfo, register_dataset

__all__ = ["TartanAirDataset"]


@register_dataset("tartanair")
class TartanAirDataset(StubLoaderMixin, BaseDepthDataset):
    info = DatasetInfo(
        name="tartanair",
        modality="gt_only",
        min_depth=1e-3,
        max_depth=80.0,
        native_size=(480, 640),
        sensor="rendered (AirSim)",
        url="https://theairlab.org/tartanair-dataset/",
        notes="inf depth for sky; select stair-bearing environments.",
    )
    todo = (
        "Index <root>/<env>/<Easy|Hard>/<PXXX>/image_left/*.png with the matching "
        "depth_left/*_left_depth.npy.",
        "Replace inf/NaN with 0 and rely on the base class mask.",
        "Add a stair-scene subset list to data/splits/tartanair/ and commit it.",
    )
