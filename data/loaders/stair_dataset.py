"""RGB-D Stair Dataset -- public staircase scenes.

STATUS: stub.  OWNER: Olesya.

Why it matters: the only *public* stair-specific RGB-D data we found.  It is the
bridge between the generic indoor benchmarks and our own Astra 2 captures, and
the natural place to show that a generic method degrades on stairs while ours
does not.

Gotchas:

* Check first whether the release ships metric depth or only disparity /
  8-bit visualisations; if the depth is 8-bit colourised, it is unusable as GT
  and the dataset can only serve as an RGB-side qualitative set. Record the
  finding in docs/datasets.md.
* Verify RGB/depth alignment before trusting it -- several stair datasets ship
  unregistered streams.
"""

from __future__ import annotations

from data.loaders._stub import StubLoaderMixin
from data.loaders.base import BaseDepthDataset, DatasetInfo, register_dataset

__all__ = ["RGBDStairDataset"]


@register_dataset("stair_dataset")
class RGBDStairDataset(StubLoaderMixin, BaseDepthDataset):
    info = DatasetInfo(
        name="rgbd_stair",
        modality="sparse",
        min_depth=0.3,
        max_depth=8.0,
        sensor="consumer RGB-D (see docs/datasets.md)",
        url="",
        notes="Confirm depth units and RGB/depth registration before use.",
    )
    todo = (
        "Confirm the download URL and licence, then fill DatasetInfo.url.",
        "Determine the depth encoding (uint16 mm? float m? disparity?) and document it.",
        "Verify RGB/depth alignment on 5 frames by hand before writing the loader.",
        "Freeze a test split into data/splits/rgbd_stair/test.json.",
    )
