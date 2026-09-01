"""Procedurally generated staircase scenes -- no download, fully deterministic.

This is **not** a benchmark dataset and no reported number may come from it.
It exists so that the whole pipeline (loader -> degradation -> adapter ->
metrics -> results table) can be exercised in CI and on a laptop with no data
on disk, and so that a new degradation model or adapter can be debugged in
seconds instead of after a 40 GB download.

The scene is a camera looking at a flight of stairs: a ground plane, a back
wall, and ``n_steps`` risers/treads.  That geometry is deliberately the hard
case for this project -- many near-horizontal depth discontinuities, which is
where dToF zone mixing and active-stereo occlusion shadows do their damage.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from data.loaders.base import BaseDepthDataset, DatasetInfo, register_dataset

__all__ = ["SyntheticStairsDataset"]


@register_dataset("synthetic_stairs")
class SyntheticStairsDataset(BaseDepthDataset):
    """Deterministic synthetic stairs, for smoke tests and pipeline debugging."""

    info = DatasetInfo(
        name="synthetic_stairs",
        modality="gt_only",
        min_depth=0.3,
        max_depth=8.0,
        native_size=(480, 640),
        sensor="none (procedural)",
        notes="CI-only. Never report metrics from this dataset.",
    )

    def __init__(
        self,
        root: str | Path = ".",
        split: str = "test",
        n_samples: int = 8,
        height: int = 480,
        width: int = 640,
        n_steps: int = 6,
        **kwargs: Any,
    ) -> None:
        self.n_samples = int(n_samples)
        self.height = int(height)
        self.width = int(width)
        self.n_steps = int(n_steps)
        kwargs.setdefault("strict", False)
        super().__init__(root=root, split=split, **kwargs)

    def _build_index(self) -> Sequence[Any]:
        return [{"id": f"synth_{i:04d}", "index": i} for i in range(self.n_samples)]

    def _load_raw(self, record: Mapping[str, Any]) -> Mapping[str, Any]:
        rng = np.random.default_rng(1000 + int(record["index"]))
        h, w = self.height, self.width

        # camera looks slightly down at a staircase starting `d0` metres away
        d0 = 1.0 + 0.25 * rng.random()
        step_depth = 0.28  # tread depth, metres
        step_rise = 0.17  # riser height, metres

        yy = np.linspace(1.0, 0.0, h)[:, None].repeat(w, axis=1)  # 1 at bottom of frame
        xx = np.linspace(-0.5, 0.5, w)[None, :].repeat(h, axis=0)

        # each horizontal band of the image is one step -> piecewise-constant depth
        band = np.clip((yy * self.n_steps).astype(int), 0, self.n_steps - 1)
        depth = d0 + band * step_depth
        # the riser face inside each band adds a small linear ramp
        frac = yy * self.n_steps - band
        depth = depth + frac * step_rise * 0.4
        # perspective: things off-axis are slightly farther
        depth = depth * (1.0 + 0.12 * xx**2)
        # a back wall caps the scene
        depth = np.minimum(depth, d0 + self.n_steps * step_depth + 0.6)

        gt = depth.astype(np.float32)

        # RGB: shading from the depth structure plus a bit of texture, so an
        # RGB-guided method has something real to key on at the discontinuities.
        edges = np.zeros_like(gt)
        edges[1:, :] = np.abs(np.diff(gt, axis=0))
        shade = 1.0 - np.clip(edges * 6.0, 0, 1)
        base = np.clip(1.2 - (gt - gt.min()) / (np.ptp(gt) + 1e-6), 0.1, 1.0)
        texture = 0.06 * rng.random((h, w))
        lum = np.clip(base * shade + texture, 0.0, 1.0)
        rgb = np.stack([lum * 0.95, lum, lum * 0.88], axis=-1).astype(np.float32)

        return {
            "rgb": rgb,
            "gt_depth": gt,
            "sample_id": record["id"],
            "meta": {"scene": "procedural_stairs", "n_steps": self.n_steps},
        }
