"""Dependency-free reference baselines.

Every depth-SR and depth-completion paper reports "Bicubic" (and often
"nearest-neighbour fill") as the floor.  We need them for three reasons:

1. **A floor for the table.**  A learned method that does not beat bicubic on a
   dataset is broken or mis-configured, and that has to be visible.
2. **Pipeline verification without weights.**  ``run_baseline.py`` and the whole
   benchmark path can be exercised end-to-end on any machine, with no
   checkpoints and no CUDA.  If the numbers move, the harness changed.
3. **Runtime reference.**  Their FPS is the cost of the plumbing (I/O,
   resizing, metric computation), so a learned method's FPS can be read as
   "network + plumbing" against a known plumbing cost.

These are *not* our method.  Our method lives in ``models/``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from baselines.base import BaselineModel, register_baseline
from utils.misc import resize_depth, sparse_to_dense_nn

__all__ = ["BicubicBaseline", "NearestFillBaseline"]


@register_baseline("bicubic")
class BicubicBaseline(BaselineModel):
    """Bicubic upsampling of the low-resolution depth.  The standard DSR floor."""

    input_modality = "lr"
    paper = "reference baseline (no learning)"

    def _load(self, checkpoint_path: Path | None) -> Any:
        return "bicubic"  # nothing to load; a non-None marker so predict() is allowed

    def _predict(self, rgb: np.ndarray, depth_in: np.ndarray, **kwargs: Any) -> np.ndarray:
        h, w = rgb.shape[:2]
        out = resize_depth(depth_in, (h, w), mode="bicubic")
        return np.clip(out, self.min_depth, self.max_depth).astype(np.float32)


@register_baseline("nn_fill")
class NearestFillBaseline(BaselineModel):
    """Fill every hole in the sparse map with its nearest measurement.

    The completion floor: it uses no image guidance at all, so anything an
    RGB-guided method gains over it is the value of the guidance.
    """

    input_modality = "sparse"
    paper = "reference baseline (no learning)"

    def __init__(self, smooth_sigma_px: float = 0.0, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.smooth_sigma_px = float(smooth_sigma_px)

    def _load(self, checkpoint_path: Path | None) -> Any:
        return "nn_fill"

    def _predict(self, rgb: np.ndarray, depth_in: np.ndarray, **kwargs: Any) -> np.ndarray:
        out = sparse_to_dense_nn(depth_in)
        if not out.any():  # nothing measured at all -> mid-range constant
            out = np.full_like(out, 0.5 * (self.min_depth + self.max_depth))
        if self.smooth_sigma_px > 0:
            from scipy import ndimage

            out = ndimage.gaussian_filter(out, self.smooth_sigma_px)
        return np.clip(out, self.min_depth, self.max_depth).astype(np.float32)
