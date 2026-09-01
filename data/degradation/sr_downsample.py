"""Synthetic depth super-resolution degradation (the DSR literature protocol).

Every depth-SR paper (DuCos included) evaluates on GT depth bicubically
downsampled by a factor of 4 / 8 / 16.  It is an unrealistic degradation --
a real ToF does not bicubically resample the world -- but we must reproduce it
exactly to be able to compare our numbers against published tables.

Two variants matter:

* ``noisy=False``: the classic clean protocol (DuCos ``--isNosiy 0``).
* ``noisy=True``: additive Gaussian noise on the low-resolution depth, the
  "real" protocol used for RGB-D-D / TOFDSR robustness ablations.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np

from data.degradation.base import BaseDegradation, register_degradation
from utils.misc import resize_depth

__all__ = ["BicubicSRDegradation"]


@register_degradation("bicubic_sr")
@dataclass
class BicubicSRDegradation(BaseDegradation):
    """Downsample GT depth by ``scale`` with bicubic interpolation."""

    key: str = "bicubic_sr"

    scale: int = 4
    mode: str = "bicubic"
    noisy: bool = False
    noise_sigma_m: float = 0.01
    """Std-dev of the additive noise on the LR depth, in metres (if ``noisy``)."""
    blur_sigma_px: float = 0.0
    """Optional pre-blur of the HR depth, mimicking the sensor PSF."""

    def apply(self, gt: np.ndarray, rng: np.random.Generator) -> Mapping[str, Any]:
        z = gt.astype(np.float32)
        if self.blur_sigma_px > 0:
            from scipy import ndimage

            z = ndimage.gaussian_filter(z, self.blur_sigma_px)

        h, w = z.shape
        s = int(self.scale)
        lr = resize_depth(z, (max(1, h // s), max(1, w // s)), mode=self.mode)
        if self.noisy and self.noise_sigma_m > 0:
            lr = lr + rng.normal(0.0, self.noise_sigma_m, size=lr.shape).astype(np.float32)
        lr = np.clip(lr, 0.0, None).astype(np.float32)

        # The completion-style view of an SR input is the LR map put back on the
        # full grid without inventing detail -> nearest upsampling.
        sparse = resize_depth(lr, (h, w), mode="nearest")
        return {
            "sparse_depth": sparse.astype(np.float32),
            "lr_depth": lr,
            "meta": {"scale": s},
        }


@register_degradation("random_sparse")
@dataclass
class RandomSparseDegradation(BaseDegradation):
    """Uniformly random sparse samples, the classic NYU-DC protocol (500 points).

    Kept because several completion papers report it, and it is the cheapest
    sanity check that a completion adapter is wired up correctly: with 500 exact
    samples and no noise a working model must beat a plain interpolation.
    """

    key: str = "random_sparse"

    n_samples: int = 500
    noise_sigma_m: float = 0.0

    def apply(self, gt: np.ndarray, rng: np.random.Generator) -> Mapping[str, Any]:
        z = gt.astype(np.float32)
        valid = np.flatnonzero((z > 0) & np.isfinite(z))
        sparse = np.zeros_like(z)
        if valid.size:
            n = min(int(self.n_samples), valid.size)
            idx = rng.choice(valid, size=n, replace=False)
            vals = z.flat[idx]
            if self.noise_sigma_m > 0:
                vals = vals + rng.normal(0.0, self.noise_sigma_m, size=vals.shape)
            sparse.flat[idx] = np.clip(vals, 0.0, None)
        return {"sparse_depth": sparse, "lr_depth": None, "meta": {"n_samples": self.n_samples}}
