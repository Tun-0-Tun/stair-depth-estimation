"""Baseline: normals from the cross product of 3-D surface tangents.

Method
------
1. Back-project every pixel to a metric 3-D point ``P = ((u-cx)/fx*z, (v-cy)/fy*z, z)``.
   Differencing raw depth gives "metres per pixel", which changes with distance and
   focal length; differencing 3-D points gives true metric tangents, so the normal
   does not depend on resolution or range.
2. Tangents by central differences, ``Pu = (P[x+1]-P[x-1])/2`` and likewise ``Pv``
   (symmetric: no half-pixel bias).
3. ``n = normalize(Pu x Pv)``, flipped to face the camera (``n_z < 0``).  On a locally
   planar patch the cross product of the two tangents is the normal -- exact on
   planes, and stairs are made of planes.
4. A pixel is valid only if its four neighbours are measured and none is across a
   depth jump (``depth_edge_rel``).  Across a nosing or occlusion edge the difference
   would span two surfaces and yield a fake "wall" normal; across a hole it would
   yield a fake cliff.  Better invalid than wrong, exactly where it matters.

RGB is accepted and ignored: this baseline is pure geometry.

Intrinsics default to ~640x480 values (``fx=570``, as in ``projector_shadow``) and are
NOT calibrated -- results that rely on them carry ``uncalibrated`` in the notes column
unless the sample provides its own intrinsics.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.ndimage import gaussian_filter

from data.norm.base import BaseNormalMap, register_normal

__all__ = ["CrossProductNormals"]


@register_normal("cross")
@dataclass
class CrossProductNormals(BaseNormalMap):
    """Normals from central-difference tangents of the back-projected depth."""

    key: str = "cross"

    fx: float = 570.0
    """Horizontal focal length in pixels at the working resolution (570 ~ 640x480)."""
    fy: float | None = None
    """Vertical focal length in pixels; ``None`` means ``fx``."""
    cx: float | None = None
    """Principal point x in pixels; ``None`` means the image centre."""
    cy: float | None = None
    """Principal point y in pixels; ``None`` means the image centre."""

    depth_edge_rel: float = 0.05
    """A neighbour whose depth differs by more than this fraction of the pixel's depth
    marks a discontinuity; the pixel is invalid."""
    smooth_sigma: float = 0.0
    """Gaussian sigma (pixels) applied to valid depth before differencing; 0 = off.
    Differentiation amplifies GT noise, this trades edge sharpness for stability."""
    eps: float = 1e-12
    """Minimum cross-product magnitude (m^2); below it the surface is degenerate."""

    def estimate(self, gt, rgb, k) -> dict[str, Any]:
        fx, fy, cx, cy = k
        h, w = gt.shape
        z = gt.astype(np.float64)
        valid = z > 0

        if self.smooth_sigma > 0:  # normalised convolution: holes do not bleed in
            num = gaussian_filter(np.where(valid, z, 0.0), self.smooth_sigma)
            den = gaussian_filter(valid.astype(np.float64), self.smooth_sigma)
            z = np.where(valid, num / np.maximum(den, 1e-8), 0.0)

        u, v = np.meshgrid(np.arange(w, dtype=np.float64), np.arange(h, dtype=np.float64))
        pts = np.stack([(u - cx) / fx * z, (v - cy) / fy * z, z], axis=-1)

        pu = (pts[1:-1, 2:] - pts[1:-1, :-2]) * 0.5
        pv = (pts[2:, 1:-1] - pts[:-2, 1:-1]) * 0.5
        n = np.cross(pu, pv)
        mag = np.linalg.norm(n, axis=-1)
        n = n / np.maximum(mag, 1e-12)[..., None]
        n[n[..., 2] > 0] *= -1.0  # face the camera

        zc = z[1:-1, 1:-1]
        tol = self.depth_edge_rel * zc
        ok = valid[1:-1, 1:-1] & (mag > self.eps)
        for nb in (z[1:-1, 2:], z[1:-1, :-2], z[2:, 1:-1], z[:-2, 1:-1]):
            ok &= (nb > 0) & (np.abs(nb - zc) < tol)

        normals = np.zeros((h, w, 3), dtype=np.float32)
        mask = np.zeros((h, w), dtype=bool)
        normals[1:-1, 1:-1] = np.where(ok[..., None], n, 0.0)
        mask[1:-1, 1:-1] = ok
        return {"normals": normals, "normal_mask": mask}
