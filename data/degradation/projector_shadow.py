"""Projector shadows of a structured-light / active-stereo sensor.

An active-stereo camera only measures surfaces its IR projector illuminates.
A surface the camera sees but the projector cannot reach -- because a nearer
surface sits in between -- returns no pattern and therefore no depth.  On
stairs the nearer surface is a nosing, and the hole is the band of riser or
tread right next to it.

Model (shadow mapping)
----------------------
The projector is a pinhole at ``offset_t`` in the camera frame, with axes
parallel to the camera's and the same intrinsics.  Every valid camera pixel is
back-projected to 3-D, moved into the projector frame, and projected into the
projector image.  A z-buffer over that image keeps the nearest surface per
projector pixel; a camera pixel whose depth is behind that nearest surface (by
more than a bias) is in shadow and becomes 0.

Geometry to keep in mind: a shadow forms only *across* the offset.  A purely
horizontal offset shadows the sides of vertical edges and leaves horizontal
nosings untouched; holes below nosings need a vertical component
(``offset_t[1] < 0``, projector above the camera).

The default offset is a **placeholder**, not the measured Astra 2 geometry.
Results produced with it are uncalibrated.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from data.degradation.base import BaseDegradation, register_degradation

__all__ = ["ProjectorShadowDegradation", "projector_shadow_mask"]


@register_degradation("projector_shadow")
@dataclass
class ProjectorShadowDegradation(BaseDegradation):
    """Invalidate camera pixels the projector cannot illuminate. Deterministic."""

    key: str = "projector_shadow"

    offset_t: tuple[float, float, float] = (0.0, -0.04, 0.0)
    """Projector centre in the camera frame, in **metres** (x right, y down,
    z forward).  Placeholder value: the real Astra 2 geometry is not measured."""

    # --- intrinsics, in **pixels** at the working resolution -------------
    fx: float = 570.0
    """Horizontal focal length in pixels.  Rescale with the image (570 ~ 640x480)."""
    fy: float | None = None
    """Vertical focal length in pixels; ``None`` means ``fx``."""
    cx: float | None = None
    """Principal point x in pixels; ``None`` means the image centre."""
    cy: float | None = None
    """Principal point y in pixels; ``None`` means the image centre."""

    # --- shadow-test bias -------------------------------------------------
    depth_tol_rel: float = 0.01
    """Constant bias: a pixel up to this fraction of its depth behind the
    nearest surface still counts as lit."""
    slope_bias: float = 1.5
    """Slope-scaled bias, in multiples of the local per-pixel depth step.
    Prevents self-shadowing ("acne") on slanted treads and floors."""

    fov_clip: bool = False
    """If True, pixels projecting outside the projector image (same size and
    intrinsics as the camera) are unlit.  Off by default: a real projector cone
    usually covers the camera's field of view."""

    def apply(self, gt: np.ndarray, rng: np.random.Generator) -> Mapping[str, Any]:
        z = gt.astype(np.float64)
        valid = np.isfinite(z) & (z > 0)
        h, w = z.shape

        shadow = projector_shadow_mask(
            z,
            valid,
            fx=self.fx,
            fy=self.fx if self.fy is None else self.fy,
            cx=(w - 1) / 2.0 if self.cx is None else self.cx,
            cy=(h - 1) / 2.0 if self.cy is None else self.cy,
            offset_t=self.offset_t,
            tol_rel=self.depth_tol_rel,
            slope_bias=self.slope_bias,
            fov_clip=self.fov_clip,
        )

        sparse = np.where(shadow | ~valid, 0.0, z).astype(np.float32)
        n_valid = max(int(valid.sum()), 1)
        return {
            "sparse_depth": sparse,
            "lr_depth": None,  # full-resolution sensor
            "meta": {
                "shadow_ratio": float(shadow.sum() / n_valid),
                "valid_ratio": float((sparse > 0).mean()),
                "calibrated": False,
            },
        }


def projector_shadow_mask(
    z: np.ndarray,
    valid: np.ndarray,
    *,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    offset_t: Sequence[float],
    tol_rel: float,
    slope_bias: float,
    fov_clip: bool,
) -> np.ndarray:
    """Return a bool (H, W) mask of valid pixels the projector does not light."""
    h, w = z.shape
    tx, ty, tz = (float(v) for v in offset_t)

    vs, us = np.nonzero(valid)
    zc = z[vs, us]

    # camera -> 3-D -> projector frame -> projector pixel
    xp = (us - cx) * zc / fx - tx
    yp = (vs - cy) * zc / fy - ty
    zp = zc - tz
    front = zp > 1e-9
    with np.errstate(divide="ignore", invalid="ignore"):
        iu = np.rint(np.where(front, fx * xp / zp + cx, -1.0)).astype(np.int64)
        iv = np.rint(np.where(front, fy * yp / zp + cy, -1.0)).astype(np.int64)
    inside = front & (iu >= 0) & (iu < w) & (iv >= 0) & (iv < h)

    # z-buffer: nearest surface per projector pixel.  Each camera pixel writes
    # the cell it later reads, so its own depth is always there.
    cell = iv[inside] * w + iu[inside]
    zbuf = np.full(h * w, np.inf)
    np.minimum.at(zbuf, cell, zp[inside])

    bias = tol_rel * zp[inside] + slope_bias * _depth_step(z, valid)[vs[inside], us[inside]]
    lit = zp[inside] <= zbuf[cell] + bias

    shadow = np.zeros((h, w), dtype=bool)
    shadow[vs[inside], us[inside]] = ~lit
    # behind the projector: never lit; outside its image: lit unless fov_clip
    outside = ~inside
    if not fov_clip:
        outside &= ~front
    shadow[vs[outside], us[outside]] = True
    return shadow


def _depth_step(z: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """Local per-pixel depth change, ignoring depth edges.

    Per axis, the smaller of the forward and backward one-sided differences:
    on a smooth slope both are the slope, right at an edge one of them is the
    jump and the other is not -- so a real shadow next to an edge survives.
    """
    zn = np.where(valid, z, np.nan)
    steps = []
    for axis in (0, 1):
        d = np.abs(np.diff(zn, axis=axis))
        pad = [(0, 0), (0, 0)]
        pad[axis] = (1, 0)
        back = np.pad(d, pad, constant_values=np.nan)
        pad[axis] = (0, 1)
        fwd = np.pad(d, pad, constant_values=np.nan)
        with np.errstate(invalid="ignore"):
            steps.append(np.nan_to_num(np.fmin(back, fwd), nan=0.0))
    return np.hypot(steps[0], steps[1])
