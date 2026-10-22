"""Lateral noise of a structured-light sensor, driven by a known normal map.

Lateral noise acts in the image plane: the sensor reads the scene at a slightly
wrong (x, y).  On a smooth surface this changes nothing; at depth edges it
produces the ragged contours of real sensors.  Following Nguyen et al. (2012,
measured on a Kinect) the std is roughly constant in pixels and grows with the
incidence angle ``theta`` between the surface normal and the optical axis::

    sigma_L(theta) = sigma_L0 + k_L * theta / (pi/2 - theta)

``sigma_L`` is in **pixels at the depth resolution**, ``theta`` in radians.  The
term is singular at 90 deg, so ``theta`` is clamped at ``theta_max_deg`` and
``sigma_L`` is capped at ``sigma_max_px``.

Implementation: every output pixel reads the input at ``(x + dx, y + dy)`` with
``(dx, dy) ~ N(0, sigma_L^2)``, using **nearest-neighbour** sampling.  Bilinear
sampling would invent intermediate depths across an edge; real sensors do not.
Optionally the offset field is Gaussian-filtered (length ``correlation_length_px``)
and renormalised to unit variance, so neighbouring pixels shift together.  The length
can follow depth, ``l(z) = l_ref * (z_ref / z)^p`` (p = 0 constant in pixels, p = 1
constant in metres on the surface), implemented with a few log-spaced filter levels.

The normal map is resolved by ``BaseDegradation.__call__`` (dataset normals, else
``data.norm`` estimated from the GT; see ``--normals`` / ``--force-normals``).

Defaults are Kinect v1 values -- **uncalibrated** for the Astra 2.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, ClassVar

import numpy as np

from data.degradation.axial import _fill_nearest, get_theta_map
from data.degradation.base import BaseDegradation, register_degradation

__all__ = ["LateralDegradation", "apply_lateral_noise"]


@register_degradation("lateral_noise")
@dataclass
class LateralDegradation(BaseDegradation):
    """Shift the sampling position of GT depth by angle-dependent Gaussian noise."""

    key: str = "lateral_noise"
    uses_normals: ClassVar[bool] = True

    # --- noise model (Nguyen et al. 2012) ---------------------------------
    sigma_floor_px: float = 0.8
    """``sigma_L0``: std in pixels on a fronto-parallel surface."""
    sigma_angle_gain: float = 0.035
    """``k_L``: coefficient of ``theta / (pi/2 - theta)``, in pixels."""
    theta_max_deg: float = 89.0
    """``theta`` is clamped here so ``pi/2 - theta`` never reaches 0."""
    sigma_max_px: float = 10.0
    """Upper cap on the std in pixels, for grazing pixels."""
    correlation_length_px: float = 0.0
    """Std of the Gaussian filter applied to the offset field, at ``correlation_ref_depth_m``;
    0 = independent pixels."""
    correlation_ref_depth_m: float = 2.0
    """``z_ref``: depth in metres at which the length equals ``correlation_length_px``."""
    correlation_depth_exponent: float = 0.0
    """``p`` in ``l(z) = l_ref * (z_ref / z)^p``.  0 = constant in pixels, 1 = constant in
    metres on the surface, negative = grows with depth.  Uncalibrated hypothesis."""
    correlation_min_px: float = 0.5
    """Lower clip of the depth-dependent length (also the finest filter level)."""
    correlation_max_px: float = 16.0
    """Upper clip of the depth-dependent length (also the coarsest filter level)."""
    correlation_levels: int = 6
    """Number of log-spaced filter levels used when ``p != 0``."""

    # --- operating range --------------------------------------------------
    z_min: float | None = 0.6
    """GT closer than this is invalid, metres; None = no lower limit."""
    z_max: float | None = 8.0
    """GT farther than this is invalid, metres; None = no upper limit."""

    # --- normal-map handling (same meaning as in AxialDegradation) --------
    normal_resize_mode: str = "nearest"
    angle_reference: str = "optical_axis"
    fill_missing_theta: bool = True

    # --- intrinsics in pixels at the depth resolution (viewing_ray only) --
    fx: float = 570.0
    fy: float | None = None
    cx: float | None = None
    cy: float | None = None

    def __post_init__(self) -> None:
        if not 0 < self.correlation_min_px <= self.correlation_max_px:
            raise ValueError("need 0 < correlation_min_px <= correlation_max_px")
        if self.correlation_levels < 2 or self.correlation_ref_depth_m <= 0:
            raise ValueError("need correlation_levels >= 2 and correlation_ref_depth_m > 0")

    def apply(
        self, gt: np.ndarray, normal: np.ndarray, rng: np.random.Generator
    ) -> Mapping[str, Any]:
        # Set depth map
        z = gt.astype(np.float64)
        valid = np.isfinite(z) & (z > 0)
        if self.z_min is not None:
            valid &= z >= self.z_min
        if self.z_max is not None:
            valid &= z <= self.z_max
        z = np.where(valid, z, 0.0)

        # Set normal map
        theta = get_theta_map(
            normal,
            z.shape,
            resize_mode=self.normal_resize_mode,
            angle_reference=self.angle_reference,
            fx=self.fx,
            fy=self.fy,
            cx=self.cx,
            cy=self.cy,
        )
        theta_missing = float((~np.isfinite(theta) & valid).sum() / max(int(valid.sum()), 1))
        if self.fill_missing_theta:
            theta = _fill_nearest(theta, target=valid)
        theta = np.nan_to_num(theta, nan=0.0)

        noisy = apply_lateral_noise(
            z,
            theta,
            rng,
            sigma_floor_px=self.sigma_floor_px,
            sigma_angle_gain=self.sigma_angle_gain,
            theta_max_deg=self.theta_max_deg,
            sigma_max_px=self.sigma_max_px,
            correlation_length_px=self.correlation_length_px,
            correlation_ref_depth_m=self.correlation_ref_depth_m,
            correlation_depth_exponent=self.correlation_depth_exponent,
            correlation_min_px=self.correlation_min_px,
            correlation_max_px=self.correlation_max_px,
            correlation_levels=self.correlation_levels,
        )
        return {
            "sparse_depth": noisy,
            "lr_depth": noisy,  # full-resolution sensor
            "meta": {
                "theta_missing_ratio": theta_missing,
                "valid_ratio": float((noisy > 0).mean()),
                "calibrated": False,
            },
        }


def apply_lateral_noise(
    z: np.ndarray,
    theta: np.ndarray,
    rng: np.random.Generator,
    *,
    sigma_floor_px: float,
    sigma_angle_gain: float,
    theta_max_deg: float,
    sigma_max_px: float,
    correlation_length_px: float = 0.0,
    correlation_ref_depth_m: float = 2.0,
    correlation_depth_exponent: float = 0.0,
    correlation_min_px: float = 0.5,
    correlation_max_px: float = 16.0,
    correlation_levels: int = 6,
) -> np.ndarray:
    """Resample ``z`` (metres, 0 = invalid) at randomly offset positions, nearest-neighbour.

    ``theta`` is (H, W) radians without NaN.  Both offset fields are always drawn
    (x first, then y), so the RNG stream does not depend on the image content.
    Values of the output are always values of ``z``; invalid (0) pixels that get
    sampled stay 0, which makes edges and holes ragged.

    With ``correlation_depth_exponent`` p != 0 the correlation length follows the
    depth of the output pixel (see :func:`correlation_length_map`).
    """
    z = np.asarray(z, dtype=np.float64)
    h, w = z.shape

    phi = np.clip(theta, 0.0, np.radians(theta_max_deg))
    sigma = sigma_floor_px + sigma_angle_gain * phi / (np.pi / 2 - phi)
    sigma = np.minimum(sigma, sigma_max_px)

    ex = rng.normal(0.0, 1.0, size=z.shape)
    ey = rng.normal(0.0, 1.0, size=z.shape)
    if correlation_length_px > 0 and correlation_depth_exponent == 0:
        ex = _correlate_unit(ex, correlation_length_px)
        ey = _correlate_unit(ey, correlation_length_px)
    elif correlation_length_px > 0:
        ell = correlation_length_map(
            z,
            correlation_length_px,
            ref_depth_m=correlation_ref_depth_m,
            exponent=correlation_depth_exponent,
            min_px=correlation_min_px,
            max_px=correlation_max_px,
        )
        ex = _correlate_unit_varying(
            ex, ell, correlation_levels, correlation_min_px, correlation_max_px
        )
        ey = _correlate_unit_varying(
            ey, ell, correlation_levels, correlation_min_px, correlation_max_px
        )

    ys, xs = np.mgrid[0:h, 0:w]
    ix = np.clip(np.rint(xs + ex * sigma), 0, w - 1).astype(np.intp)
    iy = np.clip(np.rint(ys + ey * sigma), 0, h - 1).astype(np.intp)
    return z[iy, ix].astype(np.float32)


def correlation_length_map(
    z: np.ndarray,
    length_ref_px: float,
    *,
    ref_depth_m: float,
    exponent: float,
    min_px: float,
    max_px: float,
) -> np.ndarray:
    """Per-pixel correlation length ``clip(l_ref * (z_ref / z)^p, min, max)`` in pixels.

    ``p = 0`` is a constant length (matching window fixed in pixels); ``p = 1`` keeps
    the length constant in metres on the surface; ``p < 0`` grows with depth.
    Invalid pixels (``z <= 0``) use ``z_ref``, i.e. ``l_ref``.
    """
    zs = np.where(z > 0, z, ref_depth_m)
    return np.clip(length_ref_px * (ref_depth_m / zs) ** exponent, min_px, max_px)


def _correlate_unit_varying(
    field: np.ndarray, ell: np.ndarray, levels: int, min_px: float, max_px: float
) -> np.ndarray:
    """Like :func:`_correlate_unit`, but with a per-pixel length ``ell``.

    ``ell`` snaps to the nearest of ``levels`` log-spaced lengths in ``[min_px, max_px]``
    (a grid that does not depend on the image).  The same white-noise field is filtered
    once per used level, each result is unit variance, and each pixel takes its level.
    """
    grid = np.geomspace(min_px, max_px, levels)
    idx = np.abs(np.log(ell)[..., None] - np.log(grid)).argmin(axis=-1)
    out = np.empty_like(field)
    for k in np.unique(idx):
        sel = idx == k
        out[sel] = _correlate_unit(field, float(grid[k]))[sel]
    return out


def _correlate_unit(field: np.ndarray, length: float) -> np.ndarray:
    """Gaussian-filter a white-noise field and renormalise it to unit variance."""
    from scipy import ndimage

    smooth = ndimage.gaussian_filter(field, sigma=length, mode="reflect")
    std = smooth.std()
    return smooth / std if std > 1e-12 else smooth
