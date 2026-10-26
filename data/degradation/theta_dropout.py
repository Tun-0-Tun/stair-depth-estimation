"""Missing values (holes) whose rate depends on the incidence angle theta.

A pixel is dropped (set to 0 = no measurement) with probability::

    p(theta) = p0                                              theta <= theta_0
    p(theta) = min(p_max, p0 + slope * (theta - theta_0))      theta >  theta_0

so from ``theta_0`` the percentage of missing values grows linearly with the angle
(``slope`` is the fraction added per degree).  ``theta`` is between the surface normal and the optical axis / viewing ray, as in
``axial_noise`` and ``lateral_noise``; the normal map is resolved by
``BaseDegradation.__call__``.  Pixels are dropped independently.

Defaults are placeholders -- **uncalibrated** for the Astra 2.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, ClassVar

import numpy as np

from data.degradation.axial import _fill_nearest, get_theta_map
from data.degradation.base import BaseDegradation, register_degradation

__all__ = ["ThetaDropoutDegradation", "apply_theta_dropout"]


@register_degradation("theta_dropout")
@dataclass
class ThetaDropoutDegradation(BaseDegradation):
    """Zero out depth pixels with a probability that rises linearly with theta."""

    key: str = "theta_dropout"
    uses_normals: ClassVar[bool] = True

    # --- dropout model ----------------------------------------------------
    theta_0_deg: float = 60.0
    """Angle above which the missing fraction starts to increase."""
    p0: float = 0.0
    """Missing fraction for ``theta <= theta_0``."""
    slope_per_deg: float = 0.04
    """Increase of the missing fraction per degree above ``theta_0``."""
    p_max: float = 1.0
    """Upper cap on the missing fraction."""

    # --- operating range --------------------------------------------------
    z_min: float | None = 0.6
    """GT closer than this is invalid, metres; None = no lower limit."""
    z_max: float | None = 8.0
    """GT farther than this is invalid, metres; None = no upper limit."""

    # --- normal-map handling ----------------------------------------------
    normal_resize_mode: str = "nearest"
    """``nearest`` (default, never invents a normal) or ``bilinear``."""
    angle_reference: str = "optical_axis"
    """``optical_axis`` or ``viewing_ray`` (uses the intrinsics below)."""
    fill_missing_theta: bool = True
    """Fill pixels with no usable normal from the nearest pixel that has one."""

    # --- intrinsics in pixels at the depth resolution (viewing_ray only) --
    fx: float = 570.0
    """Horizontal focal length in pixels."""
    fy: float | None = None
    """Vertical focal length in pixels; ``None`` means ``fx``."""
    cx: float | None = None
    """Principal point x in pixels; ``None`` means the image centre."""
    cy: float | None = None
    """Principal point y in pixels; ``None`` means the image centre."""

    def apply(
        self, gt: np.ndarray, normal: np.ndarray, rng: np.random.Generator
    ) -> Mapping[str, Any]:
        z = gt.astype(np.float64)
        valid = np.isfinite(z) & (z > 0)
        if self.z_min is not None:
            valid &= z >= self.z_min
        if self.z_max is not None:
            valid &= z <= self.z_max
        z = np.where(valid, z, 0.0)

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

        out = apply_theta_dropout(
            z,
            theta,
            rng,
            theta_0_deg=self.theta_0_deg,
            p0=self.p0,
            slope_per_deg=self.slope_per_deg,
            p_max=self.p_max,
        )
        n_valid = max(int(valid.sum()), 1)
        return {
            "sparse_depth": out,
            "lr_depth": out,
            "meta": {
                "theta_missing_ratio": theta_missing,
                "missing_fraction": float(((out <= 0) & valid).sum() / n_valid),
                "valid_ratio": float((out > 0).mean()),
                "calibrated": False,
            },
        }


def dropout_probability(
    theta: np.ndarray, *, theta_0_deg: float, p0: float, slope_per_deg: float, p_max: float
) -> np.ndarray:
    """Per-pixel missing probability for ``theta`` in radians."""
    excess_deg = np.maximum(np.degrees(theta) - theta_0_deg, 0.0)
    return np.clip(p0 + slope_per_deg * excess_deg, 0.0, p_max)


def apply_theta_dropout(
    z: np.ndarray,
    theta: np.ndarray,
    rng: np.random.Generator,
    *,
    theta_0_deg: float,
    p0: float,
    slope_per_deg: float,
    p_max: float,
) -> np.ndarray:
    """Return ``z`` (metres, 0 = invalid) with pixels dropped by ``p(theta)``.

    ``theta`` is (H, W) radians without NaN.  The full random field is always drawn,
    so the RNG stream does not depend on which pixels are valid.
    """
    z = np.asarray(z, dtype=np.float64)
    valid = np.isfinite(z) & (z > 0)
    p = dropout_probability(
        theta, theta_0_deg=theta_0_deg, p0=p0, slope_per_deg=slope_per_deg, p_max=p_max
    )
    drop = rng.random(z.shape) < p
    return np.where(valid & ~drop, z, 0.0).astype(np.float32)
