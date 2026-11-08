"""Missing values (holes) whose rate depends on the incidence angle theta.

A pixel is dropped (set to 0 = no measurement) with probability::

    p(theta) = p_min                                         theta <= theta0
    p(theta) = min(p_max, p_min + eta * (theta - theta0))    theta >  theta0

so from ``theta0`` (degrees) the fraction of missing values grows linearly with the angle
(``eta`` is the fraction added per degree).  ``theta`` is between the surface normal and the
optical axis / viewing ray, as in ``axial_noise`` and ``lateral_noise``; the normal map is
resolved by ``BaseDegradation.__call__``.  Pixels are dropped independently.

Defaults are placeholders -- **uncalibrated** for the Astra 2.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, ClassVar

import numpy as np

from data.degradation.base import BaseDegradation, register_degradation
from data.norm import get_theta_map
from utils.misc import fill_nan_nearest

__all__ = ["ThetaDropoutDegradation", "apply_theta_dropout"]


@register_degradation("theta_dropout")
@dataclass
class ThetaDropoutDegradation(BaseDegradation):
    """Zero out depth pixels with a probability that rises linearly with theta."""

    key: str = "theta_dropout"
    uses_normals: ClassVar[bool] = True

    # --- dropout model ----------------------------------------------------
    theta0: float = 60.0
    """Angle in degrees above which the missing fraction starts to increase."""
    p_min: float = 0.0
    """Missing fraction for ``theta <= theta0`` (0..1)."""
    eta: float = 0.04
    """Increase of the missing fraction per degree above ``theta0``."""
    p_max: float = 1.0
    """Upper cap on the missing fraction."""

    # --- normal-map handling ----------------------------------------------
    normal_resize_mode: str = "guided"
    """``guided`` (default: bilinear inside a surface, nearest across a crease or
    hole), ``nearest`` or ``bilinear``."""
    normal_resize_order: str = "normals"
    """``normals``: resize the vectors, then compute theta; ``theta``: the reverse."""
    normal_threshold_deg: float = 10.0
    """``guided`` only: angle above which a neighbour counts as another surface."""
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
        z = np.where(valid, z, 0.0)

        theta = get_theta_map(
            normal,
            z.shape,
            resize_mode=self.normal_resize_mode,
            resize_order=self.normal_resize_order,
            normal_threshold_deg=self.normal_threshold_deg,
            angle_reference=self.angle_reference,
            fx=self.fx,
            fy=self.fy,
            cx=self.cx,
            cy=self.cy,
        )
        theta_missing = float((~np.isfinite(theta) & valid).sum() / max(int(valid.sum()), 1))
        if self.fill_missing_theta:
            theta = fill_nan_nearest(theta, target=valid)
        theta = np.nan_to_num(theta, nan=0.0)

        out = apply_theta_dropout(
            z,
            theta,
            rng,
            theta0=self.theta0,
            p_min=self.p_min,
            eta=self.eta,
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
    theta: np.ndarray, *, theta0: float, p_min: float, eta: float, p_max: float
) -> np.ndarray:
    """Per-pixel missing probability for ``theta`` in radians."""
    # delta: degrees by which theta exceeds theta0 (0 below theta0, so p stays at p_min there)
    delta = np.maximum(np.degrees(theta) - theta0, 0.0)
    return np.clip(p_min + eta * delta, 0.0, p_max)


def apply_theta_dropout(
    z: np.ndarray,
    theta: np.ndarray,
    rng: np.random.Generator,
    *,
    theta0: float,
    p_min: float,
    eta: float,
    p_max: float,
) -> np.ndarray:
    """Return ``z`` (metres, 0 = invalid) with pixels dropped by ``p(theta)``.

    ``theta`` is (H, W) radians without NaN.  The full random field is always drawn,
    so the RNG stream does not depend on which pixels are valid.
    """
    z = np.asarray(z, dtype=np.float64)
    valid = np.isfinite(z) & (z > 0)
    p = dropout_probability(theta, theta0=theta0, p_min=p_min, eta=eta, p_max=p_max)
    drop = rng.random(z.shape) < p
    return np.where(valid & ~drop, z, 0.0).astype(np.float32)
