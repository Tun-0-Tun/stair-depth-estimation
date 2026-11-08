"""Lateral noise of a structured-light sensor, driven by a known normal map.

Lateral noise acts in the image plane: the sensor reads the scene at a slightly
wrong (x, y).  On a smooth surface this changes nothing; at depth edges it
produces the ragged contours of real sensors.  Following Nguyen et al. (2012,
measured on a Kinect) the std is roughly constant in pixels and grows with the
incidence angle ``theta`` between the surface normal and the optical axis::

    sigma_L(theta) = sigma0 + gamma * theta / (pi/2 - theta)

``sigma_L`` is in **pixels at the depth resolution**, ``theta`` in radians.  The
term is singular at 90 deg, so ``theta`` is clamped at ``theta_max`` and
``sigma_L`` is capped at ``sigma_max``.

Implementation: every output pixel reads the input at ``(x + dx, y + dy)`` with
``(dx, dy) ~ N(0, sigma_L^2)``, using **nearest-neighbour** sampling.  Bilinear
sampling would invent intermediate depths across an edge; real sensors do not.
Optionally the white-noise fields behind ``dx`` and ``dy`` are Gaussian-filtered
(std ``gauss_std`` pixels) and renormalised to unit variance, so neighbouring
pixels shift together, as they do when they share the sensor's matching window.

The normal map is resolved by ``BaseDegradation.__call__`` (dataset normals, else
``data.norm`` estimated from the GT; see ``--normals`` / ``--force-normals``).

Defaults are Kinect v1 values -- **uncalibrated** for the Astra 2.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, ClassVar

import numpy as np

from data.degradation.base import BaseDegradation, register_degradation
from data.norm import get_theta_map
from utils.misc import fill_nan_nearest

__all__ = ["LateralDegradation", "apply_lateral_noise"]


@register_degradation("lateral_noise")
@dataclass
class LateralDegradation(BaseDegradation):
    """Shift the sampling position of GT depth by angle-dependent Gaussian noise."""

    key: str = "lateral_noise"
    uses_normals: ClassVar[bool] = True

    # --- noise model (Nguyen et al. 2012) ---------------------------------
    sigma0: float = 0.8
    """Std in pixels on a fronto-parallel surface (``sigma_L0`` in the paper)."""
    gamma: float = 0.035
    """Coefficient of ``theta / (pi/2 - theta)``, in pixels (``k_L`` in the paper)."""
    theta_max: float = 89.0
    """``theta`` is clamped here (degrees) so ``pi/2 - theta`` never reaches 0."""
    sigma_max: float = 10.0
    """Upper cap on the std in pixels, for grazing pixels."""

    # --- correlation of the offset field ----------------------------------
    gauss_std: float = 0.0
    """Std in pixels of the Gaussian filter applied to the offset field; 0 = independent
    pixels."""

    # --- normal-map handling (same meaning as in AxialDegradation) --------
    normal_resize_mode: str = "guided"
    normal_resize_order: str = "normals"
    normal_threshold_deg: float = 10.0
    angle_reference: str = "optical_axis"
    fill_missing_theta: bool = True

    # --- intrinsics in pixels at the depth resolution (viewing_ray only) --
    fx: float = 570.0
    fy: float | None = None
    cx: float | None = None
    cy: float | None = None

    def apply(
        self, gt: np.ndarray, normal: np.ndarray, rng: np.random.Generator
    ) -> Mapping[str, Any]:
        # Set depth map
        z = gt.astype(np.float64)
        valid = np.isfinite(z) & (z > 0)
        z = np.where(valid, z, 0.0)

        # Set theta map
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

        # Get noised map
        noisy = apply_lateral_noise(
            z,
            theta,
            rng,
            sigma0=self.sigma0,
            gamma=self.gamma,
            theta_max=self.theta_max,
            sigma_max=self.sigma_max,
            gauss_std=self.gauss_std,
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
    sigma0: float,
    gamma: float,
    theta_max: float,
    sigma_max: float,
    gauss_std: float = 0.0,
) -> np.ndarray:
    """Resample ``z`` (metres, 0 = invalid) at randomly offset positions, nearest-neighbour.

    ``theta`` is (H, W) radians without NaN.  Both white-noise fields are always drawn
    (x first, then y), so the RNG stream does not depend on the image content.
    Values of the output are always values of ``z``; invalid (0) pixels that get
    sampled stay 0, which makes edges and holes ragged.

    With ``gauss_std > 0`` the white-noise fields are Gaussian-filtered and renormalised
    to unit variance before scaling by ``sigma_L``, so neighbouring pixels shift together.
    """
    z = np.asarray(z, dtype=np.float64)
    h, w = z.shape

    phi = np.clip(theta, 0.0, np.radians(theta_max))
    sigma = sigma0 + gamma * phi / (np.pi / 2 - phi)
    sigma = np.minimum(sigma, sigma_max)

    error_x = rng.normal(0.0, 1.0, size=z.shape)
    error_y = rng.normal(0.0, 1.0, size=z.shape)
    if gauss_std > 0:
        error_x = _correlate_unit(error_x, gauss_std)
        error_y = _correlate_unit(error_y, gauss_std)

    ys, xs = np.mgrid[0:h, 0:w]
    ix = np.clip(np.rint(xs + error_x * sigma), 0, w - 1).astype(np.intp)
    iy = np.clip(np.rint(ys + error_y * sigma), 0, h - 1).astype(np.intp)
    return z[iy, ix].astype(np.float32)


def _correlate_unit(field: np.ndarray, gauss_std: float) -> np.ndarray:
    """Gaussian-filter a white-noise field and renormalise it to unit variance."""
    from scipy import ndimage

    smooth = ndimage.gaussian_filter(field, sigma=gauss_std, mode="reflect")
    std = smooth.std()
    return smooth / std if std > 1e-12 else smooth
