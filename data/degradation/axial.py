"""Axial noise of a structured-light sensor, driven by a known normal map.

Axial noise acts along the sensor's z-axis.  It is zero-mean Gaussian, with a
std that depends on the depth ``D`` and on the angle ``phi`` between the surface
normal and the optical axis (Nguyen et al., 2012, measured on a Kinect)::

    sigma_z = sigma0 + alpha * (D - z0)^2                                             phi <= theta0
    sigma_z = sigma0 + alpha * (D - z0)^2 + beta / sqrt(D) * phi^2 / (pi/2 - phi)^2   phi > theta0

``D``, ``z0``, ``sigma_z`` and ``sigma0`` are in **metres**, ``phi`` in radians,
``theta0`` in degrees (60 in the paper).  The paper gives no data below 10 deg;
the first formula is used there.  The grazing term is singular at 90 deg, so
``phi`` is clamped at ``theta_max`` (degrees) and ``sigma_z`` is capped at
``sigma_max`` (metres).

The normal map is resolved by ``BaseDegradation.__call__``: the dataset's (e.g.
Hypersim's ``normal_cam``), else estimated from the **clean** GT depth with
``data.norm`` (``--normals`` / ``--force-normals``).  Estimators zero normals across
depth jumps -- the stair nosings -- so those pixels take the nearest valid angle;
native normals do not have that limitation.

Every finite, positive GT pixel is noised; the sensor's operating range is
applied by a separate degradation step, not here.

Defaults are the Kinect values from the paper -- **uncalibrated** for the Astra 2.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, ClassVar

import numpy as np

from data.degradation.base import BaseDegradation, register_degradation
from data.norm import get_theta_map
from utils.misc import fill_nan_nearest

__all__ = ["AxialDegradation", "apply_axial_noise"]


@register_degradation("axial_noise")
@dataclass
class AxialDegradation(BaseDegradation):
    """Add depth- and angle-dependent Gaussian noise along z to GT depth."""

    key: str = "axial_noise"
    uses_normals: ClassVar[bool] = True

    # --- noise model (Nguyen et al. 2012) ---------------------------------
    sigma0: float = 0.0012
    """Std in metres at depth ``z0``, fronto-parallel surface (``bias`` in the paper)."""
    alpha: float = 0.0019
    """Coefficient of ``(D - z0)^2`` in metres^-1."""
    z0: float = 0.4
    """Depth of minimal noise, in metres (``d0`` in the paper)."""
    beta: float = 0.0001
    """Coefficient of the grazing-angle term."""
    theta0: float = 60.0
    """Angle in degrees above which the grazing term is added."""
    theta_max: float = 89.0
    """``phi`` is clamped here (degrees) so ``pi/2 - phi`` never reaches 0."""
    sigma_max: float = 0.1
    """Upper cap on the std in metres, for grazing pixels."""

    # --- normal-map handling ----------------------------------------------
    normal_resize_mode: str = "guided"
    """How a normal map of a different size is resized to the depth map:
    ``guided`` (default: bilinear inside a surface, nearest across a crease or
    hole), ``nearest`` or ``bilinear``."""
    normal_resize_order: str = "normals"
    """``normals``: resize the vectors, then compute theta.  ``theta``: compute
    theta at the normal map's resolution, then resize the angle map."""
    normal_threshold_deg: float = 10.0
    """``guided`` only: taps differing from the nearest source pixel by more than
    this angle are treated as another surface."""
    angle_reference: str = "optical_axis"
    """``optical_axis``: phi against the camera z axis (as in the paper).
    ``viewing_ray``: phi against the per-pixel ray; uses the intrinsics below."""
    fill_missing_theta: bool = True
    """Fill pixels with no usable normal from the nearest pixel that has one."""

    # --- intrinsics in pixels at the depth resolution (viewing_ray only) --
    fx: float = 570.0
    """Horizontal focal length in pixels.  Rescale with the image (570 ~ 640x480)."""
    fy: float | None = None
    """Vertical focal length in pixels; ``None`` means ``fx``."""
    cx: float | None = None
    """Principal point x in pixels; ``None`` means the image centre."""
    cy: float | None = None
    """Principal point y in pixels; ``None`` means the image centre."""

    def apply(
        self,
        gt: np.ndarray,  # depth map
        normal: np.ndarray,  # normal map
        rng: np.random.Generator,  # randomizer
    ) -> Mapping[str, Any]:

        # Set depth map
        z = gt.astype(np.float64)
        valid = np.isfinite(z) & (z > 0)
        z_missing = int(valid.sum())
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
        theta_missing = int((~np.isfinite(theta) & valid).sum())
        theta_missing_ratio = float(theta_missing / max(z_missing, 1))
        if self.fill_missing_theta:
            theta = fill_nan_nearest(theta, target=valid)
        theta = np.nan_to_num(theta, nan=0.0)

        # Get noiesd map
        noisy = apply_axial_noise(
            z,
            theta,
            rng,
            sigma0=self.sigma0,
            alpha=self.alpha,
            z0=self.z0,
            beta=self.beta,
            theta0=self.theta0,
            theta_max=self.theta_max,
            sigma_max=self.sigma_max,
        )
        return {
            "sparse_depth": noisy,
            "lr_depth": noisy,
            "meta": {
                "theta_missing_ratio": theta_missing_ratio,
                "valid_ratio": float((noisy > 0).mean()),
                "calibrated": True,
            },
        }


def apply_axial_noise(
    z: np.ndarray,
    theta: np.ndarray,
    rng: np.random.Generator,
    *,
    sigma0: float,
    alpha: float,
    z0: float,
    beta: float,
    theta0: float,
    theta_max: float,
    sigma_max: float,
) -> np.ndarray:
    """Return ``z`` (metres, 0 = invalid) plus axial Gaussian noise; invalid stays 0.

    ``theta`` is (H, W) radians without NaN.  The full noise field is always drawn,
    so the RNG stream does not depend on which pixels are valid.
    """
    z = np.asarray(z, dtype=np.float64)
    valid = np.isfinite(z) & (z > 0)
    zs = np.where(valid, z, 1.0)  # keep sqrt / division safe on invalid pixels

    sigma = sigma0 + alpha * (zs - z0) ** 2
    phi = np.clip(theta, 0.0, np.radians(theta_max))
    grazing = beta / np.sqrt(zs) * phi**2 / (np.pi / 2 - phi) ** 2
    sigma = sigma + np.where(phi > np.radians(theta0), grazing, 0.0)
    sigma = np.minimum(sigma, sigma_max)

    noise = rng.normal(0.0, 1.0, size=z.shape) * sigma
    z_noised = np.where(valid, z + noise, 0.0).astype(np.float32)
    return z_noised
