"""Axial noise of a structured-light sensor, driven by a known normal map.

Axial noise acts along the sensor's z-axis.  It is zero-mean Gaussian, with a
std that depends on the depth ``D`` and on the angle ``phi`` between the surface
normal and the optical axis (Nguyen et al., 2012, measured on a Kinect)::

    sigma_z = bias + alpha * (D - d0)^2                                     phi <= 60 deg
    sigma_z = bias + alpha * (D - d0)^2 + beta / sqrt(D) * phi^2 / (pi/2 - phi)^2   phi > 60 deg

``D`` and ``sigma_z`` are in **metres**, ``phi`` in radians.  The paper gives no
data below 10 deg; the first formula is used there.  The grazing term is
singular at 90 deg, so ``phi`` is clamped at ``theta_max_deg`` and ``sigma`` is
capped at ``sigma_max_m``.

The model needs the normal map as an input (``normal=`` of ``__call__``); there
is deliberately no silent fallback that estimates normals from noisy depth.

Defaults are the Kinect values from the paper -- **uncalibrated** for the Astra 2.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, ClassVar

import numpy as np

from data.degradation.base import BaseDegradation, register_degradation
from utils.misc import resize_depth

__all__ = ["AxialDegradation", "apply_axial_noise", "get_theta_map"]


@register_degradation("axial_noise")
@dataclass
class AxialDegradation(BaseDegradation):
    """Add depth- and angle-dependent Gaussian noise along z to GT depth."""

    key: str = "axial_noise"
    uses_normals: ClassVar[bool] = True

    # --- noise model (Nguyen et al. 2012) ---------------------------------
    sigma_floor_m: float = 0.0012
    """Std in metres at the reference depth, fronto-parallel surface."""
    sigma_depth_gain: float = 0.0019
    """Coefficient of ``(D - reference_depth_m)^2`` in metres^-1."""
    reference_depth_m: float = 0.4
    """Depth of minimal noise, in metres."""
    sigma_grazing_gain: float = 0.0001
    """Coefficient of the grazing-angle term (``beta``)."""
    grazing_start_deg: float = 60.0
    """Angle above which the grazing term is added."""
    theta_max_deg: float = 89.0
    """``phi`` is clamped here so ``pi/2 - phi`` never reaches 0."""
    sigma_max_m: float = 0.1
    """Upper cap on the std in metres, for grazing pixels."""

    # --- operating range --------------------------------------------------
    z_min: float = 0.6
    """GT closer than this (and noisy depth closer than this) is invalid, metres."""
    z_max: float = 8.0
    """GT farther than this (and noisy depth farther than this) is invalid, metres."""

    # --- normal-map handling ----------------------------------------------
    normal_resize_mode: str = "nearest"
    """How a normal map of a different size is resized to the depth map:
    ``nearest`` (default, never invents a normal) or ``bilinear``."""
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
        self, gt: np.ndarray, rng: np.random.Generator, normal: np.ndarray | None = None
    ) -> Mapping[str, Any]:
        if normal is None:
            raise ValueError("AxialDegradation needs a normal map: pass normal=(H, W, 3)")
        z = gt.astype(np.float64)
        valid = np.isfinite(z) & (z >= self.z_min) & (z <= self.z_max)
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

        noisy = apply_axial_noise(
            z,
            theta,
            rng,
            sigma_floor_m=self.sigma_floor_m,
            sigma_depth_gain=self.sigma_depth_gain,
            reference_depth_m=self.reference_depth_m,
            sigma_grazing_gain=self.sigma_grazing_gain,
            grazing_start_deg=self.grazing_start_deg,
            theta_max_deg=self.theta_max_deg,
            sigma_max_m=self.sigma_max_m,
        )
        noisy[(noisy < self.z_min) | (noisy > self.z_max)] = 0.0
        return {
            "sparse_depth": noisy,
            "lr_depth": None,  # full-resolution sensor
            "meta": {
                "normal_source": "provided",
                "theta_missing_ratio": theta_missing,
                "valid_ratio": float((noisy > 0).mean()),
                "calibrated": False,
            },
        }


def get_theta_map(
    normal: np.ndarray,
    out_shape: tuple[int, int],
    *,
    resize_mode: str = "nearest",
    angle_reference: str = "optical_axis",
    fx: float = 570.0,
    fy: float | None = None,
    cx: float | None = None,
    cy: float | None = None,
) -> np.ndarray:
    """Angle theta (radians) between surface normal and optical axis / viewing ray.

    ``normal`` is (H_n, W_n, 3) in the camera frame (x right, y down, z forward),
    any sign convention.  If its size differs from ``out_shape`` the *vectors* are
    resized (``nearest`` by default: stairs are piecewise planar, and blending a
    tread with a riser would create phantom ~45 deg normals along every nosing),
    then renormalised.  Returns float32 (H, W), NaN where the normal is unusable.
    """
    n = np.asarray(normal, dtype=np.float32)
    if n.ndim != 3 or n.shape[2] != 3:
        raise ValueError(f"normal must be (H, W, 3), got {n.shape}")
    h, w = out_shape
    hn, wn = n.shape[:2]
    if abs(h / hn - w / wn) > 0.02 * (h / hn):
        raise ValueError(f"normal {n.shape[:2]} and depth {out_shape} differ in aspect ratio")

    n = np.where(np.isfinite(n), n, 0.0)  # non-finite -> zero vector -> "missing"
    if (hn, wn) != (h, w):
        n = np.stack([resize_depth(n[..., c], (h, w), mode=resize_mode) for c in range(3)], -1)
    norm = np.linalg.norm(n, axis=-1, keepdims=True)
    ok = norm[..., 0] > 1e-6
    n = n / np.maximum(norm, 1e-6)

    if angle_reference == "optical_axis":
        cos = np.abs(n[..., 2])
    elif angle_reference == "viewing_ray":
        fy = fx if fy is None else fy
        cx = (w - 1) / 2.0 if cx is None else cx
        cy = (h - 1) / 2.0 if cy is None else cy
        vs, us = np.mgrid[0:h, 0:w]
        ray = np.stack([(us - cx) / fx, (vs - cy) / fy, np.ones((h, w))], axis=-1)
        ray /= np.linalg.norm(ray, axis=-1, keepdims=True)
        cos = np.abs(np.sum(n * ray, axis=-1))
    else:
        raise ValueError(
            f"angle_reference must be optical_axis or viewing_ray, got {angle_reference!r}"
        )

    theta = np.arccos(np.clip(cos, 0.0, 1.0))
    return np.where(ok, theta, np.nan).astype(np.float32)


def apply_axial_noise(
    z: np.ndarray,
    theta: np.ndarray,
    rng: np.random.Generator,
    *,
    sigma_floor_m: float,
    sigma_depth_gain: float,
    reference_depth_m: float,
    sigma_grazing_gain: float,
    grazing_start_deg: float,
    theta_max_deg: float,
    sigma_max_m: float,
) -> np.ndarray:
    """Return ``z`` (metres, 0 = invalid) plus axial Gaussian noise; invalid stays 0.

    ``theta`` is (H, W) radians without NaN.  The full noise field is always drawn,
    so the RNG stream does not depend on which pixels are valid.
    """
    z = np.asarray(z, dtype=np.float64)
    valid = np.isfinite(z) & (z > 0)
    zs = np.where(valid, z, 1.0)  # keep sqrt / division safe on invalid pixels

    sigma = sigma_floor_m + sigma_depth_gain * (zs - reference_depth_m) ** 2
    phi = np.clip(theta, 0.0, np.radians(theta_max_deg))
    grazing = sigma_grazing_gain / np.sqrt(zs) * phi**2 / (np.pi / 2 - phi) ** 2
    sigma = sigma + np.where(phi > np.radians(grazing_start_deg), grazing, 0.0)
    sigma = np.minimum(sigma, sigma_max_m)

    noise = rng.normal(0.0, 1.0, size=z.shape) * sigma
    return np.where(valid, z + noise, 0.0).astype(np.float32)


def _fill_nearest(values: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Fill NaNs inside ``target`` with the nearest finite value (0 is a legitimate value).

    Only pixels that are both missing and in ``target`` (valid depth) are written;
    other NaNs are left as they are.  Any finite pixel may serve as a source.
    """
    from scipy import ndimage

    known = np.isfinite(values)
    todo = ~known & target
    if not todo.any():
        return values
    if not known.any():
        return np.where(todo, 0.0, values).astype(values.dtype)
    _, (iy, ix) = ndimage.distance_transform_edt(~known, return_indices=True)
    return np.where(todo, values[iy, ix], values)
