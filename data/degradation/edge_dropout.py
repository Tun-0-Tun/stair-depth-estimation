# ruff: noqa: I001
"""Missing depth next to edges.

Real active-stereo holes cluster where the geometry changes: along depth steps and where the
normal turns.  This model removes exactly those pixels::

    d_ij      = max( |n_i . (P_j - P_i)| , |n_j . (P_j - P_i)| )     j: the 8 neighbours at z_step
    edge_i    : d_ij > z_epsilon * z_i                for some j
                or angle(n_i, n_k) > normal_thresh    for some k, the 8 pixels at normal_step
    radius(p) = radius * (1 + radius_jitter * noise(p))              noise: mean 0, std 1,
                                                                     length radius_noise_sigma
    hole(p)   : distance(p, nearest edge pixel) <= radius(p)

``P_i`` is pixel ``i`` back-projected to 3-D (metres), ``n_i`` its unit normal and ``z_i`` its
depth.  ``d_ij`` is the distance of the neighbour from the tangent plane of ``i`` (and of ``i``
from that of ``j``; the larger counts): ~0 on a plane whatever its tilt, large across a depth
step, so tilted floors and treads are not edges.  Steps and distances are in pixels, angles in
degrees, ``z_epsilon`` and ``radius_jitter`` are dimensionless.

1. A pixel is an **edge pixel** (``plane_edge_mask``): one of the two conditions above holds,
   the depth step (``z_jump_mask``) or the turning normal (``normal_jump_mask``).
2. The edge mask is **closed** (``closing_size``: gaps up to ``closing_size - 1`` pixels are
   filled) and its connected patches with fewer than ``min_patch_size`` pixels are removed as
   noise specks (``close_and_filter_edges``).  Filled pixels are edge pixels from here on.
3. Every pixel is classified as **near an edge** or not (``near_edge_mask``).  The radius varies
   smoothly across the image, so the band is wider in some places than in others and differs
   from sample to sample.  The near-edge pixels are dropped.  Their share of the valid pixels
   never exceeds ``p_drop_max``: if it would, the radius is shrunk until it fits.

The only random draw is the smooth noise of the radius (none when ``radius_jitter`` is 0) and,
when the cap binds, the choice among pixels that tie at the cap; both come from the ``rng``
passed to ``apply``.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, ClassVar

import numpy as np
from scipy import ndimage

from data.degradation.base import BaseDegradation, register_degradation
from data.norm.edge_mask import (
    _median_normals,
    close_and_filter_edges,
    distance_to_edge,
    normal_jump_mask,
    plane_edge_mask,
)
from data.norm.edge_mask import unit_normals as _unit_normals
from data.norm.edge_mask import z_jump_mask

__all__ = [
    "EdgeDropoutDegradation",
    "_median_normals",
    "close_and_filter_edges",
    "near_edge_mask",
    "normal_jump_mask",
    "plane_edge_mask",
    "smooth_noise",
    "z_jump_mask",
]

_NOISE_STRIDE = 4
"""The radius noise is drawn on a grid this many times coarser than the image."""
_MIN_SCALE = 1e-3
"""Lower bound of ``1 + radius_jitter * noise``, so the division by it stays finite."""


@register_degradation("edge_dropout")
@dataclass
class EdgeDropoutDegradation(BaseDegradation):
    """Drop the pixels that are close to a depth or normal edge."""

    key: str = "edge_dropout"
    uses_normals: ClassVar[bool] = True

    # --- edge pixels (see ``plane_edge_mask``) ----------------------------
    z_epsilon: float = 0.02
    """Dimensionless.  A neighbour farther than ``z_epsilon * z`` from the tangent plane makes
    an edge pixel (0.02 = 2 cm per metre of depth)."""
    z_step: int = 4
    """Pixels.  Distance to the neighbours the tangent-plane (z) test looks at."""
    normal_thresh: float = 15.0
    """Degrees.  A pixel whose normal differs by more than this from the normal ``normal_step``
    pixels away is an edge pixel."""
    normal_step: int = 4
    """Pixels.  Distance between the two pixels whose normals are compared (not the adjacent
    ones, whose angle is mostly noise)."""
    normal_median: int = 5
    """Pixels, odd.  Median filter size applied to the normals before the normal test."""
    far_side_only: bool = False
    """Mark only the far side of a depth step (the side active stereo loses to occlusion)."""
    fx: float = 570.0
    """Horizontal focal length in pixels (570 ~ 640x480); rescale with the image."""
    fy: float | None = None
    """Vertical focal length in pixels; ``None`` means ``fx``."""
    cx: float | None = None
    """Principal point x in pixels; ``None`` means the image centre."""
    cy: float | None = None
    """Principal point y in pixels; ``None`` means the image centre."""

    # --- closing and speck filter -----------------------------------------
    closing_size: int = 3
    """Pixels, odd.  Side of the closing square: fills edge gaps up to ``closing_size - 1``
    pixels wide, so one noisy pixel does not break a real edge, and links the pieces of an edge
    into one patch.  1 means no closing."""
    min_patch_size: int = 40
    """Pixels.  Patches of connected edge pixels (after closing) smaller than this are removed
    as specks."""

    # --- closeness to an edge ---------------------------------------------
    radius: float = 4.0
    """Pixels.  A pixel at most this far from an edge pixel is dropped."""
    radius_jitter: float = 0.5
    """Relative variation of the radius across the image: ``radius * (1 + jitter * noise)``."""
    radius_noise_sigma: float = 16.0
    """Pixels.  Correlation length of that variation."""

    p_drop_max: float = 0.3
    """Upper bound on the dropped share of the valid pixels (the radius shrinks to fit)."""

    def __post_init__(self) -> None:
        if self.z_step < 1 or self.normal_step < 1:
            raise ValueError("z_step and normal_step must be >= 1")
        if self.closing_size < 1 or self.closing_size % 2 == 0:
            raise ValueError(f"closing_size must be odd and >= 1, got {self.closing_size}")
        if self.min_patch_size < 0:
            raise ValueError(f"min_patch_size must be >= 0, got {self.min_patch_size}")
        if self.radius < 0:
            raise ValueError(f"radius must be >= 0, got {self.radius}")
        if not 0.0 <= self.p_drop_max <= 1.0:
            raise ValueError(f"p_drop_max must be in [0, 1], got {self.p_drop_max}")

    def apply(
        self, gt: np.ndarray, normal: np.ndarray, rng: np.random.Generator
    ) -> Mapping[str, Any]:
        z = gt.astype(np.float64)
        valid = np.isfinite(z) & (z > 0)
        n_valid = max(int(valid.sum()), 1)
        norm = _unit_normals(normal, z.shape)

        # 1. edge pixels
        h, w = z.shape
        edge = plane_edge_mask(
            z,
            norm,
            valid,
            fx=self.fx,
            fy=self.fx if self.fy is None else self.fy,
            cx=(w - 1) / 2.0 if self.cx is None else self.cx,
            cy=(h - 1) / 2.0 if self.cy is None else self.cy,
            z_epsilon=self.z_epsilon,
            z_step=self.z_step,
            normal_thresh=self.normal_thresh,
            normal_step=self.normal_step,
            normal_median=self.normal_median,
            far_side_only=self.far_side_only,
        )
        edge_raw = edge

        # 2. fill gaps, drop specks
        edge, n_patches = close_and_filter_edges(
            edge, valid, self.min_patch_size, self.closing_size
        )

        # 3. band around the edges
        hole, radius_used = near_edge_mask(
            edge,
            valid,
            self.radius,
            rng,
            self.radius_jitter,
            self.radius_noise_sigma,
            self.p_drop_max,
        )

        sparse = np.where(hole | ~valid, 0.0, z).astype(np.float32)
        return {
            "sparse_depth": sparse,
            "lr_depth": sparse.copy(),  # full-resolution sensor: same holed map
            "meta": {
                "edge_fraction_raw": float(edge_raw.sum() / n_valid),
                "edge_fraction": float(edge.sum() / n_valid),
                "n_patches": n_patches,
                "hole_fraction": float(hole.sum() / n_valid),
                "radius_used": float(radius_used),
                "calibrated": False,
            },
        }


def smooth_noise(shape: tuple[int, int], rng: np.random.Generator, sigma: float) -> np.ndarray:
    """Smooth noise of zero mean and unit std, correlation length ``sigma`` pixels.

    White noise is drawn on a grid ``_NOISE_STRIDE`` times coarser than the image, filtered
    with a Gaussian of ``sigma / _NOISE_STRIDE`` grid pixels and upsampled bilinearly: a smooth
    field needs no full-resolution filter.
    """
    h, w = shape
    s = _NOISE_STRIDE
    low = (max(-(-h // s), 2), max(-(-w // s), 2))  # ceil, at least 2 for the interpolation
    field = ndimage.gaussian_filter(rng.standard_normal(low), max(sigma / s, 0.7))
    field = ndimage.zoom(field, (h / low[0], w / low[1]), order=1)
    field = field - field.mean()
    return field / max(float(field.std()), 1e-9)


def near_edge_mask(
    edge: np.ndarray,
    valid: np.ndarray,
    radius: float,
    rng: np.random.Generator | None = None,
    radius_jitter: float = 0.0,
    radius_noise_sigma: float = 16.0,
    p_drop_max: float = 1.0,
) -> tuple[np.ndarray, float]:
    """Classify every pixel as close to an edge (True) or not.

    A pixel is close when its distance to the nearest edge pixel is at most the local radius,
    ``radius * (1 + radius_jitter * noise)`` (``noise`` is smooth noise of unit std and length
    ``radius_noise_sigma`` drawn from ``rng``, see ``smooth_noise``; ``radius_jitter = 0``
    gives a constant radius and needs no ``rng``).  Equivalently the pixel needs a base radius
    ``distance / (1 + radius_jitter * noise)`` and is close when that is at most ``radius``.
    Only valid pixels can be close.

    If the close pixels would exceed ``p_drop_max`` of the valid pixels, only the ones that
    need the smallest base radius stay (the base radius shrinks); pixels that tie at the
    cut-off (e.g. all edge pixels, which need 0) are chosen at random.

    Returns the bool (H, W) mask and the base radius that was used.
    """
    if not edge.any() or radius <= 0:
        return np.zeros(edge.shape, dtype=bool), 0.0
    needed = distance_to_edge(edge)
    if radius_jitter > 0:
        if rng is None:
            raise ValueError("radius_jitter > 0 needs an rng")
        needed = needed / np.maximum(
            1.0 + radius_jitter * smooth_noise(edge.shape, rng, radius_noise_sigma), _MIN_SCALE
        )
    close = valid & (needed <= radius)

    budget = int(np.floor(p_drop_max * int(valid.sum())))
    if close.sum() <= budget:
        return close, float(radius)

    cutoff = float(np.partition(needed[close], budget)[budget])  # (budget + 1)-th smallest
    keep = close & (needed < cutoff)
    free = budget - int(keep.sum())  # budget left for the pixels tied at the cut-off
    if free > 0:
        tied = np.flatnonzero(close & (needed == cutoff))
        pick = rng.choice(tied, size=free, replace=False) if rng is not None else tied[:free]
        keep.ravel()[pick] = True
    return keep, float(needed[keep].max()) if keep.any() else 0.0
