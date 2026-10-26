"""Missing depth next to edges.

Real active-stereo holes cluster where the geometry changes: along depth steps and where the
normal turns.  This model removes exactly those pixels:

1. A pixel is an **edge pixel** (``edge_method``):
   ``plane`` (default, ``plane_edge_mask``): a neighbour lies off the pixel's tangent plane by
   more than ``tau_plane_rel * z`` (slope-proof: tilted planes are not edges), or the normal
   turns by more than ``tau_n_deg``;
   ``range`` (``edge_pixel_mask``): the relative depth range in a ``window`` x ``window``
   neighbourhood exceeds ``tau_z``, or the normal turns by more than ``tau_n_deg``.
2. Every pixel is classified as **near an edge** or not: its distance to the nearest edge
   pixel is at most ``radius_px`` (``near_edge_mask``).  The radius can vary smoothly across
   the image (``radius_jitter``), so the band is wider in some places than in others and
   differs from sample to sample.
3. The near-edge pixels are dropped.  Their share of the valid pixels never exceeds
   ``max_drop_frac``: if it would, the radius is shrunk until it fits.

All randomness comes from the ``rng`` passed to ``apply``.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, ClassVar

import numpy as np
from scipy import ndimage
from scipy.spatial import cKDTree

from data.degradation.base import BaseDegradation, register_degradation

__all__ = [
    "EdgeDropoutDegradation",
    "clean_edge_mask",
    "edge_patches",
    "edge_pixel_mask",
    "fbm",
    "near_edge_mask",
    "plane_edge_mask",
]


@register_degradation("edge_dropout")
@dataclass
class EdgeDropoutDegradation(BaseDegradation):
    """Drop the pixels that are close to a depth or normal edge."""

    key: str = "edge_dropout"
    uses_normals: ClassVar[bool] = True

    # --- edge pixels ------------------------------------------------------
    edge_method: str = "plane"
    """``"plane"``: point-to-plane discontinuity + normal jump (slope-proof, see
    ``plane_edge_mask``); ``"range"``: relative depth range in a window + normal jump."""
    tau_n_deg: float = 15.0
    """A neighbour whose normal differs by more than this (degrees) makes an edge pixel."""
    window: int = 3
    """Odd neighbourhood size used to look for normal (and, for ``range``, depth) variety."""

    # plane method
    tau_plane_rel: float = 0.02
    """A neighbour farther than ``tau_plane_rel * z`` from the tangent plane makes an edge."""
    step_px: int = 4
    """Distance (pixels) to the neighbours the tangent-plane test looks at."""
    normal_smooth_px: int = 5
    """Median filter size applied to the normals before the normal-jump test."""
    normal_step_px: int = 4
    """The normal-jump test compares a normal with the ones this many pixels away (not the
    adjacent ones, whose angle is mostly noise)."""
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

    # range method
    tau_z: float = 0.05
    """A relative depth range ``(max z - min z) / z`` above this makes an edge pixel."""

    # cleaning (both methods)
    min_edge_px: int = 40
    """Groups of connected edge pixels smaller than this are removed as noise specks."""
    close_px: int = 1
    """Closing iterations applied before that, so one noisy pixel does not break a real edge."""

    # --- patches of edge pixels -------------------------------------------
    link_px: float = 6.0
    """Edge pixels closer than this (pixels) belong to the same patch."""
    patch_len_px: tuple[float, float] = (20.0, 200.0)
    """A patch with more edge pixels than its target length is cut along the edge; each patch
    draws its target log-uniformly from this range, so short and long patches appear together."""
    p_drop: float = 0.5
    """Probability that a patch of edge pixels is dropped."""

    # --- closeness to an edge ---------------------------------------------
    radius_px: float = 4.0
    """A pixel at most this far (pixels) from an edge pixel of a dropped patch is dropped."""
    radius_jitter: float = 0.5
    """Relative variation of the radius across the image: ``radius * (1 + jitter * noise)``."""
    noise_sigma_px: float = 16.0
    """Correlation length (pixels) of the coarsest octave of that variation."""
    octaves: int = 3
    """Octaves of the fractal noise (each halves the correlation length)."""

    max_drop_frac: float = 0.3
    """Upper bound on the dropped share of the valid pixels (the radius shrinks to fit)."""

    def __post_init__(self) -> None:
        if self.edge_method not in ("plane", "range"):
            raise ValueError(f"edge_method must be 'plane' or 'range', got {self.edge_method!r}")
        if self.window < 3 or self.window % 2 == 0:
            raise ValueError(f"window must be odd and >= 3, got {self.window}")
        if self.step_px < 1 or self.normal_step_px < 1:
            raise ValueError("step_px and normal_step_px must be >= 1")
        if self.min_edge_px < 0 or self.close_px < 0:
            raise ValueError("min_edge_px and close_px must be >= 0")
        self.patch_len_px = (float(self.patch_len_px[0]), float(self.patch_len_px[1]))
        if not 1.0 <= self.patch_len_px[0] <= self.patch_len_px[1]:
            raise ValueError(f"patch_len_px must satisfy 1 <= lo <= hi, got {self.patch_len_px}")
        if self.link_px < 0:
            raise ValueError(f"link_px must be >= 0, got {self.link_px}")
        if not 0.0 <= self.p_drop <= 1.0:
            raise ValueError(f"p_drop must be in [0, 1], got {self.p_drop}")
        if self.radius_px < 0:
            raise ValueError(f"radius_px must be >= 0, got {self.radius_px}")
        if not 0.0 <= self.max_drop_frac <= 1.0:
            raise ValueError(f"max_drop_frac must be in [0, 1], got {self.max_drop_frac}")

    def apply(
        self, gt: np.ndarray, normal: np.ndarray, rng: np.random.Generator
    ) -> Mapping[str, Any]:
        z = gt.astype(np.float64)
        valid = np.isfinite(z) & (z > 0)
        n_valid = max(int(valid.sum()), 1)
        n = _unit_normals(normal, z.shape)

        if self.edge_method == "plane":
            h, w = z.shape
            edge = plane_edge_mask(
                z,
                n,
                valid,
                fx=self.fx,
                fy=self.fx if self.fy is None else self.fy,
                cx=(w - 1) / 2.0 if self.cx is None else self.cx,
                cy=(h - 1) / 2.0 if self.cy is None else self.cy,
                tau_plane_rel=self.tau_plane_rel,
                step_px=self.step_px,
                tau_n_deg=self.tau_n_deg,
                normal_smooth_px=self.normal_smooth_px,
                normal_step_px=self.normal_step_px,
                far_side_only=self.far_side_only,
            )
        else:
            edge = edge_pixel_mask(z, n, valid, self.tau_n_deg, self.tau_z, self.window)
        edge_raw = edge
        edge = clean_edge_mask(edge, valid, self.min_edge_px, self.close_px)

        # group edge pixels into patches, drop each patch with probability p_drop
        patches = edge_patches(edge, self.link_px, self.patch_len_px, rng)
        n_patches = int(patches.max())
        dropped = rng.random(n_patches + 1) < self.p_drop
        dropped[0] = False
        masked = dropped[patches]  # edge pixels of the dropped patches

        hole, radius_used = near_edge_mask(
            masked,
            valid,
            self.radius_px,
            rng,
            self.radius_jitter,
            self.noise_sigma_px,
            self.octaves,
            self.max_drop_frac,
        )

        sparse = np.where(hole | ~valid, 0.0, z).astype(np.float32)
        return {
            "sparse_depth": sparse,
            "lr_depth": sparse.copy(),  # full-resolution sensor: same holed map
            "meta": {
                "edge_method": self.edge_method,
                "edge_fraction_raw": float(edge_raw.sum() / n_valid),
                "edge_fraction": float(edge.sum() / n_valid),
                "n_patches": n_patches,
                "n_dropped": int(dropped.sum()),
                "patch_edge_px_p10_p50_p90": _percentiles(np.bincount(patches.ravel())[1:]),
                "hole_fraction": float(hole.sum() / n_valid),
                "radius_used": float(radius_used),
                "calibrated": False,
            },
        }


def _unit_normals(normal: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    n = np.asarray(normal, dtype=np.float64)
    if n.shape[:2] != shape or n.shape[2:] != (3,):
        raise ValueError(f"normal {n.shape} does not match depth {shape}")
    n = np.where(np.isfinite(n), n, 0.0)
    norm = np.linalg.norm(n, axis=-1, keepdims=True)
    return np.where(norm > 1e-6, n / np.maximum(norm, 1e-6), 0.0)


def fbm(shape: tuple[int, int], rng: np.random.Generator, sigma: float, octaves: int) -> np.ndarray:
    """Fractal noise of unit std: Gaussian-filtered white noise summed over octaves.

    Octave ``o`` has correlation length ``sigma / 2**o`` and amplitude ``0.5**o``, so
    structure exists at several scales.
    """
    out = np.zeros(shape)
    for o in range(max(octaves, 1)):
        field = ndimage.gaussian_filter(rng.standard_normal(shape), max(sigma / 2**o, 0.7))
        out += (0.5**o) * field / max(float(field.std()), 1e-9)
    return out / max(float(out.std()), 1e-9)


def edge_pixel_mask(
    z: np.ndarray, n: np.ndarray, valid: np.ndarray, tau_n_deg: float, tau_z: float, window: int
) -> np.ndarray:
    """Valid pixels whose ``window`` x ``window`` neighbourhood has a depth or normal jump."""
    big = float(np.max(z, initial=1.0)) * 4.0 + 1.0
    zmax = ndimage.maximum_filter(np.where(valid, z, 0.0), size=window, mode="nearest")
    zmin = ndimage.minimum_filter(np.where(valid, z, big), size=window, mode="nearest")
    with np.errstate(invalid="ignore", divide="ignore"):
        depth_jump = (zmax - zmin) / np.where(valid, z, 1.0) > tau_z

    return valid & (depth_jump | _normal_jump(n, tau_n_deg, window))


def _normal_jump(n: np.ndarray, tau_n_deg: float, window: int) -> np.ndarray:
    """Pixels with a neighbour (in the ``window``) whose normal differs by more than the angle."""
    r = window // 2
    h, w = n.shape[:2]
    has_n = np.any(n != 0.0, axis=-1)
    pn = np.pad(n, ((r, r), (r, r), (0, 0)))
    pv = np.pad(has_n, r)
    min_cos = np.ones((h, w))
    for dy in range(-r, r + 1):
        for dx in range(-r, r + 1):
            if dy == 0 and dx == 0:
                continue
            nb = pn[r + dy : r + dy + h, r + dx : r + dx + w]
            ok = pv[r + dy : r + dy + h, r + dx : r + dx + w] & has_n
            min_cos = np.minimum(min_cos, np.where(ok, np.sum(n * nb, axis=-1), 1.0))
    return min_cos < np.cos(np.deg2rad(tau_n_deg))


def _normal_jump_at(n: np.ndarray, tau_n_deg: float, step: int) -> np.ndarray:
    """Pixels whose normal differs by more than the angle from one ``step`` pixels away.

    The 8 neighbours at exactly ``step`` pixels (axes and diagonals) are compared, not the
    adjacent pixels: the angle between normals a few pixels apart is a real change of surface
    orientation, while between adjacent pixels it is mostly noise.
    """
    h, w = n.shape[:2]
    has_n = np.any(n != 0.0, axis=-1)
    pn = np.pad(n, ((step, step), (step, step), (0, 0)))
    pv = np.pad(has_n, step)
    min_cos = np.ones((h, w))
    for dy in (-step, 0, step):
        for dx in (-step, 0, step):
            if dy == 0 and dx == 0:
                continue
            sl = (slice(step + dy, step + dy + h), slice(step + dx, step + dx + w))
            ok = pv[sl] & has_n
            min_cos = np.minimum(min_cos, np.where(ok, np.sum(n * pn[sl], axis=-1), 1.0))
    return min_cos < np.cos(np.deg2rad(tau_n_deg))


def clean_edge_mask(
    edge: np.ndarray, valid: np.ndarray, min_px: int, close_px: int = 1
) -> np.ndarray:
    """Keep only long edges: drop connected groups of edge pixels with fewer than ``min_px``.

    Real edges (steps, nosings, outlines) are long connected curves; false positives on flat
    surfaces are isolated specks.  Groups are found on a binary closing of ``close_px``
    iterations, which reconnects a real edge that one noisy pixel broke, but only the original
    edge pixels are returned: the filter never adds pixels.
    """
    if min_px <= 1 and close_px <= 0:
        return edge
    grouped = edge
    if close_px > 0 and edge.any():
        # pad by replication: scipy's closing would otherwise erode the image border
        c = close_px
        closed = ndimage.binary_closing(
            np.pad(edge, c, mode="edge"), structure=np.ones((3, 3), bool), iterations=c
        )
        grouped = closed[c:-c, c:-c] & valid
    labels, count = ndimage.label(grouped, structure=np.ones((3, 3), dtype=int))
    if count == 0:
        return edge
    size = np.bincount(labels.ravel(), minlength=count + 1)
    keep = size >= min_px
    keep[0] = False
    return keep[labels] & edge


def edge_patches(
    edge: np.ndarray,
    link_px: float,
    patch_len_px: tuple[float, float] | None,
    rng: np.random.Generator,
) -> np.ndarray:
    """Group edge pixels into patches; returns an (H, W) label map (0 = not an edge pixel).

    Edge pixels closer than ``link_px`` (8-connected after linking) share a patch.  On real
    scenes the edges of a whole image link up into one network, so a patch with more edge
    pixels than its target length is cut along the edge into pieces (nearest of random seeds
    placed on its edge pixels).  Each patch draws its target log-uniformly from
    ``patch_len_px`` (``None``: no cutting), giving short and long patches side by side.
    """
    eight = np.ones((3, 3), dtype=int)
    if not edge.any():
        return np.zeros(edge.shape, dtype=np.int64)
    near = ndimage.distance_transform_edt(~edge) <= link_px / 2.0 if link_px > 1.0 else edge
    labels = np.where(edge, ndimage.label(near, structure=eight)[0], 0).astype(np.int64)
    labels = _relabel(labels)
    if patch_len_px is None or labels.max() == 0:
        return labels

    count = int(labels.max())
    n_edge = np.bincount(labels.ravel(), minlength=count + 1)
    lo, hi = patch_len_px
    target = np.exp(rng.uniform(np.log(lo), np.log(hi), size=count + 1))
    flat = labels.ravel()
    order = np.argsort(flat, kind="stable")
    bounds = np.concatenate([[0], np.cumsum(n_edge)])
    width = labels.shape[1]
    out = flat.copy()
    next_id = count + 1
    for lid in np.flatnonzero(n_edge > 1.5 * target):
        if lid == 0:
            continue
        pix = order[bounds[lid] : bounds[lid + 1]]
        k = int(np.clip(round(n_edge[lid] / target[lid]), 2, pix.size))
        seeds = rng.choice(pix, size=k, replace=False)
        pts = np.stack([pix // width, pix % width], axis=1)
        sp = np.stack([seeds // width, seeds % width], axis=1)
        _, nearest = cKDTree(sp).query(pts)
        out[pix] = next_id + nearest
        next_id += k
    return _relabel(out.reshape(labels.shape))


def _relabel(labels: np.ndarray) -> np.ndarray:
    """Make ids consecutive ``1..N`` keeping 0 as background."""
    ids, inv = np.unique(labels, return_inverse=True)
    inv = inv.reshape(labels.shape)
    return (inv if ids[0] == 0 else inv + 1).astype(np.int64) * (labels > 0)


def _percentiles(values: np.ndarray) -> list[float]:
    if values.size == 0:
        return [0.0, 0.0, 0.0]
    return [float(v) for v in np.percentile(values, [10, 50, 90])]


def plane_edge_mask(
    z: np.ndarray,
    n: np.ndarray,
    valid: np.ndarray,
    *,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    tau_plane_rel: float = 0.02,
    step_px: int = 4,
    tau_n_deg: float = 15.0,
    normal_smooth_px: int = 5,
    normal_step_px: int = 4,
    far_side_only: bool = False,
) -> np.ndarray:
    """Edge pixels from point-to-plane discontinuities (slope-proof) and normal jumps.

    Each valid pixel ``i`` is back-projected to a 3-D point ``P_i`` with unit normal ``n_i``.
    For the 8 neighbours ``j`` at ``step_px`` the distance of ``P_j`` from the tangent plane of
    ``i`` is ``|n_i . (P_j - P_i)|`` (and the same with ``n_j``; the larger counts).  On a
    plane it is ~0 whatever the tilt, so slanted floors and treads are not edges; across a
    depth step it is large.  ``i`` is an edge pixel if that distance exceeds
    ``tau_plane_rel * z_i`` for any neighbour, or if its (median-smoothed over
    ``normal_smooth_px``) normal differs by more than ``tau_n_deg`` from the normal
    ``normal_step_px`` away (creases and nosings).  Both sides of a step are marked, unless
    ``far_side_only``: then the
    depth-step pixels must be farther than the neighbour that caused the jump (the side an
    active-stereo sensor loses to occlusion); normal jumps are kept on both sides.
    """
    h, w = z.shape
    if normal_smooth_px > 1:
        n = np.stack(
            [ndimage.median_filter(n[..., c], size=normal_smooth_px) for c in range(3)], axis=-1
        )
        norm = np.linalg.norm(n, axis=-1, keepdims=True)
        n = np.where(norm > 1e-6, n / np.maximum(norm, 1e-6), 0.0)

    vs, us = np.mgrid[0:h, 0:w].astype(np.float64)
    zz = np.where(valid, z, 0.0)
    pts = np.stack([(us - cx) * zz / fx, (vs - cy) * zz / fy, zz], axis=-1)

    s = int(step_px)
    pp = np.pad(pts, ((s, s), (s, s), (0, 0)))
    pn = np.pad(n, ((s, s), (s, s), (0, 0)))
    pvld = np.pad(valid, s)
    plane = np.zeros((h, w), dtype=bool)
    for dy in (-s, 0, s):
        for dx in (-s, 0, s):
            if dy == 0 and dx == 0:
                continue
            sl = (slice(s + dy, s + dy + h), slice(s + dx, s + dx + w))
            diff = pp[sl] - pts
            d = np.maximum(
                np.abs(np.sum(n * diff, axis=-1)), np.abs(np.sum(pn[sl] * diff, axis=-1))
            )
            jump = valid & pvld[sl] & (d > tau_plane_rel * zz)
            if far_side_only:
                jump &= zz > pp[sl][..., 2]
            plane |= jump
    return valid & (plane | _normal_jump_at(n, tau_n_deg, int(normal_step_px)))


def near_edge_mask(
    edge: np.ndarray,
    valid: np.ndarray,
    radius_px: float,
    rng: np.random.Generator | None = None,
    jitter: float = 0.0,
    sigma_px: float = 16.0,
    octaves: int = 3,
    max_frac: float = 1.0,
) -> tuple[np.ndarray, float]:
    """Classify every pixel as close to an edge (True) or not.

    A pixel is close when its distance to the nearest edge pixel is at most the local radius,
    ``radius_px * (1 + jitter * noise)`` (``noise`` is smooth fractal noise of unit std drawn
    from ``rng``; ``jitter = 0`` gives a constant radius and needs no ``rng``).  Only valid
    pixels can be close.  If the close pixels would exceed ``max_frac`` of the valid pixels,
    the radius is shrunk (the pixels needing the smallest radius stay, ties at random) until
    they fit.

    Returns the bool (H, W) mask and the base radius that was used.
    """
    if not edge.any() or radius_px <= 0:
        return np.zeros(edge.shape, dtype=bool), 0.0
    dist = ndimage.distance_transform_edt(~edge)
    if jitter > 0:
        if rng is None:
            raise ValueError("jitter > 0 needs an rng")
        scale = np.maximum(1.0 + jitter * fbm(edge.shape, rng, sigma_px, octaves), 0.0)
    else:
        scale = np.ones(edge.shape)

    # base radius each pixel would need to be close; edge pixels (distance 0) are always close
    with np.errstate(divide="ignore", invalid="ignore"):
        need = np.where(dist == 0, 0.0, np.where(scale > 0, dist / scale, np.inf))
    close = valid & (need <= radius_px)

    budget = int(np.floor(max_frac * int(valid.sum())))
    radius = float(radius_px)
    if close.sum() > budget:
        # keep the pixels that need the smallest radius; ties (e.g. all edge pixels) at random
        idx = np.flatnonzero(close)
        tie = rng.random(idx.size) if rng is not None else np.arange(idx.size)
        keep = idx[np.lexsort((tie, need.ravel()[idx]))[:budget]]
        close = np.zeros(edge.shape, dtype=bool)
        close.ravel()[keep] = True
        radius = float(need.ravel()[keep].max()) if keep.size else 0.0
    return close, radius
