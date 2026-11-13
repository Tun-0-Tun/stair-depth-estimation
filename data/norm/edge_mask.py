"""Depth / normal edge masks.

What an "edge" means in this project: a pixel where either a depth step
(``z_jump_mask``) or a normal crease (``normal_jump_mask``) fires.  Both tests take
unit normals as input, so the operator lives next to ``data/norm/*.py`` rather than in
any degradation or metric module -- it is pure geometry derived from depth and normals,
and the degradation (``edge_dropout``) and the boundary-error / step-height metrics all
consume it unchanged.  See ``data/degradation/edge_dropout.py`` for the "band around
the edge" step that turns a mask into dropped pixels.

Two small helpers are co-located so the whole edge toolkit is one import:

``distance_to_edge``
    Distance transform of ``~edge``, in pixels, used by both ``near_edge_mask`` (inside
    the degradation) and the DBE metric.

``edge_normal_direction``
    Unit image-plane gradient of a depth map, used by the step-height metric to orient
    its cross-edge sampling.
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage

__all__ = [
    "close_and_filter_edges",
    "distance_to_edge",
    "edge_normal_direction",
    "normal_jump_mask",
    "plane_edge_mask",
    "unit_normals",
    "z_jump_mask",
]


def unit_normals(normal: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    """Re-normalise a ``(H, W, 3)`` normal map and zero invalid rows.

    Kept public (as ``unit_normals``) because tests and the degradation adapter use it
    to prepare the input of the edge tests.
    """
    n = np.asarray(normal, dtype=np.float64)
    if n.shape[:2] != shape or n.shape[2:] != (3,):
        raise ValueError(f"normal {n.shape} does not match depth {shape}")
    n = np.where(np.isfinite(n), n, 0.0)
    norm = np.linalg.norm(n, axis=-1, keepdims=True)
    return np.where(norm > 1e-6, n / np.maximum(norm, 1e-6), 0.0)


def _ring(step: int) -> list[tuple[int, int]]:
    """The 8 offsets (dy, dx) at exactly ``step`` pixels: axes and diagonals."""
    return [(dy, dx) for dy in (-step, 0, step) for dx in (-step, 0, step) if (dy, dx) != (0, 0)]


def _median_normals(n: np.ndarray, size: int) -> np.ndarray:
    """Median filter of ``size`` pixels on each normal component, renormalised (1 = unchanged)."""
    if size <= 1:
        return n
    n = np.stack([ndimage.median_filter(n[..., c], size=size) for c in range(3)], axis=-1)
    norm = np.linalg.norm(n, axis=-1, keepdims=True)
    return np.where(norm > 1e-6, n / np.maximum(norm, 1e-6), 0.0)


def close_and_filter_edges(
    edge: np.ndarray, valid: np.ndarray, min_patch_size: int, closing_size: int = 3
) -> tuple[np.ndarray, int]:
    """Fill small gaps in the edge mask, then keep only the long edges.

    Real edges (steps, nosings, outlines) are long connected curves; false positives on flat
    surfaces are isolated specks.  The mask is closed with a ``closing_size`` square (a gap up
    to ``closing_size - 1`` pixels wide is filled, so one noisy pixel does not break a real
    edge in two), its 8-connected patches are counted, and the patches with fewer than
    ``min_patch_size`` pixels (filled pixels included) are removed.  Filled pixels stay in the
    mask.

    Returns the cleaned bool mask and the number of patches kept.
    """
    closed = edge
    if closing_size > 1 and edge.any():
        # pad by replication: scipy's closing would otherwise erode the image border
        r = closing_size // 2
        closed = ndimage.binary_closing(
            np.pad(edge, r, mode="edge"), structure=np.ones((closing_size, closing_size), bool)
        )
        closed = closed[r:-r, r:-r] & valid
    patch, n_patches = ndimage.label(closed, structure=np.ones((3, 3), dtype=int))
    big = np.bincount(patch.ravel(), minlength=n_patches + 1) >= min_patch_size
    big[0] = False  # patch 0 is the background
    return big[patch], int(big.sum())


def plane_edge_mask(
    z: np.ndarray,
    n: np.ndarray,
    valid: np.ndarray,
    *,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    z_epsilon: float = 0.02,
    z_step: int = 4,
    normal_thresh: float = 15.0,
    normal_step: int = 4,
    normal_median: int = 5,
    far_side_only: bool = False,
) -> np.ndarray:
    """Edge pixels: a depth step (``z_jump_mask``) or a turning normal (``normal_jump_mask``).

    The unit normals ``n`` are median-filtered over ``normal_median`` pixels first; both tests
    use the filtered normals.
    """
    n = _median_normals(n, normal_median)
    z_jump = z_jump_mask(
        z,
        n,
        valid,
        fx=fx,
        fy=fy,
        cx=cx,
        cy=cy,
        z_epsilon=z_epsilon,
        z_step=z_step,
        far_side_only=far_side_only,
    )
    normal_jump = normal_jump_mask(n, normal_thresh, normal_step)
    return valid & (z_jump | normal_jump)


def z_jump_mask(
    z: np.ndarray,
    n: np.ndarray,
    valid: np.ndarray,
    *,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    z_epsilon: float = 0.02,
    z_step: int = 4,
    far_side_only: bool = False,
) -> np.ndarray:
    """Pixels with a neighbour off their tangent plane: depth steps, slope-proof.

    Each valid pixel ``i`` is back-projected to a 3-D point ``P_i`` with unit normal ``n_i``.
    For the 8 neighbours ``j`` at ``z_step`` the distance of ``P_j`` from the tangent plane of
    ``i`` is ``|n_i . (P_j - P_i)|`` (and the same with ``n_j``; the larger counts).  On a
    plane it is ~0 whatever the tilt, so slanted floors and treads are not edges; across a
    depth step it is large.  ``i`` is marked if that distance exceeds ``z_epsilon * z_i`` for
    any neighbour.  Both sides of a step are marked, unless ``far_side_only``: then only the
    pixels farther than the neighbour that caused the jump (the side an active-stereo sensor
    loses to occlusion).  ``n`` are the (already smoothed) unit normals.
    """
    h, w = z.shape

    # 1. Project to 3-D
    vs, us = np.mgrid[0:h, 0:w].astype(np.float64)
    zz = np.where(valid, z, 0.0)
    pts = np.stack([(us - cx) * zz / fx, (vs - cy) * zz / fy, zz], axis=-1)

    # 2. A neighbour off the tangent plane
    s = int(z_step)
    pp = np.pad(pts, ((s, s), (s, s), (0, 0)))
    pn = np.pad(n, ((s, s), (s, s), (0, 0)))
    pvld = np.pad(valid, s)
    plane = np.zeros((h, w), dtype=bool)
    for dy, dx in _ring(s):
        sl = (slice(s + dy, s + dy + h), slice(s + dx, s + dx + w))
        diff = pp[sl] - pts
        d = np.maximum(np.abs(np.sum(n * diff, axis=-1)), np.abs(np.sum(pn[sl] * diff, axis=-1)))
        jump = valid & pvld[sl] & (d > z_epsilon * zz)
        if far_side_only:
            jump &= zz > pp[sl][..., 2]
        plane |= jump
    return plane


def normal_jump_mask(
    n: np.ndarray, normal_thresh: float = 15.0, normal_step: int = 4
) -> np.ndarray:
    """Pixels whose normal turns: creases and nosings.

    A pixel is marked if its normal differs by more than ``normal_thresh`` degrees from the
    normal of any of the 8 pixels exactly ``normal_step`` pixels away (axes and diagonals).
    Adjacent pixels are not compared: the angle between normals a few pixels apart is a real
    change of surface orientation, while between adjacent pixels it is mostly noise.  Pixels
    without a normal (zero vector) never take part.  ``n`` are the (already smoothed) unit
    normals.
    """
    h, w = n.shape[:2]
    step = int(normal_step)
    has_n = np.any(n != 0.0, axis=-1)
    pn = np.pad(n, ((step, step), (step, step), (0, 0)))
    pv = np.pad(has_n, step)
    min_cos = np.ones((h, w))
    for dy, dx in _ring(step):
        sl = (slice(step + dy, step + dy + h), slice(step + dx, step + dx + w))
        ok = pv[sl] & has_n
        min_cos = np.minimum(min_cos, np.where(ok, np.sum(n * pn[sl], axis=-1), 1.0))
    return min_cos < np.cos(np.deg2rad(normal_thresh))


def distance_to_edge(edge: np.ndarray) -> np.ndarray:
    """Per-pixel distance to the nearest edge pixel, in pixels.

    Thin wrapper around ``scipy.ndimage.distance_transform_edt(~edge)``.  Returns an
    all-``inf`` array (same shape as ``edge``, float64) when there is no edge pixel, so
    callers must decide what to do with no edges rather than silently reading a zero.
    """
    edge = np.asarray(edge, dtype=bool)
    if not edge.any():
        return np.full(edge.shape, np.inf, dtype=np.float64)
    return ndimage.distance_transform_edt(~edge).astype(np.float64)


def edge_normal_direction(
    depth: np.ndarray,
    valid: np.ndarray | None = None,
    eps: float = 1e-8,
) -> tuple[np.ndarray, np.ndarray]:
    """Unit image-plane gradient of a depth map.

    Returns ``(n, has_dir)``.  ``n`` is ``(H, W, 2)`` float64 with components
    ``(n_y, n_x)`` (``np.gradient`` axis order), i.e. the direction in which depth grows
    fastest.  ``has_dir`` is the ``(H, W)`` bool mask of pixels where ``||grad|| >= eps``
    and (if ``valid`` is passed) the pixel is valid.  Pixels outside ``has_dir`` have
    ``n = 0`` and must not be used for cross-edge sampling.
    """
    d = np.asarray(depth, dtype=np.float64)
    if d.ndim != 2:
        raise ValueError(f"depth must be (H, W); got shape {d.shape}")
    gy, gx = np.gradient(d)
    mag = np.sqrt(gy * gy + gx * gx)
    has_dir = mag >= eps
    if valid is not None:
        has_dir &= np.asarray(valid, dtype=bool)
    safe = np.where(has_dir, mag, 1.0)
    n = np.stack([gy / safe, gx / safe], axis=-1)
    n = np.where(has_dir[..., None], n, 0.0)
    return n, has_dir
