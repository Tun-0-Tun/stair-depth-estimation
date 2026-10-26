"""Patch-wise missing depth: whole blocks of pixels that share a geometric property vanish.

Real active-stereo holes are not independent pixels.  They are patches tied to local
geometry: a slanted tread the matcher cannot lock onto, a band along a depth edge.  This
model cuts the image into such patches and removes some of them as a whole.

Two patch generators (``patch_sources``), each producing its own label map:

``surface``
    Neighbouring pixels are linked when their normals differ by less than ``tau_n_deg``
    *and* their relative depth differs by less than ``tau_z``.  Connected components are
    surfaces of similar normal and depth.  Components larger than the random cell size are
    cut into domain-warped Voronoi cells so a floor becomes several irregular blobs.
``edge``
    A pixel is an edge pixel when, inside a ``window`` x ``window`` neighbourhood, the
    relative depth range exceeds ``tau_z`` or the normal turns by more than ``tau_n_deg``.
    Edge pixels are grown by a smoothly varying radius into uneven blobs, and connected
    blobs are the patches.

Per map: patches below ``min_area_px`` are merged into the nearest larger patch (or
discarded), then each patch is dropped with a probability that grows with the viewing angle
and the depth.  The border of the dropped region is made fractal (multi-octave noise),
speckled (probabilistic transition of width ``fade_px``) and the interior gets a few valid
islands, so shapes are organic rather than square or line-like.  Hole = union over the
sources.  All randomness comes from the ``rng`` passed to ``apply``.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, ClassVar

import numpy as np
from scipy import ndimage
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree

from data.degradation.axial import get_theta_map
from data.degradation.base import BaseDegradation, register_degradation

__all__ = [
    "BlockDropoutDegradation",
    "edge_patches",
    "edge_pixel_mask",
    "merge_small_patches",
    "surface_patches",
]

_SOURCES = ("surface", "edge")


@register_degradation("block_dropout")
@dataclass
class BlockDropoutDegradation(BaseDegradation):
    """Drop whole patches of similar normal/depth and patches built from edge pixels."""

    key: str = "block_dropout"
    uses_normals: ClassVar[bool] = True

    patch_sources: tuple[str, ...] = ("surface", "edge")
    """Any of ``"surface"``, ``"edge"``."""

    # --- shared geometry thresholds ---------------------------------------
    tau_n_deg: float = 15.0
    """Normals further apart than this (degrees) break a surface / mark an edge pixel."""
    tau_z: float = 0.05
    """Relative depth change ``|dz| / z`` that breaks a surface / marks an edge pixel."""

    # --- surface patches ----------------------------------------------------
    cell_px: tuple[float, float] = (24.0, 64.0)
    """Range of the Voronoi cell size (pixels) that cuts large surfaces; drawn once per image."""
    warp_px: float = 12.0
    """Amplitude (pixels) of the random warp that bends the Voronoi cell borders."""

    # --- edge patches -----------------------------------------------------
    window: int = 3
    """Odd neighbourhood size used to look for depth / normal variety around a pixel."""
    edge_width_px: float = 6.0
    """Largest radius (pixels) edge pixels are grown by; the radius varies from 0.3x to 1x."""
    edge_sigma_px: float = 8.0
    """Correlation length (pixels) of that radius variation."""

    # --- small-patch handling ---------------------------------------------
    min_area_px: int = 64
    """Patches smaller than this are merged into the nearest larger patch."""
    merge_dist_px: float = 6.0
    """A small patch merges only if a larger patch is this close (pixels)."""
    discard_small: bool = False
    """Small patches with no larger patch in reach are removed instead of kept."""

    # --- which patches go ---------------------------------------------------
    p0: float = 0.1
    """Base probability that a patch is dropped."""
    p_theta: float = 0.3
    """Added probability times ``1 - cos(theta)``; theta = normal vs optical axis."""
    p_depth: float = 0.1
    """Added probability times ``z / z_ref`` (far surfaces drop more)."""
    z_ref: float = 4.0
    """Depth (m) at which the depth term equals ``p_depth``."""
    p_max: float = 0.9

    # --- boundary roughening ------------------------------------------------
    grow_px: int = 3
    """Each dropped patch grows or shrinks by a random integer in ``[-grow_px, grow_px]``."""
    ragged_px: float = 5.0
    """Amplitude (pixels) of the fractal random offset applied to the border."""
    ragged_sigma_px: float = 8.0
    """Correlation length (pixels) of the coarsest octave of that offset."""
    octaves: int = 3
    """Octaves of the fractal noise (each halves the correlation length)."""
    fade_px: float = 3.0
    """Width (pixels) of the speckled transition at a hole border; 0 gives a hard border."""
    island_frac: float = 0.05
    """Fraction of a hole's interior left valid as islands (0 = holes are solid)."""

    def __post_init__(self) -> None:
        self.patch_sources = tuple(self.patch_sources)
        bad = set(self.patch_sources) - set(_SOURCES)
        if bad or not self.patch_sources:
            raise ValueError(f"patch_sources must be a non-empty subset of {_SOURCES}, got {bad}")
        if self.window < 3 or self.window % 2 == 0:
            raise ValueError(f"window must be odd and >= 3, got {self.window}")

    def apply(
        self, gt: np.ndarray, normal: np.ndarray, rng: np.random.Generator
    ) -> Mapping[str, Any]:
        z = gt.astype(np.float64)
        valid = np.isfinite(z) & (z > 0)
        n = _unit_normals(normal, z.shape)
        theta = get_theta_map(n, z.shape)

        hole = np.zeros(z.shape, dtype=bool)
        n_patches = n_merged = n_dropped = 0
        per_source: dict[str, int] = {}
        for source in self.patch_sources:
            if source == "surface":
                lo, hi = self.cell_px
                labels = surface_patches(
                    z,
                    n,
                    valid,
                    self.tau_n_deg,
                    self.tau_z,
                    float(rng.uniform(lo, hi)),
                    rng,
                    self.warp_px,
                    self.octaves,
                )
            else:
                labels = edge_patches(
                    z,
                    n,
                    valid,
                    self.tau_n_deg,
                    self.tau_z,
                    self.window,
                    self.edge_width_px,
                    rng,
                    self.edge_sigma_px,
                    self.octaves,
                )
            labels, merged = merge_small_patches(
                labels, self.min_area_px, self.merge_dist_px, self.discard_small
            )
            dropped = self._pick_dropped(labels, z, theta, rng)
            hole |= self._roughen(labels, dropped, rng)
            n_patches += int(labels.max())
            n_merged += merged
            n_dropped += int(dropped.sum())
            per_source[source] = int(dropped.sum())

        hole &= valid
        sparse = np.where(hole | ~valid, 0.0, z).astype(np.float32)
        return {
            "sparse_depth": sparse,
            "lr_depth": sparse.copy(),  # full-resolution sensor: same holed map
            "meta": {
                "n_patches": n_patches,
                "n_merged": n_merged,
                "n_dropped": n_dropped,
                "n_dropped_by_source": per_source,
                "hole_fraction": float(hole.sum() / max(int(valid.sum()), 1)),
                "calibrated": False,
            },
        }

    def _pick_dropped(
        self, labels: np.ndarray, z: np.ndarray, theta: np.ndarray, rng: np.random.Generator
    ) -> np.ndarray:
        """Bool vector over label ids ``0..N`` (index 0 is never dropped)."""
        count = int(labels.max()) + 1
        flat = labels.ravel()
        theta_ok = np.isfinite(theta).ravel()
        cos_sum = np.bincount(
            flat,
            weights=np.where(theta_ok, 1.0 - np.cos(np.nan_to_num(theta)).ravel(), 0.0),
            minlength=count,
        )
        cos_cnt = np.bincount(flat, weights=theta_ok.astype(float), minlength=count)
        z_sum = np.bincount(flat, weights=z.ravel(), minlength=count)
        area = np.bincount(flat, minlength=count)
        with np.errstate(invalid="ignore", divide="ignore"):
            one_minus_cos = np.where(cos_cnt > 0, cos_sum / cos_cnt, 0.0)
            z_mean = np.where(area > 0, z_sum / area, 0.0)
        p = np.clip(
            self.p0 + self.p_theta * one_minus_cos + self.p_depth * z_mean / self.z_ref,
            0.0,
            self.p_max,
        )
        dropped = rng.random(count) < p
        dropped[0] = False
        return dropped

    def _roughen(
        self, labels: np.ndarray, dropped: np.ndarray, rng: np.random.Generator
    ) -> np.ndarray:
        """Hole mask from the dropped patches, with grown/shrunk and ragged borders."""
        kept = np.where(dropped[labels], labels, 0)
        core = kept > 0
        if not core.any():
            return core
        k = rng.integers(-self.grow_px, self.grow_px + 1, size=dropped.size).astype(np.float64)
        noise = fbm(core.shape, rng, self.ragged_sigma_px, self.octaves)

        # signed distance to the core: > 0 outside, < 0 inside
        d_out, idx = ndimage.distance_transform_edt(~core, return_indices=True)
        sd = (
            d_out - ndimage.distance_transform_edt(core) if not core.all() else -np.ones(core.shape)
        )
        nearest = kept[idx[0], idx[1]]  # id of the closest dropped patch
        s = sd - k[nearest] - self.ragged_px * noise  # < 0: inside the ragged border

        if self.fade_px > 0:
            # soft border: speckle outside, porous rim inside, solid / clean further away
            p_missing = np.clip(0.5 - s / (2.0 * self.fade_px), 0.0, 1.0)
            hole = rng.random(core.shape) < p_missing
        else:
            hole = s <= 0

        if self.island_frac > 0:
            # valid islands inside big holes, so large holes are not perfectly filled
            f = fbm(core.shape, rng, self.ragged_sigma_px, self.octaves)
            islands = f > np.quantile(f, 1.0 - self.island_frac)
            hole &= ~(islands & (s < -self.fade_px))
        return hole


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
    structure exists at several scales (ragged borders, not smooth blobs).
    """
    out = np.zeros(shape)
    for o in range(max(octaves, 1)):
        field = ndimage.gaussian_filter(rng.standard_normal(shape), max(sigma / 2**o, 0.7))
        out += (0.5**o) * field / max(float(field.std()), 1e-9)
    return out / max(float(out.std()), 1e-9)


def _relabel(labels: np.ndarray) -> np.ndarray:
    """Make ids consecutive ``1..N`` keeping 0 as background."""
    ids, inv = np.unique(labels, return_inverse=True)
    inv = inv.reshape(labels.shape)
    return (inv if ids[0] == 0 else inv + 1).astype(np.int64) * (labels > 0)


def surface_patches(
    z: np.ndarray,
    n: np.ndarray,
    valid: np.ndarray,
    tau_n_deg: float,
    tau_z: float,
    cell: float,
    rng: np.random.Generator,
    warp_px: float = 0.0,
    octaves: int = 3,
) -> np.ndarray:
    """Label map (0 = invalid) of connected pixels with similar normal and depth.

    Components larger than ``2 * cell**2`` are cut into about ``area / cell**2`` Voronoi
    cells whose borders are bent by a smooth random warp of amplitude ``warp_px``.
    """
    h, w = z.shape
    cos_t = np.cos(np.deg2rad(tau_n_deg))
    idx = np.arange(h * w).reshape(h, w)

    def link(a, b, za, zb, na, nb):
        ok = a & b
        ok &= np.abs(za - zb) <= tau_z * np.minimum(za, zb)
        ok &= np.sum(na * nb, axis=-1) >= cos_t
        return ok

    right = link(valid[:, :-1], valid[:, 1:], z[:, :-1], z[:, 1:], n[:, :-1], n[:, 1:])
    down = link(valid[:-1], valid[1:], z[:-1], z[1:], n[:-1], n[1:])
    src = np.concatenate([idx[:, :-1][right], idx[:-1][down]])
    dst = np.concatenate([idx[:, 1:][right], idx[1:][down]])
    graph = coo_matrix((np.ones(src.size, dtype=np.int8), (src, dst)), shape=(h * w, h * w))
    _, comp = connected_components(graph, directed=False)
    labels = _relabel(np.where(valid, (comp + 1).reshape(h, w), 0))

    # cut big components into domain-warped Voronoi cells (curved, irregular borders)
    area = np.bincount(labels.ravel())
    big_ids = np.flatnonzero(area > 2.0 * cell * cell)
    big_ids = big_ids[big_ids > 0]
    if big_ids.size == 0:
        return labels
    vs, us = np.mgrid[0:h, 0:w].astype(np.float64)
    sigma = max(cell / 2.0, 1.0)
    wu = us + warp_px * fbm((h, w), rng, sigma, octaves)
    wv = vs + warp_px * fbm((h, w), rng, sigma, octaves)
    out = labels.copy()
    next_id = int(labels.max()) + 1
    for lid in big_ids:
        pix = np.flatnonzero(labels.ravel() == lid)
        n_seeds = max(2, round(pix.size / (cell * cell)))
        seeds = rng.choice(pix, size=min(n_seeds, pix.size), replace=False)
        pts = np.stack([wv.ravel()[pix], wu.ravel()[pix]], axis=1)
        sp = np.stack([wv.ravel()[seeds], wu.ravel()[seeds]], axis=1)
        _, nearest = cKDTree(sp).query(pts)
        flat = out.ravel()
        flat[pix] = next_id + nearest
        next_id += len(seeds)
    return _relabel(out)


def edge_pixel_mask(
    z: np.ndarray, n: np.ndarray, valid: np.ndarray, tau_n_deg: float, tau_z: float, window: int
) -> np.ndarray:
    """Valid pixels whose ``window`` x ``window`` neighbourhood has a depth or normal jump."""
    big = float(np.max(z, initial=1.0)) * 4.0 + 1.0
    zmax = ndimage.maximum_filter(np.where(valid, z, 0.0), size=window, mode="nearest")
    zmin = ndimage.minimum_filter(np.where(valid, z, big), size=window, mode="nearest")
    with np.errstate(invalid="ignore", divide="ignore"):
        depth_jump = (zmax - zmin) / np.where(valid, z, 1.0) > tau_z

    r = window // 2
    h, w = z.shape
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
    normal_jump = min_cos < np.cos(np.deg2rad(tau_n_deg))
    return valid & (depth_jump | normal_jump)


def edge_patches(
    z: np.ndarray,
    n: np.ndarray,
    valid: np.ndarray,
    tau_n_deg: float,
    tau_z: float,
    window: int,
    width_px: float,
    rng: np.random.Generator,
    sigma_px: float = 8.0,
    octaves: int = 3,
) -> np.ndarray:
    """Label map of connected blobs grown around edge pixels (8-connectivity).

    Every edge pixel is dilated by a radius that varies smoothly across the image between
    ``0.3 * width_px`` and ``width_px``, so thin edge chains swell into uneven blobs
    instead of uniform lines.
    """
    edge = edge_pixel_mask(z, n, valid, tau_n_deg, tau_z, window)
    if width_px > 0 and edge.any():
        radius = width_px * (0.65 + 0.35 * np.clip(fbm(z.shape, rng, sigma_px, octaves), -2, 2) / 2)
        radius = np.maximum(radius, 0.3 * width_px)
        # the blob around an edge pixel reaches as far as the local radius allows
        dist, idx = ndimage.distance_transform_edt(~edge, return_indices=True)
        edge = (dist <= radius[idx[0], idx[1]]) & valid
    labels, _ = ndimage.label(edge, structure=np.ones((3, 3), dtype=int))
    return labels.astype(np.int64)


def merge_small_patches(
    labels: np.ndarray, min_area_px: int, merge_dist_px: float, discard_small: bool = False
) -> tuple[np.ndarray, int]:
    """Merge patches under ``min_area_px`` into the nearest larger patch.

    Returns the relabelled map (ids ``1..N``) and the number of patches merged.  A small
    patch with no larger patch within ``merge_dist_px`` is kept, or removed if
    ``discard_small``.
    """
    area = np.bincount(labels.ravel())
    small_ids = np.flatnonzero((area < min_area_px) & (area > 0))
    small_ids = small_ids[small_ids > 0]
    large = (area >= min_area_px) & (np.arange(area.size) > 0)
    if small_ids.size == 0:
        return labels, 0
    if not large.any():
        return (np.zeros_like(labels), 0) if discard_small else (labels, 0)

    large_mask = large[labels]
    dist, inds = ndimage.distance_transform_edt(~large_mask, return_indices=True)
    nearest_label = labels[inds[0], inds[1]]
    close = ndimage.minimum(dist, labels, index=small_ids)
    # a patch takes the label of the larger patch nearest to any of its pixels
    pos = ndimage.minimum_position(dist, labels, index=small_ids)
    target = np.array([nearest_label[p] for p in pos], dtype=np.int64)

    remap = np.arange(area.size)
    near = close <= merge_dist_px
    remap[small_ids[near]] = target[near]
    if discard_small:
        remap[small_ids[~near]] = 0
    return _relabel(remap[labels]), int(near.sum())
