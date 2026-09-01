"""Degradation model of the Orbbec Astra 2 (active stereo).

OWNER: Olesya.  This file is the deliverable "Astra 2 noise model" from the SOW.

What an active-stereo camera actually does to depth, and how each effect is
modelled here:

1. **Disparity quantisation.**  The camera matches a projected pattern between
   two views and reports ``d = f * B / Z`` quantised to ``1/subpixel_bits`` of a
   pixel.  The resulting depth error grows as ``Z^2``: ``dZ = Z^2 / (f*B) * dd``.
   Modelled exactly, in the disparity domain, rather than as depth-domain noise
   -- this is the single most important qualitative difference from a dToF.
2. **Matching noise.**  Zero-mean Gaussian on disparity, ``sigma_disp_px``.
   Again quadratic in depth once mapped back.
3. **Occlusion shadows.**  A surface visible to the left camera but hidden from
   the right produces a band of invalid pixels next to every depth
   discontinuity, of width ``f*B*(1/Z_fg - 1/Z_bg)`` pixels on the far side.
   This is *structured* invalidity -- exactly the kind that hurts on stair
   nosings -- and a Bernoulli dropout does not reproduce it.
4. **Edge fattening / flying pixels.**  Block matching near a discontinuity
   returns a depth between foreground and background.
5. **Low-signal dropout.**  Dark, specular or distant surfaces return no
   pattern.  Modelled as blobs, with probability rising with depth (the
   projector's irradiance falls as ``1/Z^2``) and falling with RGB luminance.
6. **Operating range.**  Outside ``[z_min, z_max]`` the camera reports nothing.

Defaults correspond to the Astra 2 datasheet (baseline ~ 40 mm, ~0.6-8 m range,
1/8-pixel subpixel interpolation) at 640x480.  **They are a starting point, not
a calibration.**  The calibration task is: capture stair scenes with the Astra 2
plus a reference (e.g. a laser scanner or a slow high-quality stereo sweep),
fit ``sigma_disp_px``, ``dropout_*`` and ``edge_*`` to the observed error
statistics, and record the fitted values in
``configs/degradation/astra2_calibrated.yaml``.  Until then every result
produced with this model must say "uncalibrated" in the notes column.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np

from data.degradation.base import BaseDegradation, register_degradation

__all__ = ["ActiveStereoAstra2Degradation"]


@register_degradation("active_stereo_astra2")
@dataclass
class ActiveStereoAstra2Degradation(BaseDegradation):
    """Simulate an Orbbec Astra 2 depth frame from clean ground-truth depth."""

    key: str = "active_stereo_astra2"

    # --- geometry -------------------------------------------------------
    baseline_m: float = 0.040
    """Stereo baseline in metres (Astra 2 IR pair, ~40 mm)."""
    focal_px: float = 570.0
    """Focal length in pixels at the working resolution. Rescale with the image."""
    subpixel_bits: int = 3
    """Disparity is quantised to 1/2**subpixel_bits px (3 -> 1/8 px)."""

    # --- range ----------------------------------------------------------
    z_min: float = 0.6
    z_max: float = 8.0

    # --- noise ----------------------------------------------------------
    sigma_disp_px: float = 0.08
    """Std-dev of the disparity matching error, in pixels."""

    # --- structured invalidity -----------------------------------------
    occlusion: bool = True
    """Model the right-camera occlusion shadow next to depth discontinuities."""
    edge_fatten_px: int = 2
    """Half-width of the flying-pixel band around a discontinuity."""
    edge_invalid_prob: float = 0.35
    """Probability that a pixel in that band is invalidated rather than smeared."""
    depth_edge_threshold: float = 0.05
    """Relative depth jump that counts as a discontinuity (5%)."""

    # --- low-signal dropout ---------------------------------------------
    dropout_base: float = 0.02
    """Dropout probability at ``z_ref`` on an average-albedo surface."""
    dropout_z_ref: float = 1.5
    dropout_blob_sigma_px: float = 6.0
    """Spatial correlation of the dropout field; 0 gives salt-and-pepper."""
    dropout_max: float = 0.85

    def apply(self, gt: np.ndarray, rng: np.random.Generator) -> Mapping[str, Any]:
        from scipy import ndimage

        z = gt.astype(np.float64)
        valid = np.isfinite(z) & (z > 0)

        fb = self.focal_px * self.baseline_m
        with np.errstate(divide="ignore", invalid="ignore"):
            disp = np.where(valid, fb / np.maximum(z, 1e-6), 0.0)

        # 2. matching noise, in the disparity domain
        if self.sigma_disp_px > 0:
            disp = disp + rng.normal(0.0, self.sigma_disp_px, size=disp.shape)

        # 1. subpixel quantisation
        step = 1.0 / (2**self.subpixel_bits)
        disp = np.round(disp / step) * step

        with np.errstate(divide="ignore", invalid="ignore"):
            z_meas = np.where(disp > 1e-9, fb / disp, 0.0)
        z_meas = np.nan_to_num(z_meas, nan=0.0, posinf=0.0, neginf=0.0)

        invalid = ~valid

        # 3. occlusion shadow: a background pixel is unmatched if some pixel to
        #    its left is closer by more than the local disparity difference.
        if self.occlusion:
            invalid |= _occlusion_shadow(z, fb, valid)

        # 4. edge fattening: mix neighbouring depths inside a band around edges
        edge = _depth_edges(z, valid, self.depth_edge_threshold)
        if self.edge_fatten_px > 0 and edge.any():
            band = ndimage.binary_dilation(
                edge, structure=np.ones((3, 3), bool), iterations=int(self.edge_fatten_px)
            )
            smoothed = ndimage.uniform_filter(
                np.where(valid, z_meas, 0.0), size=2 * self.edge_fatten_px + 1
            )
            weights = ndimage.uniform_filter(
                valid.astype(np.float64), size=2 * self.edge_fatten_px + 1
            )
            with np.errstate(divide="ignore", invalid="ignore"):
                smoothed = np.where(weights > 1e-6, smoothed / np.maximum(weights, 1e-6), z_meas)
            z_meas = np.where(band, smoothed, z_meas)
            invalid |= band & (rng.random(z.shape) < self.edge_invalid_prob)

        # 5. low-signal dropout, spatially correlated and rising with Z^2
        p = self.dropout_base * (np.maximum(z, self.z_min) / self.dropout_z_ref) ** 2
        p = np.clip(p, 0.0, self.dropout_max)
        field = rng.random(z.shape)
        if self.dropout_blob_sigma_px > 0:
            field = ndimage.gaussian_filter(field, self.dropout_blob_sigma_px)
            # gaussian filtering shrinks the range; renormalise to a uniform-ish [0,1]
            lo, hi = field.min(), field.max()
            field = (field - lo) / max(hi - lo, 1e-9)
        invalid |= field < p

        # 6. operating range
        invalid |= (z_meas < self.z_min) | (z_meas > self.z_max)

        sparse = np.where(invalid, 0.0, z_meas).astype(np.float32)
        return {
            "sparse_depth": sparse,
            "lr_depth": None,  # the Astra 2 outputs full-resolution depth
            "meta": {"valid_ratio": float((sparse > 0).mean()), "calibrated": False},
        }


def _depth_edges(z: np.ndarray, valid: np.ndarray, rel_threshold: float) -> np.ndarray:
    """Pixels where the relative depth jump to a 4-neighbour exceeds the threshold."""
    edge = np.zeros(z.shape, dtype=bool)
    for axis in (0, 1):
        d = np.abs(np.diff(z, axis=axis))
        base = np.maximum(
            np.take(z, range(0, z.shape[axis] - 1), axis=axis),
            np.take(z, range(1, z.shape[axis]), axis=axis),
        )
        v = np.take(valid, range(0, z.shape[axis] - 1), axis=axis) & np.take(
            valid, range(1, z.shape[axis]), axis=axis
        )
        jump = v & (d > rel_threshold * np.maximum(base, 1e-6))
        if axis == 0:
            edge[:-1] |= jump
            edge[1:] |= jump
        else:
            edge[:, :-1] |= jump
            edge[:, 1:] |= jump
    return edge


def _occlusion_shadow(z: np.ndarray, fb: float, valid: np.ndarray) -> np.ndarray:
    """Left-right consistency: mark pixels the second camera cannot see.

    Warp each pixel by its disparity into the right image and keep, per target
    column, only the closest source pixel; everything that loses the contest is
    occluded and gets no depth.  This is the standard forward-warp z-buffer
    test, run at whole-pixel resolution -- enough to reproduce the shadow bands.
    """
    h, w = z.shape
    with np.errstate(divide="ignore", invalid="ignore"):
        disp = np.where(valid & (z > 0), fb / np.maximum(z, 1e-6), 0.0)
    xs = np.arange(w)[None, :].repeat(h, axis=0)
    tgt = np.rint(xs - disp).astype(np.int64)
    inside = valid & (tgt >= 0) & (tgt < w)

    occluded = np.zeros((h, w), dtype=bool)
    for y in range(h):
        row_valid = inside[y]
        if not row_valid.any():
            continue
        cols = tgt[y][row_valid]
        depths = z[y][row_valid]
        src = np.flatnonzero(row_valid)
        # sort by target column, then by depth: the first entry per column wins
        order = np.lexsort((depths, cols))
        cols_s, src_s = cols[order], src[order]
        winner = np.ones(w, dtype=np.int64) * -1
        first = np.concatenate(([True], cols_s[1:] != cols_s[:-1]))
        winner[cols_s[first]] = src_s[first]
        lost = np.ones(w, dtype=bool)
        lost[winner[winner >= 0]] = False
        occluded[y] = lost & row_valid
    return occluded
