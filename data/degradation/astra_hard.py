"""Strong, stair-aware degradation of Astra 2 depth for the A' experiments.

Extends ``active_stereo_astra2`` (disparity quantisation + matching noise, occlusion
shadows, edge fattening, depth-dependent dropout -- all kept, run first) with the hole
and artefact types the stair task needs.  Each type has its own switch:

1. ``edge_holes``   bands along tread/riser (and stair/background) boundaries, random width
2. ``glare_holes``  specular / bright and flat low-texture RGB regions
3. ``blob_holes``   large irregular blobs
4. ``depth_noise``  disparity noise (quadratic in z once mapped back to depth); amplitude x severity
5. ``flying``       "flying pixels" on depth discontinuities (disparity quantisation lives in the base)

Then the HR map is area-downsampled by ``scale`` with a validity mask
(``masked_downsample``), which is also what the A' data pipeline uses -- one function,
so a baseline and A' see the same information.

Severity presets (``light`` ~ real Astra, ``medium``, ``strong``) are STARTING VALUES.
They must be tuned against ``scripts/astra_defect_stats.py`` output on the real data
(input invalid fraction of ``strong`` should be 2-3x that of real Astra).

Edge holes use StairNet masks cached by ``scripts/stairnet_cache.py`` when present
(``stair_cache_dir``); otherwise they fall back to depth discontinuities of the GT and
``meta['edge_source']`` says so.  Randomness: seeded per sample; ``stochastic=True``
(training) additionally mixes in ``self.epoch``, so every epoch sees a new but
reproducible corruption.  Test = ``stochastic=False``.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from data.degradation.active_stereo_astra2 import ActiveStereoAstra2Degradation, _depth_edges
from data.degradation.base import BaseDegradation, register_degradation
from utils.misc import sparse_to_dense_nn

__all__ = ["AstraHardDegradation", "SEVERITY", "masked_downsample"]

# multipliers / amplitudes per severity. Starting values, see module docstring.
SEVERITY: dict[str, dict[str, float]] = {
    "light": dict(
        sigma_disp=0.08, dropout=0.004, edge_cov=0.10, edge_w=(2, 4), glare_p=0.3,
        blob_n=(0, 2), blob_r=(6, 15), fly_p=0.10,
    ),
    "medium": dict(
        sigma_disp=0.16, dropout=0.010, edge_cov=0.30, edge_w=(3, 8), glare_p=0.6,
        blob_n=(1, 4), blob_r=(8, 28), fly_p=0.25,
    ),
    "strong": dict(
        sigma_disp=0.30, dropout=0.020, edge_cov=0.60, edge_w=(4, 14), glare_p=0.9,
        blob_n=(3, 8), blob_r=(12, 45), fly_p=0.45,
    ),
}


def masked_downsample(
    depth: np.ndarray, scale: int, min_frac: float = 0.5
) -> tuple[np.ndarray, np.ndarray]:
    """Area-average the valid (``>0``) pixels of each ``scale x scale`` block.

    Returns ``(lr, valid)``: ``lr`` is 0 where fewer than ``min_frac`` of the block's
    pixels are valid, ``valid`` the matching bool mask.  Sizes not divisible by
    ``scale`` are zero-padded (counts as invalid).
    """
    h, w = depth.shape
    hp, wp = -(-h // scale) * scale, -(-w // scale) * scale
    d = np.zeros((hp, wp), np.float32)
    d[:h, :w] = depth
    v = (d > 0).astype(np.float32)
    blocks = lambda a: a.reshape(hp // scale, scale, wp // scale, scale).sum(axis=(1, 3))  # noqa: E731
    cnt, tot = blocks(v), blocks(d * v)
    valid = cnt >= max(1.0, min_frac * scale * scale)
    lr = np.where(valid, tot / np.maximum(cnt, 1.0), 0.0).astype(np.float32)
    return lr, valid


def _smooth_field(rng: np.random.Generator, shape: tuple[int, int], sigma: float) -> np.ndarray:
    """Unit-variance spatially correlated noise."""
    from scipy import ndimage

    f = ndimage.gaussian_filter(rng.standard_normal(shape).astype(np.float32), sigma)
    return f / max(float(f.std()), 1e-6)


@register_degradation("astra_hard")
@dataclass
class AstraHardDegradation(BaseDegradation):
    key: str = "astra_hard"
    needs_context: bool = True

    severity: str = "strong"
    scale: int = 8
    lr_min_frac: float = 0.5
    edge_holes: bool = True
    glare_holes: bool = True
    blob_holes: bool = True
    depth_noise: bool = True
    flying: bool = True
    stair_bg_boundary: bool = True
    glare_max_frac: float = 0.08
    """Cap on the share of the frame removed by glare/flat-region holes."""
    stochastic: bool = False
    stair_cache_dir: str | None = None
    """Directory with ``<sample_id>.npz`` StairNet masks (see data/stairnet/cache.py)."""
    mask_clean: dict | bool | None = None
    """Parameters of data.stairnet.edges.clean_masks for the cached masks (None = defaults)."""
    base: dict[str, Any] = field(default_factory=dict)
    """Overrides for the wrapped ActiveStereoAstra2Degradation."""
    epoch: int = 0

    def __post_init__(self) -> None:
        if self.severity not in SEVERITY:
            raise ValueError(f"severity must be one of {sorted(SEVERITY)}, got {self.severity!r}")

    # ---- helpers ---------------------------------------------------------
    def _edge_map(self, gt: np.ndarray, context: Mapping[str, Any] | None) -> tuple[np.ndarray, str]:
        sid = None if context is None else context.get("sample_id")
        if self.stair_cache_dir and sid is not None:
            from data.stairnet.cache import load_masks
            from data.stairnet.edges import boundary_map, clean_masks

            m = load_masks(self.stair_cache_dir, sid, shape=gt.shape)
            if m is not None:
                return boundary_map(clean_masks(m["cls"], self.mask_clean), self.stair_bg_boundary), "stairnet"
        valid = gt > 0
        return _depth_edges(gt, valid, 0.05), "depth_edges"

    def apply(self, gt: np.ndarray, rng: np.random.Generator, context=None) -> Mapping[str, Any]:
        from scipy import ndimage

        sev = SEVERITY[self.severity]
        if self.stochastic:  # fresh but reproducible corruption every epoch
            rng = np.random.default_rng([int(rng.integers(2**31)), int(self.epoch)])
        h, w = gt.shape
        real_hole = ~(gt > 0)

        # base Astra model: quantisation, matching noise (x severity), shadows, dropout
        bp = dict(
            baseline_m=0.0752,  # measured: Orbbec recording /cam/disparityParam (baseline 75.2 mm); depth stream fx 1426 px @1600x1200
            focal_px=627.2,  # = colour-frame focal of the 528x688 frames (disparity-domain quantisation is then ~2.3x coarser than the sensor's)
            sigma_disp_px=sev["sigma_disp"] if self.depth_noise else 0.0,
            dropout_base=sev["dropout"],
            edge_fatten_px=2 if self.flying else 0,
        )
        bp.update(self.base)
        base = ActiveStereoAstra2Degradation(**bp)
        z = np.asarray(base.apply(gt, rng)["sparse_depth"], np.float32)
        hole = z <= 0
        base_hole = hole & ~real_hole

        # 5. flying pixels on discontinuities
        if self.flying:
            edge = _depth_edges(gt, gt > 0, 0.05) & (z > 0)
            pick = edge & (rng.random(z.shape) < sev["fly_p"])
            if pick.any():
                lo = ndimage.minimum_filter(np.where(z > 0, z, np.inf), 3)
                hi = ndimage.maximum_filter(z, 3)
                lam = rng.random(z.shape).astype(np.float32)
                fly = lam * lo + (1 - lam) * hi
                z = np.where(pick & np.isfinite(fly), fly, z).astype(np.float32)

        added = np.zeros_like(hole)
        # 1. holes along stair edges, random width, partial coverage
        edge_src = "off"
        if self.edge_holes:
            edge, edge_src = self._edge_map(gt, context)
            if edge.any():
                dist = ndimage.distance_transform_edt(~edge)
                lo_w, hi_w = sev["edge_w"]
                u = 0.5 + 0.5 * np.tanh(_smooth_field(rng, (h, w), 25.0))  # ~U(0,1), smooth
                width = lo_w + (hi_w - lo_w) * u
                gate = _smooth_field(rng, (h, w), 12.0) < np.sqrt(2) * _erfinv(2 * sev["edge_cov"] - 1)
                added |= (dist <= width / 2) & gate
        # 2. glare / flat regions from the RGB
        if self.glare_holes and context is not None and context.get("rgb") is not None:
            rgb = np.asarray(context["rgb"], np.float32)
            gray = rgb.mean(-1)
            mean = ndimage.uniform_filter(gray, 7)
            std = np.sqrt(np.maximum(ndimage.uniform_filter(gray**2, 7) - mean**2, 0))
            # saturated / very dark AND flat (a flat mid-grey wall is NOT a sensor failure)
            extreme = (gray > 0.85) | (gray < 0.12)
            cand = ndimage.binary_opening((gray > 0.92) | ((std < 0.012) & extreme), np.ones((5, 5), bool))
            lab, n = ndimage.label(cand)
            if n:
                sizes = ndimage.sum(cand, lab, range(1, n + 1))
                used = 0.0
                for j in rng.permutation(n):  # random order, stop at the per-frame cap
                    if sizes[j] >= 400 and rng.random() < sev["glare_p"] and used + sizes[j] <= self.glare_max_frac * h * w:
                        added |= lab == j + 1
                        used += sizes[j]
        # 3. large irregular blobs
        if self.blob_holes:
            nf = _smooth_field(rng, (h, w), 8.0)
            yy, xx = np.mgrid[:h, :w]
            for _ in range(int(rng.integers(sev["blob_n"][0], sev["blob_n"][1] + 1))):
                cy, cx = rng.integers(h), rng.integers(w)
                r = rng.uniform(*sev["blob_r"])
                rr = np.hypot((yy - cy) / rng.uniform(0.6, 1.4), xx - cx)
                added |= rr / r + 0.25 * nf < 1.0

        z = np.where(added, 0.0, z).astype(np.float32)
        lr, lr_valid = masked_downsample(z, self.scale, self.lr_min_frac)
        lr_dense = sparse_to_dense_nn(lr) if lr_valid.any() else lr
        return {
            "sparse_depth": z,
            "lr_depth": lr_dense,
            "meta": {
                "severity": self.severity,
                "edge_source": edge_src,
                "input_invalid_frac": float((z <= 0).mean()),
                "input_invalid_frac_in_gt_valid": float((z[~real_hole] <= 0).mean())
                if (~real_hole).any()
                else 0.0,
                "added_hole_frac": float(((added | base_hole) & ~real_hole).mean()),
                "lr_invalid_frac": float(1.0 - lr_valid.mean()),
                "calibrated": False,
            },
        }


def _erfinv(x: float) -> float:
    from scipy.special import erfinv

    return float(erfinv(np.clip(x, -0.999, 0.999)))
