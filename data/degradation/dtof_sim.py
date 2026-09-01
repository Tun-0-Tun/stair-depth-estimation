"""Zone-based dToF simulation (VL53L5CX-class sensors, as in ZJU-L5).

This is the input DEPTHOR was designed for, so we need it to (a) reproduce the
baseline's published setting and (b) compare a dToF against our active-stereo
camera under identical conditions.

Sensor model
------------
A lightweight dToF is not a depth camera; it is a small grid of
``zones_h x zones_w`` (typically 8x8) SPAD histogram bins, each integrating the
return over a wide sub-FoV.  Consequences, all modelled here:

* **Zone quantisation.**  One distance per zone, covering many pixels.  The
  reported value is a statistic of the surfaces inside the zone, not the depth
  at its centre.
* **Multi-peak / mixed pixels.**  A zone straddling a stair nosing sees two
  surfaces.  Real firmware reports either the strongest or the nearest peak;
  ``peak_selection`` switches between them.  This is the dominant error source
  on stairs, which is exactly why it matters for this project.
* **Range-dependent noise.**  Photon-limited: sigma grows roughly linearly with
  distance for a fixed integration time.
* **Zone dropout.**  Low-reflectance or out-of-range zones report nothing.
* **Zone confidence footprint.**  Firmware often reports a value valid only for
  the central part of the zone; ``fill_ratio`` controls how much of the zone
  footprint we write into the sparse map.

The output is a *sparse* full-resolution map: patches of constant depth on a
grid, zeros elsewhere -- i.e. the same structure as a real ZJU-L5 frame.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np

from data.degradation.base import BaseDegradation, register_degradation

__all__ = ["DToFSimDegradation"]


@register_degradation("dtof_sim")
@dataclass
class DToFSimDegradation(BaseDegradation):
    """Simulate a low-resolution zone-based dToF from clean ground-truth depth."""

    key: str = "dtof_sim"

    zones_h: int = 8
    zones_w: int = 8
    fov_fraction: float = 1.0
    """Fraction of the RGB frame covered by the dToF FoV, centred. L5 sees a
    narrower FoV than the RGB camera; 1.0 means "same FoV"."""

    peak_selection: str = "strongest"
    """``strongest`` (mode of the in-zone depth histogram), ``nearest`` (min) or
    ``mean``.  ZJU-L5 firmware behaves closest to ``strongest``."""
    hist_bins: int = 64

    sigma_base_m: float = 0.010
    sigma_per_m: float = 0.012
    """sigma(z) = sigma_base_m + sigma_per_m * z."""

    z_min: float = 0.2
    z_max: float = 4.0
    zone_dropout: float = 0.05
    fill_ratio: float = 0.6
    """Side length of the written patch as a fraction of the zone footprint."""
    min_valid_fraction: float = 0.25
    """A zone with less valid GT than this reports nothing."""

    def apply(self, gt: np.ndarray, rng: np.random.Generator) -> Mapping[str, Any]:
        h, w = gt.shape
        sparse = np.zeros((h, w), dtype=np.float32)

        fov_h = round(h * self.fov_fraction)
        fov_w = round(w * self.fov_fraction)
        y0, x0 = (h - fov_h) // 2, (w - fov_w) // 2

        ys = np.linspace(y0, y0 + fov_h, self.zones_h + 1).round().astype(int)
        xs = np.linspace(x0, x0 + fov_w, self.zones_w + 1).round().astype(int)

        n_reported = 0
        for i in range(self.zones_h):
            for j in range(self.zones_w):
                ya, yb, xa, xb = ys[i], ys[i + 1], xs[j], xs[j + 1]
                patch = gt[ya:yb, xa:xb]
                if patch.size == 0:
                    continue
                vals = patch[np.isfinite(patch) & (patch > self.z_min) & (patch < self.z_max)]
                if vals.size < self.min_valid_fraction * patch.size:
                    continue
                if rng.random() < self.zone_dropout:
                    continue

                d = self._select_peak(vals, rng)
                d = d + rng.normal(0.0, self.sigma_base_m + self.sigma_per_m * d)
                if not (self.z_min < d < self.z_max):
                    continue

                # write the confidence footprint at the zone centre
                cy, cx = (ya + yb) / 2.0, (xa + xb) / 2.0
                ph = max(1, round((yb - ya) * self.fill_ratio))
                pw = max(1, round((xb - xa) * self.fill_ratio))
                sy = slice(max(0, int(cy - ph / 2)), min(h, int(cy + ph / 2) + 1))
                sx = slice(max(0, int(cx - pw / 2)), min(w, int(cx + pw / 2) + 1))
                sparse[sy, sx] = np.float32(d)
                n_reported += 1

        return {
            "sparse_depth": sparse,
            "lr_depth": None,
            "meta": {
                "n_zones_reported": n_reported,
                "n_zones_total": self.zones_h * self.zones_w,
                "valid_ratio": float((sparse > 0).mean()),
            },
        }

    def _select_peak(self, vals: np.ndarray, rng: np.random.Generator) -> float:
        if self.peak_selection == "nearest":
            return float(vals.min())
        if self.peak_selection == "mean":
            return float(vals.mean())
        if self.peak_selection != "strongest":
            raise ValueError(
                f"peak_selection must be strongest|nearest|mean, got {self.peak_selection!r}"
            )
        # strongest return == the most populated histogram bin, which on a
        # two-surface zone snaps to whichever surface fills more of the FoV.
        counts, edges = np.histogram(vals, bins=self.hist_bins)
        k = int(np.argmax(counts))
        inside = vals[(vals >= edges[k]) & (vals <= edges[k + 1])]
        return float(inside.mean()) if inside.size else float(np.median(vals))
