"""Edge band ``E`` and ignore mask from StairNet class maps (canonical ids 0 bg / 1 tread / 2 riser).

Modes (the StairNetV3 heat-map idea, applied to our boundaries):
  binary    1 inside the band around a boundary segment
  gaussian  Y = exp(-d^2 / (2 sigma^2)), d = distance (along the segment) to the nearest segment
            end, sigma = half the segment length: 1 at the ends, exp(-1/2) ~ 0.61 in the middle
Boundaries are traced as contours of the tread|riser (and optionally stair|background)
transition and approximated by polylines (``cv2.approxPolyDP``); every polyline piece is one
"segment".  ``band_px`` is the full band width.
"""

from __future__ import annotations

import cv2
import numpy as np
from scipy import ndimage


DEFAULT_CLEAN = {"min_area": 1500, "open_px": 3, "min_stair_frac": 0.05, "top_k": 0}


def clean_masks(cls: np.ndarray, params: dict | None = None) -> np.ndarray:
    """Suppress StairNet false positives (city scenes, textures): per class, binary opening, drop connected
    regions smaller than ``min_area`` px (and keep only the ``top_k`` largest if >0); if the cleaned stair
    area is below ``min_stair_frac`` of the frame the whole frame is declared "no stairs" (all background),
    so neither the edge weight nor the planar loss act on it.  ``params=None`` -> DEFAULT_CLEAN, ``{}``/``False`` -> off.
    """
    if params is False or params == {}:
        return cls
    p = {**DEFAULT_CLEAN, **(params or {})}
    out = np.zeros_like(cls)
    k = np.ones((p["open_px"],) * 2, bool) if p["open_px"] > 1 else None
    for c in (1, 2):
        m = cls == c
        if k is not None:
            m = ndimage.binary_opening(m, k)
        lab, n = ndimage.label(m)
        if not n:
            continue
        sizes = np.asarray(ndimage.sum(m, lab, range(1, n + 1)))
        ids = np.flatnonzero(sizes >= p["min_area"])
        if p["top_k"]:
            ids = ids[np.argsort(-sizes[ids])[: p["top_k"]]]
        out[np.isin(lab, ids + 1)] = c
    if (out > 0).mean() < p["min_stair_frac"]:
        out[:] = 0
    return out


def boundary_map(cls: np.ndarray, stair_bg: bool = True) -> np.ndarray:
    """bool (H, W): pixels on a tread|riser (and stair|background) transition."""
    stair = cls > 0
    b = np.zeros(cls.shape, bool)
    for a, c in ((cls[:, :-1], cls[:, 1:]), (cls[:-1, :], cls[1:, :])):
        diff = a != c
        keep = diff & stair_pair(a, c, stair_bg)
        if a.shape[0] == cls.shape[0]:
            b[:, :-1] |= keep
            b[:, 1:] |= keep
        else:
            b[:-1, :] |= keep
            b[1:, :] |= keep
    del stair
    return b


def stair_pair(a: np.ndarray, c: np.ndarray, stair_bg: bool) -> np.ndarray:
    tr = ((a == 1) & (c == 2)) | ((a == 2) & (c == 1))
    if not stair_bg:
        return tr
    return tr | ((a == 0) & (c > 0)) | ((c == 0) & (a > 0))


def segments_from_boundary(b: np.ndarray, eps: float = 2.0, min_len: float = 4.0) -> np.ndarray:
    """(N, 4) float32 ``x1, y1, x2, y2`` polyline pieces of the boundary contours."""
    cnts, _ = cv2.findContours(b.astype(np.uint8), cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    segs = []
    for c in cnts:
        p = cv2.approxPolyDP(c, eps, False)[:, 0, :].astype(np.float32)
        for a, q in zip(p[:-1], p[1:]):
            if np.hypot(*(q - a)) >= min_len:
                segs.append([*a, *q])
    return np.asarray(segs, np.float32).reshape(-1, 4)


def rasterize(segs: np.ndarray, shape, band_px: float, mode: str) -> tuple[np.ndarray, np.ndarray]:
    """``(E float32, band bool)`` for segments, ``E = max`` over segments."""
    H, W = shape
    E = np.zeros(shape, np.float32)
    r = band_px / 2.0
    for x1, y1, x2, y2 in segs:
        x0, xe = int(max(min(x1, x2) - r - 1, 0)), int(min(max(x1, x2) + r + 2, W))
        y0, ye = int(max(min(y1, y2) - r - 1, 0)), int(min(max(y1, y2) + r + 2, H))
        yy, xx = np.mgrid[y0:ye, x0:xe].astype(np.float32)
        dx, dy = x2 - x1, y2 - y1
        L2 = max(dx * dx + dy * dy, 1e-6)
        t = np.clip(((xx - x1) * dx + (yy - y1) * dy) / L2, 0.0, 1.0)
        d = np.hypot(xx - (x1 + t * dx), yy - (y1 + t * dy))
        inb = d <= r
        if mode == "binary":
            y = np.ones_like(t)
        else:
            L = np.sqrt(L2)
            d_end = np.minimum(t, 1.0 - t) * L
            y = np.exp(-(d_end**2) / (2.0 * (L / 2.0) ** 2))
        sl = E[y0:ye, x0:xe]
        np.maximum(sl, np.where(inb, y, 0.0), out=sl)
    return E, E > 0


def build_edge_maps(
    cls: np.ndarray,
    conf: np.ndarray,
    mode: str = "gaussian",
    band_px: float = 6.0,
    conf_thr: float = 0.7,
    stair_bg: bool = True,
) -> dict[str, np.ndarray]:
    """``E`` float (0 outside the band), ``band`` bool, ``ignore`` bool (low StairNet confidence).

    A boundary pixel sits where the softmax of two classes crosses, so its own confidence is
    low by construction; the boundary is kept when *somewhere near it* (within the band) the
    network is confident, i.e. local max of ``conf`` >= ``conf_thr``.
    """
    if mode == "none":
        z = np.zeros(cls.shape, np.float32)
        return {"E": z, "band": z > 0, "ignore": conf < conf_thr}
    segs = segments_from_boundary(boundary_map(cls, stair_bg))
    E, band = rasterize(segs, cls.shape, band_px, mode)
    k = int(band_px) | 1
    keep = ndimage.maximum_filter(conf, size=k + 4) >= conf_thr
    return {"E": E * keep, "band": band & keep, "ignore": conf < conf_thr}
