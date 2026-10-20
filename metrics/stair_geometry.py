"""Step geometry error (StairNetV2-style): height and depth (run) of the 3 steps nearest the camera.

Needs the intrinsics ``K`` and tread/riser masks.  Everything is measured in the "up" frame of
the GT: up = robust mean of the GT tread normals; height = P.up, depth coordinate = P.fwd with
fwd = camera z projected on the horizontal plane.  Treads are the connected components of the
tread mask, ordered by distance; ``rise_k`` = height(tread k+1) - height(tread k) (k = 0..2),
``run_k`` = extent (max - min; sensitive to mask boundary pixels) of tread k along fwd, k = 0..2.  The same masks are used
for prediction and GT, so the error isolates the depth error (the masks themselves are
StairNet's, not ground truth).
"""

from __future__ import annotations

import numpy as np
import torch
from scipy import ndimage

from models.planar_loss import normals_from_depth, robust_up


def _points(depth: np.ndarray, K) -> np.ndarray:
    fx, fy, cx, cy = K
    v, u = np.mgrid[: depth.shape[0], : depth.shape[1]].astype(np.float64)
    return np.stack([(u - cx) / fx * depth, (v - cy) / fy * depth, depth], -1)


def _treads(tread: np.ndarray, depth: np.ndarray, min_px: int):
    lab, n = ndimage.label(tread)
    comps = [lab == i for i in range(1, n + 1) if (lab == i).sum() >= min_px]
    return sorted(comps, key=lambda m: float(np.median(depth[m])))


def up_vector(gt: np.ndarray, tread: np.ndarray, K, min_px: int = 200) -> np.ndarray | None:
    t = torch.from_numpy(gt.astype(np.float32))[None, None]
    n, ok = normals_from_depth(t, K, t > 0)
    up, has = robust_up(n, torch.from_numpy(tread)[None, None] & ok, min_px)
    return up[0].numpy().astype(np.float64) if has[0] else None


def step_geometry(depth, tread, K, up, n_steps: int = 3, min_px: int = 200):
    """``{"rise": [n], "run": [n]}`` (NaN where a step is missing)."""
    P = _points(depth, K)
    fwd = np.array([0.0, 0.0, 1.0]) - up * up[2]
    fwd /= np.linalg.norm(fwd)
    comps = _treads(tread & (depth > 0), depth, min_px)
    h = [float(np.median(P[m] @ up)) for m in comps]
    rise = [h[k + 1] - h[k] if k + 1 < len(h) else np.nan for k in range(n_steps)]
    run = []
    for k in range(n_steps):
        if k < len(comps):
            c = P[comps[k]] @ fwd
            run.append(float(c.max() - c.min()))
        else:
            run.append(np.nan)
    return {"rise": np.array(rise), "run": np.array(run)}


def step_geometry_error(pred, gt, tread, K, n_steps: int = 3) -> dict[str, np.ndarray] | None:
    """Absolute (m) and relative errors of rise/run for the nearest steps; ``None`` without a usable up vector."""
    up = up_vector(gt, tread & (gt > 0), K)
    if up is None:
        return None
    g, p = step_geometry(gt, tread, K, up, n_steps), step_geometry(pred, tread, K, up, n_steps)
    out = {}
    for k in ("rise", "run"):
        out[f"{k}_abs"] = np.abs(p[k] - g[k])
        out[f"{k}_rel"] = np.abs(p[k] - g[k]) / np.maximum(np.abs(g[k]), 1e-6)
    return out
