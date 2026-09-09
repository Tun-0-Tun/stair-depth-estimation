"""Small shared helpers: seeding, device resolution, resizing.

Resizing lives here rather than in each loader/adapter because getting depth
resizing wrong is a silent accuracy bug: depth must never be interpolated
across a discontinuity, and sparse depth must never be interpolated at all.
"""

from __future__ import annotations

import os
import random
from pathlib import Path
from typing import Any

import numpy as np

__all__ = [
    "DATA_ROOT_ENV",
    "data_root",
    "REPO_ROOT",
    "downsample_depth",
    "resize_depth",
    "resize_rgb",
    "resolve_device",
    "set_seed",
    "sparse_to_dense_nn",
]

REPO_ROOT = Path(__file__).resolve().parent.parent

#: Where the datasets live. They are shared between several people and are far
#: too big for the repo, so they sit in one external folder that everyone
#: mounts at their own path. Set STAIR_DATA_ROOT to point at it; configs then
#: name datasets relatively ("void_1500", "MinJiang-Dataset") and stay portable.
DATA_ROOT_ENV = "STAIR_DATA_ROOT"


def data_root() -> Path:
    """The external dataset folder. Defaults to ``<repo>/../data``."""
    return Path(os.environ.get(DATA_ROOT_ENV, REPO_ROOT.parent / "data")).expanduser()


def set_seed(seed: int, deterministic: bool = True) -> None:
    """Seed python / numpy / torch.  Call once at the top of every entry point."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        if deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    except ImportError:
        pass


def resolve_device(requested: str | None = None) -> str:
    """``"auto"``/``None`` -> best available; otherwise validate the request."""
    try:
        import torch
    except ImportError:
        return "cpu"

    if requested in (None, "auto"):
        if torch.cuda.is_available():
            return "cuda"
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    dev = str(requested)
    if dev.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(
            f"device={dev!r} requested but torch reports no CUDA. "
            "Use device=cpu / device=mps, or fix the CUDA install."
        )
    if dev == "mps" and not (
        getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()
    ):
        raise RuntimeError("device='mps' requested but MPS is unavailable")
    return dev


def _torch_interp(arr: np.ndarray, size: tuple[int, int], mode: str) -> np.ndarray:
    import torch
    import torch.nn.functional as F

    t = torch.from_numpy(np.ascontiguousarray(arr)).float()
    squeeze_channels = t.ndim == 2
    # (H, W) -> (1, 1, H, W);  (H, W, C) -> (1, C, H, W)
    t = t[None, None] if squeeze_channels else t.permute(2, 0, 1)[None]
    kwargs: dict[str, Any] = {"size": size, "mode": mode}
    if mode in ("bilinear", "bicubic"):
        kwargs["align_corners"] = False
    out = F.interpolate(t, **kwargs)
    out = out[0, 0] if squeeze_channels else out[0].permute(1, 2, 0)
    return out.numpy()


def resize_rgb(rgb: np.ndarray, size: tuple[int, int], mode: str = "bilinear") -> np.ndarray:
    """Resize an ``(H, W, 3)`` float image to ``size = (H_out, W_out)``."""
    if rgb.shape[:2] == tuple(size):
        return rgb
    return _torch_interp(rgb.astype(np.float32), tuple(size), mode)


def resize_depth(depth: np.ndarray, size: tuple[int, int], mode: str = "nearest") -> np.ndarray:
    """Resize an ``(H, W)`` depth map.

    ``nearest`` by default: bilinear/bicubic interpolation of depth invents
    values that lie *between* a foreground and a background surface, i.e. on no
    surface at all, and it smears sparse zeros into the measurements.  Use
    ``mode="bilinear"`` only for already-dense predictions.
    """
    if depth.shape[:2] == tuple(size):
        return depth
    return _torch_interp(depth.astype(np.float32), tuple(size), mode)


def downsample_depth(depth: np.ndarray, scale: int, mode: str = "bicubic") -> np.ndarray:
    """Factor-``scale`` downsampling used by the depth super-resolution setting."""
    h, w = depth.shape[:2]
    return resize_depth(depth, (h // scale, w // scale), mode=mode)


def sparse_to_dense_nn(sparse: np.ndarray, max_dist: int | None = None) -> np.ndarray:
    """Fill zeros in a sparse depth map with the nearest non-zero value.

    Used by adapters whose upstream model expects a *dense* low-quality depth
    (DuCos) while our loader produced a *sparse* one (dToF).  ``max_dist`` in
    pixels leaves farther holes at zero.
    """
    from scipy import ndimage

    known = sparse > 0
    if not known.any():
        return np.zeros_like(sparse)
    dist, (iy, ix) = ndimage.distance_transform_edt(~known, return_indices=True)
    filled = sparse[iy, ix]
    if max_dist is not None:
        filled = np.where(dist <= max_dist, filled, 0.0)
    return filled.astype(np.float32)
