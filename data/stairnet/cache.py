"""Offline StairNet outputs on disk, one ``<sample_id>.npz`` per frame, in the loader's frame.

``cls``  uint8 (H, W): 0 background, 1 tread, 2 riser (canonical ids, mapped from the net's
         channels by ``StairNetSeg.class_map``: net channel 2 -> tread, 1 -> riser)
``conf`` float16 (H, W): softmax confidence of the arg-max class
``lines`` float32 (N, 6): class (0 convex / 1 concave), x1, y1, x2, y2, confidence, frame pixels
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np

#: set this when the data folder is read-only for you: every script then reads/writes the masks there
CACHE_ENV = "STAIRNET_CACHE_DIR"


def cache_dir(relative: str = "ours/stairnet_cache") -> Path:
    """``$STAIRNET_CACHE_DIR`` if set, else ``$STAIR_DATA_ROOT/<relative>`` (an absolute ``relative`` wins)."""
    from utils.misc import data_root

    if os.environ.get(CACHE_ENV):
        return Path(os.environ[CACHE_ENV]).expanduser()
    return data_root() / relative


def save_masks(cache_dir, sample_id, cls, conf, lines=None) -> None:
    d = Path(cache_dir)
    d.mkdir(parents=True, exist_ok=True)
    lines = np.zeros((0, 6), np.float32) if lines is None else np.asarray(lines, np.float32)
    np.savez_compressed(
        d / f"{sample_id}.npz",
        cls=cls.astype(np.uint8),
        conf=conf.astype(np.float16),
        lines=lines,
    )


def load_masks(cache_dir, sample_id, shape=None):
    """Dict with ``cls`` / ``conf`` (float32) / ``lines``, or ``None`` if not cached.

    Raises if the cached frame has another size than ``shape`` -- a stale cache from another
    crop/target_size must never be silently mis-aligned with the depth.
    """
    f = Path(cache_dir) / f"{sample_id}.npz"
    if not f.exists():
        return None
    with np.load(f) as z:
        out = {"cls": z["cls"], "conf": z["conf"].astype(np.float32), "lines": z["lines"]}
    if shape is not None and out["cls"].shape != tuple(shape):
        raise ValueError(
            f"{f}: cached masks are {out['cls'].shape}, the frame is {tuple(shape)}; "
            "re-run scripts/stairnet_cache.py with the current crop/target_size"
        )
    return out
