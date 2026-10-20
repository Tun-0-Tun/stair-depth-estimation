"""Tiny synthetic ``ours``-style folder (staircases ray-cast analytically) for pipeline tests / smoke runs."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent))


def make(root: Path, n_stairs: int = 6, shots: int = 2, h: int = 128, w: int = 176) -> Path:
    import pandas as pd

    import test_stair_losses as T

    rng = np.random.default_rng(0)
    rows, sid = [], 0
    K = (T.K[0] * w / T.W, T.K[1] * h / T.H, T.K[2] * w / T.W, T.K[3] * h / T.H)
    root.mkdir(parents=True, exist_ok=True)
    for st in range(n_stairs):
        for _ in range(shots):
            sid += 1
            old = T.K, T.H, T.W
            T.K, T.H, T.W = K, h, w
            try:
                z, tr, ri = T.staircase(z0=float(rng.uniform(1.8, 3.0)), rise=float(rng.uniform(0.15, 0.19)))
            finally:
                T.K, T.H, T.W = old
            rgb = np.where(tr[..., None], [150, 120, 90], np.where(ri[..., None], [90, 90, 110], [60, 60, 60])).astype(np.uint8)
            rgb = np.clip(rgb + rng.integers(-12, 12, rgb.shape), 0, 255).astype(np.uint8)
            Image.fromarray(rgb).save(root / f"s{sid}_rgb.png")
            Image.fromarray((z * 1000).astype(np.uint16)).save(root / f"s{sid}_depth_aligned_to_rgb.png")
            Image.fromarray(((z > 0) * 255).astype(np.uint8)).save(root / f"s{sid}_mask_aligned_valid.png")
            rows.append({"shot id": sid, "stairs id": st + 1, "view category": "front", "illumination": "day",
                         "occlusion": "none", "comments": ""})
    pd.DataFrame(rows).to_excel(root / "shots.xlsx", index=False)
    return root
