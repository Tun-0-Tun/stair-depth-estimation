"""audit_gt must recover a known noise level, hole share and quantisation step."""

from __future__ import annotations

import sys

import numpy as np

sys.path.insert(0, "scripts")

from audit_gt import audit_frames


def _plane(h=240, w=320, z0=2.0, tilt=0.002):
    xx = np.arange(w)[None, :] * tilt
    return (z0 + xx + np.zeros((h, 1))).astype(np.float64)


def test_recovers_sensor_noise_holes_and_quantisation():
    rng = np.random.default_rng(1)
    clean = _plane()
    noisy = [clean + rng.normal(0, 0.010, clean.shape) for _ in range(6)]  # 10 mm sensor noise
    for g in noisy:
        g[:60] = 0.0  # a quarter of the frame is holes
    out = audit_frames(noisy, 0.3, 8.0)
    assert abs(out["holes_pct"] - 25.0) < 0.5
    # 7x7 plane fit leaves sqrt(46/49) of the noise: ~9.7 mm at 2 m
    assert 8.5 < out["noise_mm_1.5-3m"] < 10.5, out

    laser = audit_frames([clean], 0.3, 8.0)
    assert laser["noise_mm_1.5-3m"] < 0.1, "a smooth plane is noise-free"

    q = np.round(clean / 0.017) * 0.017  # 17 mm quantisation, like rgbd_stair
    assert 15 < audit_frames([q], 0.3, 8.0)["step_mm"] < 19
