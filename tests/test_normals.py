"""Known-answer tests for normal-map construction (no data needed)."""

from __future__ import annotations

import numpy as np
import pytest

from data.degradation import build_degradation
from data.loaders import build_dataset
from data.norm import NORMAL_REGISTRY, ConstantNormalMap, CrossProductNormals, build_normal_map
from utils.config import load_config

H, W, F = 120, 160, 100.0
K = (F, F, (W - 1) / 2, (H - 1) / 2)


def _plane_depth(n, d=2.0):
    """Depth of the plane n . P = d seen by the pinhole K, n in the camera frame."""
    u, v = np.meshgrid(np.arange(W, dtype=np.float64), np.arange(H, dtype=np.float64))
    ray = np.stack([(u - K[2]) / F, (v - K[3]) / F, np.ones_like(u)], axis=-1)
    n = np.asarray(n, dtype=np.float64)
    return (d / (ray @ n)).astype(np.float32)


def test_registry_and_config():
    assert {"constant", "cross"} <= set(NORMAL_REGISTRY)
    assert isinstance(
        build_normal_map(dict(load_config("normals", "gt_cross"))), CrossProductNormals
    )
    assert build_normal_map(None) is None and build_normal_map({"type": "none"}) is None
    with pytest.raises(KeyError):
        build_normal_map({"type": "nope"})


def test_fronto_parallel_plane():
    out = CrossProductNormals()(np.full((H, W), 2.0, np.float32), intrinsics=K)
    m = out["normal_mask"]
    assert m[1:-1, 1:-1].all() and not m[0].any() and not m[:, -1].any()
    np.testing.assert_allclose(out["normals"][m], np.tile([0, 0, -1], (m.sum(), 1)), atol=1e-5)
    assert out["meta"]["intrinsics_source"] == "call"


@pytest.mark.parametrize("n", [(0.0, -0.5, -1.0), (0.3, 0.0, -1.0), (0.2, -0.6, -1.0)])
def test_tilted_plane_matches_analytic_normal(n):
    n = np.asarray(n) / np.linalg.norm(n)
    out = CrossProductNormals(depth_edge_rel=0.2)(_plane_depth(-n), intrinsics=K)
    m = out["normal_mask"]
    assert m[1:-1, 1:-1].all()
    np.testing.assert_allclose(out["normals"][m], np.tile(n, (m.sum(), 1)), atol=2e-3)


def test_sphere_normals_point_along_radius():
    r, c = 1.0, np.array([0.0, 0.0, 3.0])
    u, v = np.meshgrid(np.arange(W, dtype=np.float64), np.arange(H, dtype=np.float64))
    ray = np.stack([(u - K[2]) / F, (v - K[3]) / F, np.ones_like(u)], axis=-1)
    ray /= np.linalg.norm(ray, axis=-1, keepdims=True)
    b = ray @ c
    disc = b**2 - (c @ c - r**2)
    t = np.where(disc > 0, b - np.sqrt(np.maximum(disc, 0)), 0.0)
    pts = ray * t[..., None]
    gt = pts[..., 2].astype(np.float32)
    expected = (pts - c) / r  # outward = facing the camera for the visible cap
    out = CrossProductNormals(depth_edge_rel=0.2)(gt, intrinsics=K)
    m = out["normal_mask"] & (disc > 0.2)  # away from the silhouette
    assert m.sum() > 500
    cos = np.sum(out["normals"][m] * expected[m], axis=-1)
    assert cos.min() > 0.999


def test_holes_and_depth_steps_are_invalid():
    gt = np.full((H, W), 2.0, np.float32)
    gt[:, 80:] = 3.0  # step edge
    gt[50:55, 20:25] = 0.0  # hole
    out = CrossProductNormals()(gt, intrinsics=K)
    m, n = out["normal_mask"], out["normals"]
    assert not m[:, 79:81].any()  # columns next to the step
    assert not m[49:56, 20:25].any() and not m[50:55, 19:26].any()  # hole + its 4-neighbour ring
    assert m[10, 10] and m[10, 120]
    assert (n[~m] == 0).all()


def test_output_properties_and_determinism():
    gt = _plane_depth(np.array([0.0, 0.4, 1.0]) / np.linalg.norm([0.0, 0.4, 1.0]))
    rgb = np.random.default_rng(0).random((H, W, 3)).astype(np.float32)
    est = CrossProductNormals(depth_edge_rel=0.2, smooth_sigma=1.0)
    a, b, c = est(gt, rgb), est(gt, rgb), est(gt)
    for key in ("normals", "normal_mask"):
        np.testing.assert_array_equal(a[key], b[key])
        np.testing.assert_array_equal(a[key], c[key])  # rgb is ignored by the baseline
    m = a["normal_mask"]
    assert a["normals"].dtype == np.float32 and m.any()
    np.testing.assert_allclose(np.linalg.norm(a["normals"][m], axis=-1), 1.0, atol=1e-5)
    assert (a["normals"][m][:, 2] < 0).all()
    assert a["meta"]["intrinsics_source"] == "config"


def test_constant_normal_map():
    gt = np.zeros((H, W), np.float32)
    gt[10:20, 10:20] = 1.5
    out = ConstantNormalMap(normal=(0.0, -2.0, 0.0))(gt)
    np.testing.assert_array_equal(out["normal_mask"], gt > 0)
    np.testing.assert_allclose(out["normals"][gt > 0], np.tile([0, -1, 0], (100, 1)), atol=1e-6)
    assert (out["normals"][gt == 0] == 0).all()
    with pytest.raises(ValueError):
        ConstantNormalMap(normal=(0.0, 0.0, 0.0))


def test_shape_validation():
    with pytest.raises(ValueError):
        CrossProductNormals()(np.zeros((3, H, W), np.float32))
    with pytest.raises(ValueError):
        CrossProductNormals()(np.ones((H, W), np.float32), rgb=np.zeros((H, W), np.float32))


def test_on_synthetic_stairs_sample():
    deg = build_degradation(dict(load_config("degradation", "astra2_uncalibrated")))
    cfg = dict(load_config("dataset", "synthetic_stairs"))
    sample = build_dataset(cfg, degradation=deg, max_samples=2)[0]
    out = build_normal_map(dict(load_config("normals", "gt_cross")))(
        sample["gt_depth"], sample["rgb"]
    )
    assert out["normals"].shape == (*sample["gt_depth"].shape, 3)
    assert out["normal_mask"].mean() > 0.3
    m = out["normal_mask"]
    np.testing.assert_allclose(np.linalg.norm(out["normals"][m], axis=-1), 1.0, atol=1e-5)
