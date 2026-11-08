import numpy as np
import pytest

from data.degradation import AxialDegradation, IdentityDegradation, build_degradation
from data.norm import get_theta_map
from utils.misc import fill_nan_nearest


def _normal(h, w, deg):
    """Constant normal at ``deg`` degrees from the optical axis."""
    t = np.radians(deg)
    n = np.zeros((h, w, 3), np.float32)
    n[..., 1], n[..., 2] = np.sin(t), np.cos(t)
    return n


def _std(depth, deg, seed=0, size=200):
    gt = np.full((size, size), depth, np.float32)
    out = AxialDegradation()(gt, seed=seed, normal=_normal(size, size, deg))["sparse_depth"]
    return out.std()


def test_seeded():
    gt = np.full((32, 32), 2.0, np.float32)
    n = _normal(32, 32, 20)
    deg = AxialDegradation()
    a = deg(gt, seed=1, normal=n)["sparse_depth"]
    np.testing.assert_array_equal(a, deg(gt, seed=1, normal=n)["sparse_depth"])
    assert not np.array_equal(a, deg(gt, seed=2, normal=n)["sparse_depth"])


@pytest.mark.parametrize("depth", [1.0, 3.0])
def test_std_matches_formula(depth):
    expected = 0.0012 + 0.0019 * (depth - 0.4) ** 2
    assert _std(depth, 30) == pytest.approx(expected, rel=0.05)


def test_grazing_angle_increases_noise_and_stays_finite():
    assert _std(2.0, 80) > _std(2.0, 30)
    gt = np.full((50, 50), 2.0, np.float32)
    out = AxialDegradation()(gt, seed=0, normal=_normal(50, 50, 90))["sparse_depth"]
    assert np.isfinite(out).all()
    assert np.abs(out - 2.0).max() < 8 * 0.1  # sigma capped at sigma_max


def test_invalid_stays_zero_and_no_range_clip():
    gt = np.full((40, 40), 2.0, np.float32)
    gt[:10] = 0.0
    gt[10:20] = 20.0  # far depth: no operating-range clip in this step
    out = AxialDegradation()(gt, seed=0, normal=_normal(40, 40, 10))["sparse_depth"]
    assert (out[:10] == 0).all() and (out[10:] > 0).all()


def test_normal_is_upscaled_and_holes_filled():
    gt = np.full((64, 64), 2.0, np.float32)
    n = _normal(32, 32, 30)
    n[5:9, 5:9] = np.nan
    res = AxialDegradation()(gt, seed=0, normal=n)
    assert res["sparse_depth"].shape == (64, 64)
    assert res["meta"]["theta_missing_ratio"] > 0
    theta = get_theta_map(n, (64, 64))
    assert np.isnan(theta).any()
    assert np.nanmax(np.abs(theta - np.radians(30))) < 1e-5


def test_fill_only_touches_valid_depth_pixels():
    theta = np.full((8, 8), 0.5, np.float32)
    theta[2:6, 2:6] = np.nan
    valid = np.ones((8, 8), bool)
    valid[2:4, 2:6] = False
    out = fill_nan_nearest(theta, valid)
    assert np.isnan(out[2:4, 2:6]).all()
    assert np.isfinite(out[4:6, 2:6]).all()


def _tilted_plane(deg, h=100, w=100, z0=2.0, fy=570.0):
    """Plane whose normal is ``deg`` degrees from the optical axis (tilted about x)."""
    s = np.tan(np.radians(deg))
    v = np.arange(h, dtype=np.float64)[:, None] - (h - 1) / 2
    return np.broadcast_to(z0 / (1.0 - s * v / fy), (h, w)).astype(np.float32)


def test_direct_apply_without_normal_raises():
    with pytest.raises(ValueError):
        AxialDegradation().apply(np.full((8, 8), 2.0, np.float32), None, np.random.default_rng(0))


def test_bad_normal_shape_raises():
    gt = np.full((16, 16), 2.0, np.float32)
    with pytest.raises(ValueError):
        AxialDegradation()(gt, seed=0, normal=_normal(16, 8, 10))


def test_other_degradations_unaffected_and_buildable():
    gt = np.full((8, 8), 1.0, np.float32)
    np.testing.assert_array_equal(
        IdentityDegradation()(gt, normal=_normal(8, 8, 0))["sparse_depth"], gt
    )
    assert isinstance(build_degradation({"type": "axial_noise"}), AxialDegradation)
