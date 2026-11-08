import numpy as np
import pytest

from data.degradation import SequentialDegradation, ThetaDropoutDegradation, build_degradation


def _normal(h, w, deg):
    """Constant normal at ``deg`` degrees from the optical axis."""
    t = np.radians(deg)
    n = np.zeros((h, w, 3), np.float32)
    n[..., 1], n[..., 2] = np.sin(t), np.cos(t)
    return n


def _missing(deg, seed=0, size=200, **kw):
    gt = np.full((size, size), 2.0, np.float32)
    deg_model = ThetaDropoutDegradation(**kw)
    out = deg_model(gt, seed=seed, normal=_normal(size, size, deg))["sparse_depth"]
    return float((out == 0).mean())


def test_seeded():
    gt = np.full((32, 32), 2.0, np.float32)
    n = _normal(32, 32, 75)
    deg = ThetaDropoutDegradation()
    a = deg(gt, seed=1, normal=n)["sparse_depth"]
    np.testing.assert_array_equal(a, deg(gt, seed=1, normal=n)["sparse_depth"])
    assert not np.array_equal(a, deg(gt, seed=2, normal=n)["sparse_depth"])


@pytest.mark.parametrize("angle", [0, 30, 60])
def test_base_rate_up_to_theta0(angle):
    assert _missing(angle, p_min=0.0) == 0.0
    assert _missing(angle, p_min=0.1) == pytest.approx(0.1, abs=0.01)


@pytest.mark.parametrize("angle", [62, 65, 70, 80])
def test_linear_growth_above_theta0(angle):
    expected = 0.05 + 0.02 * (angle - 60)
    got = _missing(angle, theta0=60.0, p_min=0.05, eta=0.02)
    assert got == pytest.approx(expected, abs=0.01)


def test_default_slope_is_steeper_than_one_percent_per_degree():
    assert _missing(70) == pytest.approx(0.4, abs=0.01)


def test_capped_at_p_max():
    assert _missing(85, theta0=60.0, eta=0.05, p_max=0.4) == pytest.approx(0.4, abs=0.01)


def test_invalid_stays_zero_and_survivors_unchanged():
    gt = np.full((64, 64), 2.0, np.float32)
    gt[:10] = 0.0
    out = ThetaDropoutDegradation()(gt, seed=0, normal=_normal(64, 64, 80))["sparse_depth"]
    assert (out[:10] == 0).all()
    assert set(np.unique(out)) <= {0.0, 2.0}


def test_missing_normal_raises():
    with pytest.raises(ValueError):
        ThetaDropoutDegradation().apply(np.ones((8, 8), np.float32), None, np.random.default_rng(0))


def test_nan_normals_are_filled():
    n = _normal(32, 32, 75)
    n[:8, :8] = np.nan
    meta = ThetaDropoutDegradation()(np.full((32, 32), 2.0, np.float32), seed=0, normal=n)["meta"]
    assert meta["theta_missing_ratio"] > 0
    assert meta["calibrated"] is False


def test_builds_and_chains():
    assert isinstance(build_degradation({"type": "theta_dropout"}), ThetaDropoutDegradation)
    out = SequentialDegradation(steps=["axial_noise", "theta_dropout"])(
        np.full((32, 32), 2.0, np.float32), normal=_normal(32, 32, 80), seed=0
    )
    assert out["sparse_depth"].shape == (32, 32)
    assert (out["sparse_depth"] == 0).any()
