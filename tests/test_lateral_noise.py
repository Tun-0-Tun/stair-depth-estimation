import numpy as np
import pytest

from data.degradation import LateralDegradation, build_degradation


def _normal(h, w, deg):
    """Constant normal at ``deg`` degrees from the optical axis."""
    t = np.radians(deg)
    n = np.zeros((h, w, 3), np.float32)
    n[..., 1], n[..., 2] = np.sin(t), np.cos(t)
    return n


def _step(h=200, w=200, near=1.0, far=3.0):
    gt = np.full((h, w), far, np.float32)
    gt[:, : w // 2] = near
    return gt


def test_seeded():
    gt = _step(32, 32)
    n = _normal(32, 32, 20)
    deg = LateralDegradation()
    a = deg(gt, seed=1, normal=n)["sparse_depth"]
    np.testing.assert_array_equal(a, deg(gt, seed=1, normal=n)["sparse_depth"])
    assert not np.array_equal(a, deg(gt, seed=2, normal=n)["sparse_depth"])


def test_nearest_only_no_invented_depths():
    gt = _step()
    out = LateralDegradation()(gt, seed=0, normal=_normal(*gt.shape, 30))["sparse_depth"]
    assert set(np.unique(out)) <= {1.0, 3.0}
    assert (out != gt).any()  # the edge actually became ragged


def test_flat_depth_unchanged():
    gt = np.full((40, 40), 2.0, np.float32)
    out = LateralDegradation()(gt, seed=0, normal=_normal(40, 40, 50))["sparse_depth"]
    np.testing.assert_array_equal(out, gt)


def test_sigma_matches_formula():
    # Flip probability of the pixel at distance d from a step edge is P(|dx| > d - 0.5)-ish;
    # compare the measured std of the horizontal offsets via a depth ramp instead:
    # depth = column index / 100 + 1, so the output encodes the sampled column exactly.
    h, w = 400, 400
    gt = (1.0 + np.tile(np.arange(w), (h, 1)) / 100.0).astype(np.float32)
    cols = np.arange(w)[None, :]
    for deg, expected in [(0, 0.8), (60, 0.8 + 0.035 * (np.pi / 3) / (np.pi / 6))]:
        out = LateralDegradation()(gt, seed=0, normal=_normal(h, w, deg))["sparse_depth"]
        sampled = np.rint((out - 1.0) * 100.0)
        dx = (sampled - cols)[:, 20:-20]  # skip border clipping
        # rounding to integer pixels adds ~1/12 variance
        assert np.sqrt(dx.var() - 1 / 12) == pytest.approx(expected, rel=0.1)


def test_grazing_angle_increases_offsets_and_stays_finite():
    gt = _step()
    a = LateralDegradation()(gt, seed=0, normal=_normal(*gt.shape, 20))["sparse_depth"]
    b = LateralDegradation()(gt, seed=0, normal=_normal(*gt.shape, 89))["sparse_depth"]
    assert (b != gt).sum() > (a != gt).sum()
    out = LateralDegradation()(gt, seed=0, normal=_normal(*gt.shape, 90))["sparse_depth"]
    assert np.isfinite(out).all()


def test_correlation_smooths_offsets_keeps_sigma():
    h, w = 400, 400
    gt = (1.0 + np.tile(np.arange(w), (h, 1)) / 100.0).astype(np.float32)
    cols = np.arange(w)[None, :]
    n = _normal(h, w, 0)

    def offsets(length):
        out = LateralDegradation(gauss_std=length)(gt, seed=0, normal=n)["sparse_depth"]
        return (np.rint((out - 1.0) * 100.0) - cols)[:, 20:-20]

    iid, corr = offsets(0.0), offsets(3.0)

    def neighbour_corr(d):
        return np.corrcoef(d[:, :-1].ravel(), d[:, 1:].ravel())[0, 1]

    assert abs(neighbour_corr(iid)) < 0.1
    assert neighbour_corr(corr) > 0.5
    assert corr.std() == pytest.approx(iid.std(), rel=0.2)


def test_invalid_stays_invalid_and_range():
    gt = np.full((40, 40), 2.0, np.float32)
    gt[:10] = 0.0
    gt[10:20] = 20.0  # far but valid: no range clipping here
    out = LateralDegradation()(gt, seed=0, normal=_normal(40, 40, 10))["sparse_depth"]
    assert (out[:6] == 0).all()  # well inside the hole
    assert (out[24:] > 0).all()


def test_normal_is_upscaled_and_holes_filled():
    gt = np.full((64, 64), 2.0, np.float32)
    n = _normal(32, 32, 30)
    n[5:9, 5:9] = np.nan
    res = LateralDegradation()(gt, seed=0, normal=n)
    assert res["sparse_depth"].shape == (64, 64)
    assert res["meta"]["theta_missing_ratio"] > 0
    assert res["meta"]["calibrated"] is False


def test_bad_shape_raises():
    gt = np.full((16, 16), 2.0, np.float32)
    with pytest.raises(ValueError):
        LateralDegradation()(gt, seed=0, normal=_normal(16, 8, 10))


def test_buildable():
    assert isinstance(build_degradation({"type": "lateral_noise"}), LateralDegradation)
