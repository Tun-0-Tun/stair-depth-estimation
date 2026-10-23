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
        out = LateralDegradation(z_min=0.0, z_max=10.0)(gt, seed=0, normal=_normal(h, w, deg))[
            "sparse_depth"
        ]
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
        out = LateralDegradation(z_min=0.0, z_max=10.0, correlation_length_px=length)(
            gt, seed=0, normal=n
        )["sparse_depth"]
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
    gt[10:20] = 20.0  # beyond z_max
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


def _two_depth(h=300, w=400, near=1.0, far=4.0):
    z = np.full((h, w), far, np.float64)
    z[:, : w // 2] = near
    return z


def test_correlation_length_map_follows_power_law_and_clips():
    from data.degradation.lateral import correlation_length_map

    z = np.array([[0.0, 1.0, 2.0, 4.0, 100.0]])
    kw = {"ref_depth_m": 2.0, "min_px": 0.5, "max_px": 16.0}
    # p = 1: l = 4 * 2 / z; invalid pixel uses z_ref -> l_ref
    ell = correlation_length_map(z, 4.0, exponent=1.0, **kw)
    np.testing.assert_allclose(ell, [[4.0, 8.0, 4.0, 2.0, 0.5]])
    # p = 0: constant
    np.testing.assert_allclose(correlation_length_map(z, 4.0, exponent=0.0, **kw), 4.0)
    # p < 0: grows with depth, clipped at max
    assert correlation_length_map(z, 4.0, exponent=-1.0, **kw)[0, -1] == 16.0


def test_varying_correlation_is_longer_where_ell_is_longer_and_keeps_unit_std():
    from data.degradation.lateral import _correlate_unit_varying

    h, w = 300, 400
    field = np.random.default_rng(0).normal(size=(h, w))
    ell = np.full((h, w), 0.5)
    ell[:, w // 2 :] = 6.0
    out = _correlate_unit_varying(field, ell, 6, 0.5, 16.0)

    def nb_corr(a):
        return np.corrcoef(a[:, :-1].ravel(), a[:, 1:].ravel())[0, 1]

    short, long_ = out[:, 20 : w // 2 - 20], out[:, w // 2 + 20 : -20]
    assert nb_corr(long_) > nb_corr(short) + 0.3
    assert short.std() == pytest.approx(1.0, rel=0.15)
    assert long_.std() == pytest.approx(1.0, rel=0.15)


def test_depth_exponent_zero_matches_constant_length_path():
    from data.degradation.lateral import apply_lateral_noise

    z = _two_depth()
    theta = np.zeros_like(z)
    kw = {
        "sigma_floor_px": 2.0,
        "sigma_angle_gain": 0.1,
        "theta_max_deg": 89.0,
        "sigma_max_px": 10.0,
    }
    a = apply_lateral_noise(z, theta, np.random.default_rng(3), correlation_length_px=4.0, **kw)
    b = apply_lateral_noise(
        z,
        theta,
        np.random.default_rng(3),
        correlation_length_px=4.0,
        correlation_depth_exponent=0.0,
        **kw,
    )
    np.testing.assert_array_equal(a, b)


def test_depth_dependent_correlation_runs_and_is_seeded():
    gt = _two_depth().astype(np.float32)
    n = _normal(*gt.shape, 30)
    deg = LateralDegradation(
        correlation_length_px=4.0, correlation_depth_exponent=1.0, z_min=None, z_max=None
    )
    a = deg(gt, seed=1, normal=n)["sparse_depth"]
    np.testing.assert_array_equal(a, deg(gt, seed=1, normal=n)["sparse_depth"])
    assert set(np.unique(a)) <= {1.0, 4.0}


def test_bad_correlation_params_raise():
    with pytest.raises(ValueError):
        LateralDegradation(correlation_min_px=0.0)
    with pytest.raises(ValueError):
        LateralDegradation(correlation_min_px=5.0, correlation_max_px=1.0)
    with pytest.raises(ValueError):
        LateralDegradation(correlation_levels=1)
    with pytest.raises(ValueError):
        LateralDegradation(correlation_ref_depth_m=0.0)
