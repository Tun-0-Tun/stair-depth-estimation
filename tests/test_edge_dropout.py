import numpy as np
import pytest
from scipy import ndimage

from data.degradation import EdgeDropoutDegradation, build_degradation
from data.degradation.edge_dropout import (
    _unit_normals,
    clean_edge_mask,
    edge_patches,
    edge_pixel_mask,
    near_edge_mask,
    plane_edge_mask,
)

F = 570.0


def _plane_edges(z, n, **kw):
    h, w = z.shape
    return plane_edge_mask(
        z.astype(np.float64),
        _unit_normals(n, z.shape),
        z > 0,
        fx=F,
        fy=F,
        cx=(w - 1) / 2,
        cy=(h - 1) / 2,
        **kw,
    )


def _tilted_plane(deg, h=100, w=160, d=1.0):
    """Plane n.P = d with unit normal (0, sin t, cos t) in the camera frame, seen by a pinhole."""
    t = np.radians(deg)
    vs = np.arange(h)[:, None] * np.ones((1, w))
    z = d / (np.cos(t) + np.sin(t) * (vs - (h - 1) / 2) / F)
    n = np.zeros((h, w, 3))
    n[..., 1], n[..., 2] = np.sin(t), np.cos(t)
    return z.astype(np.float32), n.astype(np.float32)


def _two_planes(h=96, w=128):
    """Left: near plane facing the camera. Right: far plane tilted 60 deg."""
    z = np.empty((h, w), np.float32)
    n = np.zeros((h, w, 3), np.float32)
    half = w // 2
    z[:, :half], n[:, :half, 2] = 1.5, 1.0
    t = np.radians(60)
    z[:, half:], n[:, half:, 1], n[:, half:, 2] = 3.0, np.sin(t), np.cos(t)
    return z, n


def _stairs(h=240, w=320):
    """Floor ramp plus three risers/treads: depth steps and tilted normals."""
    z = np.tile(np.linspace(1.0, 4.0, h, dtype=np.float32)[:, None], (1, w))
    n = np.zeros((h, w, 3), np.float32)
    n[..., 1], n[..., 2] = 0.3, 0.95
    for i, r in enumerate((80, 130, 180)):
        z[r : r + 25] += 0.6 * (i + 1)
        n[r : r + 25] = (0.0, 0.0, 1.0)
    return z, n


def _edges(z, n, tau_n=15.0, tau_z=0.05, window=3):
    return edge_pixel_mask(
        z.astype(np.float64), _unit_normals(n, z.shape), z > 0, tau_n, tau_z, window
    )


def test_seeded():
    z, n = _two_planes()
    deg = EdgeDropoutDegradation()
    a = deg(z, normal=n, seed=1)["sparse_depth"]
    np.testing.assert_array_equal(a, deg(z, normal=n, seed=1)["sparse_depth"])
    assert any(not np.array_equal(a, deg(z, normal=n, seed=s)["sparse_depth"]) for s in range(2, 8))


def test_output_contract():
    z, n = _two_planes()
    z[:10] = 0.0
    out = EdgeDropoutDegradation(p_drop=1.0, max_drop_frac=1.0)(z, normal=n, seed=0)
    s, lr = out["sparse_depth"], out["lr_depth"]
    assert s.shape == z.shape and s.dtype == np.float32
    np.testing.assert_array_equal(s, lr)
    assert np.isfinite(s).all() and (s >= 0).all()
    assert (s[z == 0] == 0).all()
    kept = s > 0
    np.testing.assert_array_equal(s[kept], z[kept])  # holes only remove, never alter
    assert (s == 0).sum() > (z == 0).sum()  # something was dropped
    assert out["meta"]["calibrated"] is False
    assert out["meta"]["hole_fraction"] == pytest.approx(1 - kept[z > 0].mean())


def test_edge_detector_fires_on_step_not_on_plane():
    z, n = _two_planes()
    edge = _edges(z, n)
    half = z.shape[1] // 2
    assert edge[:, half - 1 : half + 1].all()
    assert not edge[:, : half - 3].any() and not edge[:, half + 3 :].any()
    flat = np.full_like(z, 2.0)
    n_flat = np.zeros_like(n)
    n_flat[..., 2] = 1.0
    assert not _edges(flat, n_flat).any()


def test_plane_classifier_ignores_slope_that_fools_the_range_classifier():
    z, n = _tilted_plane(75)
    # tight threshold: the 3x3 depth range on this slanted plane (~1.3%) is flagged everywhere
    assert _edges(z, n, tau_z=0.005).mean() > 0.9
    assert not _plane_edges(z, n).any()  # a plane is a plane, whatever its tilt


def test_plane_classifier_marks_both_sides_of_a_depth_step():
    z = np.full((60, 80), 1.5, np.float32)
    z[:, 40:] = 3.0
    n = np.zeros((60, 80, 3), np.float32)
    n[..., 2] = 1.0
    edge = _plane_edges(z, n, step_px=2)
    assert edge[:, 38:42].all()  # two pixels each side (step_px)
    assert not edge[:, :37].any() and not edge[:, 43:].any()


def test_far_side_only_marks_only_the_far_side_of_a_step():
    z = np.full((60, 80), 1.5, np.float32)
    z[:, 40:] = 3.0
    n = np.zeros((60, 80, 3), np.float32)
    n[..., 2] = 1.0
    edge = _plane_edges(z, n, step_px=2, far_side_only=True)
    assert edge[:, 40:42].all()  # the far (right) side
    assert not edge[:, :40].any() and not edge[:, 43:].any()


def test_plane_classifier_is_robust_to_depth_noise():
    z, n = _tilted_plane(60, d=2.0)
    noisy = z + np.random.default_rng(0).normal(0.0, 0.001, z.shape).astype(np.float32)
    assert not _plane_edges(noisy, n).any()


def test_plane_classifier_finds_creases_through_the_normals():
    z = np.full((60, 80), 2.0, np.float32)
    n = np.zeros((60, 80, 3), np.float32)
    n[:, :40, 2] = 1.0
    t = np.radians(60)
    n[:, 40:, 1], n[:, 40:, 2] = np.sin(t), np.cos(t)
    edge = _plane_edges(z, n)  # no depth step at all
    assert edge[:, 36:44].all()  # within normal_step_px (4) of the crease, both sides
    assert not edge[:, :35].any() and not edge[:, 44:].any()


def _noisy_flat(deg_noise, h=200, w=240, seed=0):
    """Flat wall facing the camera: normals jitter by ``deg_noise`` degrees, depth by 1 mm."""
    rng = np.random.default_rng(seed)
    z = (2.0 + rng.normal(0.0, 0.001, (h, w))).astype(np.float32)
    n = np.zeros((h, w, 3))
    n[..., :2] = rng.normal(0.0, np.radians(deg_noise), (h, w, 2))
    n[..., 2] = 1.0
    return z, (n / np.linalg.norm(n, axis=-1, keepdims=True)).astype(np.float32)


def test_larger_support_removes_false_edges_on_a_noisy_flat_surface():
    z, n = _noisy_flat(8.0)
    old = _plane_edges(z, n, tau_n_deg=10.0, normal_step_px=1, normal_smooth_px=3, step_px=2)
    new = _plane_edges(z, n, tau_n_deg=10.0)  # larger defaults: step 4, normals 4 px apart, 5x5
    assert old.mean() > 0.01  # the old support flags a visible share of a flat wall
    assert new.mean() < old.mean() / 3  # larger support alone removes most of them
    assert clean_edge_mask(new, z > 0, 40, 1).mean() < 0.001  # and the speck filter the rest


def test_speck_filter_removes_isolated_pixels_but_keeps_long_edges():
    z = np.full((80, 120), 2.0, np.float32)
    z[:, 60:] = 3.0  # a long depth step
    z[20, 20] += 0.5  # one outlier pixel far from it
    n = np.zeros((80, 120, 3), np.float32)
    n[..., 2] = 1.0
    raw = _plane_edges(z, n)
    assert raw[10:30, 10:30].any()  # the outlier is flagged ...
    clean = clean_edge_mask(raw, z > 0, 40, 1)
    assert not clean[:, :50].any()  # ... and removed
    assert clean[:, 56:64].all()  # the 80-px step stays
    assert clean_edge_mask(raw, z > 0, 0, 0).sum() == raw.sum()  # no-op when disabled


def test_degradation_reports_the_speck_filter():
    z, n = _noisy_flat(8.0)
    meta = EdgeDropoutDegradation(tau_n_deg=10.0)(z, normal=n, seed=0)["meta"]
    assert meta["edge_fraction"] <= meta["edge_fraction_raw"]
    assert meta["n_patches"] >= 0


def _lines(rows, cols=(20, 80), shape=(120, 120)):
    edge = np.zeros(shape, bool)
    for r in rows:
        edge[r, cols[0] : cols[1]] = True
    return edge


def test_edge_pixels_are_grouped_by_distance():
    rng = np.random.default_rng
    far = edge_patches(_lines([20, 60]), 6.0, None, rng(0))  # 40 px apart
    near = edge_patches(_lines([20, 24]), 6.0, None, rng(0))  # 4 px apart
    assert far.max() == 2 and near.max() == 1
    assert ((far > 0) == _lines([20, 60])).all()  # labels sit on edge pixels only
    assert edge_patches(np.zeros((9, 9), bool), 6.0, None, rng(0)).max() == 0


def test_long_edge_groups_are_cut_into_short_and_long_patches():
    edge = np.zeros((40, 400), bool)
    edge[20, 10:390] = True  # one 380-px edge
    whole = edge_patches(edge, 6.0, None, np.random.default_rng(0))
    assert whole.max() == 1
    sizes = []
    for seed in range(10):
        lab = edge_patches(edge, 6.0, (20.0, 200.0), np.random.default_rng(seed))
        assert ((lab > 0) == edge).all()  # same pixels, only more labels
        sizes.extend(np.bincount(lab.ravel())[1:])
    sizes = np.array(sizes)
    assert sizes.max() / sizes.min() >= 3  # short and long patches side by side


def test_patch_is_dropped_as_a_whole_with_probability_p():
    z = np.full((160, 200), 2.0, np.float32)
    z[:, 100:] = 3.0  # one long depth step; no cutting so it is a single patch
    n = np.zeros((160, 200, 3), np.float32)
    n[..., 2] = 1.0
    kw = {"patch_len_px": (1e5, 1e5), "radius_jitter": 0.0, "max_drop_frac": 1.0}
    assert EdgeDropoutDegradation(p_drop=0.0, **kw)(z, normal=n, seed=0)["meta"]["n_dropped"] == 0
    none = EdgeDropoutDegradation(p_drop=0.0, **kw)(z, normal=n, seed=0)
    np.testing.assert_array_equal(none["sparse_depth"], z)
    everything = EdgeDropoutDegradation(p_drop=1.0, **kw)(z, normal=n, seed=0)["meta"]
    assert everything["n_patches"] == everything["n_dropped"] == 1
    hits = [
        EdgeDropoutDegradation(p_drop=0.5, **kw)(z, normal=n, seed=s)["meta"]["n_dropped"]
        for s in range(40)
    ]
    assert 0.3 < np.mean(hits) < 0.7 and set(hits) == {0, 1}


def test_only_dropped_patches_get_a_band():
    z = np.full((160, 300), 2.0, np.float32)
    z[:, 150:] = 3.0  # step 1
    z[:, 250:] = 2.0  # step 2, 100 px away: separate patch
    n = np.zeros((160, 300, 3), np.float32)
    n[..., 2] = 1.0
    deg = EdgeDropoutDegradation(
        p_drop=0.5, patch_len_px=(1e5, 1e5), link_px=6.0, radius_jitter=0.0, max_drop_frac=1.0
    )
    seen = set()
    for seed in range(30):
        out = deg(z, normal=n, seed=seed)
        hole = out["sparse_depth"] == 0
        first, second = hole[:, 140:160].any(), hole[:, 240:260].any()
        assert not hole[:, :100].any() and not hole[:, 180:220].any()  # nothing elsewhere
        assert out["meta"]["n_dropped"] == int(first) + int(second)
        seen.add((first, second))
    assert seen == {(False, False), (True, False), (False, True), (True, True)}


def test_near_edge_mask_is_a_distance_band():
    z, n = _two_planes()
    edge, valid = _edges(z, n), z > 0
    dist = ndimage.distance_transform_edt(~edge)
    near, radius = near_edge_mask(edge, valid, 5.0)
    assert radius == 5.0
    assert near[edge].all()  # edge pixels are close to an edge
    np.testing.assert_array_equal(near, dist <= 5.0)  # exactly the distance band
    bigger, _ = near_edge_mask(edge, valid, 9.0)
    assert (bigger & near).sum() == near.sum() and bigger.sum() > near.sum()  # superset
    assert not near_edge_mask(np.zeros_like(edge), valid, 5.0)[0].any()  # no edge, no band


def test_near_edge_mask_never_marks_invalid_pixels():
    z, n = _two_planes()
    z[:, :20] = 0.0
    edge, valid = _edges(z, n), z > 0
    near, _ = near_edge_mask(edge, valid, 12.0)
    assert not near[~valid].any()


def test_jitter_varies_radius_per_seed_but_is_reproducible():
    z, n = _stairs()
    edge, valid = _edges(z, n), z > 0
    a, _ = near_edge_mask(edge, valid, 6.0, np.random.default_rng(1), 0.6)
    b, _ = near_edge_mask(edge, valid, 6.0, np.random.default_rng(1), 0.6)
    c, _ = near_edge_mask(edge, valid, 6.0, np.random.default_rng(2), 0.6)
    np.testing.assert_array_equal(a, b)
    assert not np.array_equal(a, c)
    assert a[edge].all()  # the band always contains the edge pixels themselves


def test_degradation_drops_exactly_the_near_edge_pixels():
    z, n = _stairs()
    deg = EdgeDropoutDegradation(
        edge_method="range",
        p_drop=1.0,
        radius_px=5.0,
        radius_jitter=0.0,
        max_drop_frac=1.0,
        min_edge_px=0,
        close_px=0,
    )
    s = deg(z, normal=n, seed=0)["sparse_depth"]
    near, _ = near_edge_mask(_edges(z, n), z > 0, 5.0)
    assert (s[near] == 0).all()
    np.testing.assert_array_equal(s[~near], z[~near])


@pytest.mark.parametrize("cap", [0.02, 0.05, 0.15])
def test_dropped_share_never_exceeds_cap(cap):
    z, n = _stairs()
    deg = EdgeDropoutDegradation(p_drop=1.0, radius_px=20.0, max_drop_frac=cap)
    for seed in range(6):
        out = deg(z, normal=n, seed=seed)
        assert 0.0 < out["meta"]["hole_fraction"] <= cap + 1e-9
        assert (out["sparse_depth"] == 0).mean() <= cap + 1e-9


def test_cap_does_not_bind_when_band_is_small():
    z, n = _stairs()
    kw = {"p_drop": 1.0, "radius_px": 3.0, "radius_jitter": 0.0}
    free = EdgeDropoutDegradation(**kw, max_drop_frac=1.0)
    capped = EdgeDropoutDegradation(**kw, max_drop_frac=0.9)
    f = free(z, normal=n, seed=0)
    c = capped(z, normal=n, seed=0)
    np.testing.assert_array_equal(f["sparse_depth"], c["sparse_depth"])
    assert c["meta"]["radius_used"] == 3.0


def test_registered_and_config():
    import yaml

    with open("configs/degradation/edge_dropout.yaml") as f:
        deg = build_degradation(yaml.safe_load(f))
    assert isinstance(deg, EdgeDropoutDegradation)


def test_rejects_bad_config():
    with pytest.raises(ValueError):
        EdgeDropoutDegradation(window=4)
    with pytest.raises(ValueError):
        EdgeDropoutDegradation(edge_method="canny")
    with pytest.raises(ValueError):
        EdgeDropoutDegradation(p_drop=1.5)
    with pytest.raises(ValueError):
        EdgeDropoutDegradation(patch_len_px=(50.0, 10.0))
    with pytest.raises(ValueError):
        EdgeDropoutDegradation(radius_px=-1.0)
    with pytest.raises(ValueError):
        EdgeDropoutDegradation(max_drop_frac=-0.1)
