import numpy as np
import pytest
from scipy import ndimage

from data.degradation import EdgeDropoutDegradation, build_degradation
from data.degradation.edge_dropout import (
    _median_normals,
    _unit_normals,
    close_and_filter_edges,
    near_edge_mask,
    normal_jump_mask,
    plane_edge_mask,
    smooth_noise,
    z_jump_mask,
)

F = 570.0


def _edges(z, n, **kw):
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


def test_seeded():
    z, n = _two_planes()
    deg = EdgeDropoutDegradation()
    a = deg(z, normal=n, seed=1)["sparse_depth"]
    np.testing.assert_array_equal(a, deg(z, normal=n, seed=1)["sparse_depth"])
    assert any(not np.array_equal(a, deg(z, normal=n, seed=s)["sparse_depth"]) for s in range(2, 8))


def test_output_contract():
    z, n = _two_planes()
    z[:10] = 0.0
    out = EdgeDropoutDegradation(p_drop_max=1.0)(z, normal=n, seed=0)
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
    assert not edge[:, : half - 6].any() and not edge[:, half + 6 :].any()
    flat = np.full_like(z, 2.0)
    n_flat = np.zeros_like(n)
    n_flat[..., 2] = 1.0
    assert not _edges(flat, n_flat).any()


def test_plane_classifier_ignores_a_steep_slope():
    z, n = _tilted_plane(75)
    assert np.ptp(z) / z.mean() > 0.1  # the depth changes a lot along the slope ...
    assert not _edges(z, n).any()  # ... but a plane is a plane, whatever its tilt


def test_plane_classifier_marks_both_sides_of_a_depth_step():
    z = np.full((60, 80), 1.5, np.float32)
    z[:, 40:] = 3.0
    n = np.zeros((60, 80, 3), np.float32)
    n[..., 2] = 1.0
    edge = _edges(z, n, z_step=2)
    assert edge[:, 38:42].all()  # two pixels each side (z_step)
    assert not edge[:, :37].any() and not edge[:, 43:].any()


def test_far_side_only_marks_only_the_far_side_of_a_step():
    z = np.full((60, 80), 1.5, np.float32)
    z[:, 40:] = 3.0
    n = np.zeros((60, 80, 3), np.float32)
    n[..., 2] = 1.0
    edge = _edges(z, n, z_step=2, far_side_only=True)
    assert edge[:, 40:42].all()  # the far (right) side
    assert not edge[:, :40].any() and not edge[:, 43:].any()


def test_plane_classifier_is_robust_to_depth_noise():
    z, n = _tilted_plane(60, d=2.0)
    noisy = z + np.random.default_rng(0).normal(0.0, 0.001, z.shape).astype(np.float32)
    assert not _edges(noisy, n).any()


def test_plane_classifier_finds_creases_through_the_normals():
    z = np.full((60, 80), 2.0, np.float32)
    n = np.zeros((60, 80, 3), np.float32)
    n[:, :40, 2] = 1.0
    t = np.radians(60)
    n[:, 40:, 1], n[:, 40:, 2] = np.sin(t), np.cos(t)
    edge = _edges(z, n)  # no depth step at all
    assert edge[:, 36:44].all()  # within normal_step (4) of the crease, both sides
    assert not edge[:, :35].any() and not edge[:, 44:].any()


def _jump_masks(z, n, **kw):
    """The two tests on their own; ``n`` are raw normals, unit-normalised and unsmoothed."""
    h, w = z.shape
    un = _unit_normals(n, z.shape)
    zj = z_jump_mask(
        z.astype(np.float64), un, z > 0, fx=F, fy=F, cx=(w - 1) / 2, cy=(h - 1) / 2, **kw
    )
    return zj, normal_jump_mask(un)


def test_z_jump_needs_a_depth_step_and_normal_jump_needs_a_turning_normal():
    z = np.full((60, 80), 1.5, np.float32)
    z[:, 40:] = 3.0  # depth step, the normals do not change
    n = np.zeros((60, 80, 3), np.float32)
    n[..., 2] = 1.0
    zj, nj = _jump_masks(z, n, z_step=2)
    assert zj[:, 38:42].all() and not nj.any()

    z = np.full((60, 80), 2.0, np.float32)  # crease: the normal turns, the depth does not
    n = np.zeros((60, 80, 3), np.float32)
    n[:, :40, 2] = 1.0
    n[:, 40:, 1], n[:, 40:, 2] = np.sin(np.radians(60)), np.cos(np.radians(60))
    zj, nj = _jump_masks(z, n)
    assert not zj.any() and nj[:, 36:44].all()
    assert not nj[:, :35].any() and not nj[:, 44:].any()


def test_plane_edge_mask_is_the_union_of_the_two_jump_masks():
    z, n = _two_planes()
    valid = z > 0
    h, w = z.shape
    un = _median_normals(_unit_normals(n, z.shape), 5)
    zj = z_jump_mask(z.astype(np.float64), un, valid, fx=F, fy=F, cx=(w - 1) / 2, cy=(h - 1) / 2)
    nj = normal_jump_mask(un)
    np.testing.assert_array_equal(_edges(z, n), valid & (zj | nj))
    assert zj.any() and nj.any()


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
    old = _edges(z, n, normal_thresh=10.0, normal_step=1, normal_median=3, z_step=2)
    new = _edges(z, n, normal_thresh=10.0)  # larger defaults: step 4, normals 4 px apart, 5x5
    assert old.mean() > 0.01  # the old support flags a visible share of a flat wall
    assert new.mean() < old.mean() / 3  # larger support alone removes most of them
    assert (
        close_and_filter_edges(new, z > 0, 40, 3)[0].mean() < 0.001
    )  # and the speck filter the rest


def test_speck_filter_removes_isolated_pixels_but_keeps_long_edges():
    z = np.full((80, 120), 2.0, np.float32)
    z[:, 60:] = 3.0  # a long depth step
    z[20, 20] += 0.5  # one outlier pixel far from it
    n = np.zeros((80, 120, 3), np.float32)
    n[..., 2] = 1.0
    raw = _edges(z, n)
    assert raw[10:30, 10:30].any()  # the outlier is flagged ...
    clean, n_patches = close_and_filter_edges(raw, z > 0, 40, 3)
    assert not clean[:, :50].any()  # ... and removed
    assert clean[:, 56:64].all()  # the 80-px step stays
    assert n_patches == 1
    off, n_off = close_and_filter_edges(raw, z > 0, 0, 1)
    assert off.sum() == raw.sum() and n_off >= 2  # no-op when disabled: every group stays


def test_closing_fills_small_gaps_and_the_filter_counts_the_filled_pixels():
    valid = np.ones((30, 100), bool)
    edge = np.zeros((30, 100), bool)
    edge[15, 10:30] = True  # two 20-px pieces ...
    edge[15, 32:52] = True  # ... with a 2-px gap: alone each is below min_patch_size = 30
    clean, n_patches = close_and_filter_edges(edge, valid, 30, 3)
    assert n_patches == 1 and clean[15, 10:52].all() and clean.sum() == 42  # gap filled and kept
    wide = edge.copy()
    wide[15, 32:52] = False
    wide[15, 36:56] = True  # a 6-px gap is wider than closing_size - 1: two groups, both removed
    gone, n_gone = close_and_filter_edges(wide, valid, 30, 3)
    assert not gone.any() and n_gone == 0
    filled, _ = close_and_filter_edges(edge, valid, 0, 3)
    assert (filled & ~edge).sum() == 2 and not (edge & ~filled).any()  # closing only adds
    invalid = valid.copy()
    invalid[15, 30:32] = False  # a filled pixel must not land on invalid depth
    assert not close_and_filter_edges(edge, invalid, 0, 3)[0][15, 30:32].any()


def test_closing_keeps_edges_at_the_image_border():
    valid = np.ones((20, 60), bool)
    edge = np.zeros((20, 60), bool)
    edge[0, :50] = True  # runs along the top border
    clean, n_patches = close_and_filter_edges(edge, valid, 40, 5)
    assert n_patches == 1 and clean[0, :50].all()


def test_degradation_reports_the_speck_filter():
    z, n = _noisy_flat(8.0)
    meta = EdgeDropoutDegradation(normal_thresh=10.0)(z, normal=n, seed=0)["meta"]
    assert meta["edge_fraction"] < 0.001 < meta["edge_fraction_raw"]
    assert meta["n_patches"] >= 0


def test_every_edge_gets_a_band():
    z = np.full((160, 300), 2.0, np.float32)
    z[:, 150:] = 3.0  # step 1
    z[:, 250:] = 2.0  # step 2, 100 px away
    n = np.zeros((160, 300, 3), np.float32)
    n[..., 2] = 1.0
    out = EdgeDropoutDegradation(radius_jitter=0.0, p_drop_max=1.0)(z, normal=n, seed=0)
    hole = out["sparse_depth"] == 0
    assert hole[:, 144:156].all() and hole[:, 244:256].all()  # both steps lose pixels
    assert not hole[:, :100].any() and not hole[:, 180:220].any()  # nothing elsewhere
    assert out["meta"]["n_patches"] == 2


def test_smooth_noise_has_unit_std_and_is_reproducible():
    a = smooth_noise((200, 300), np.random.default_rng(3), 16.0)
    b = smooth_noise((200, 300), np.random.default_rng(3), 16.0)
    assert a.shape == (200, 300)
    np.testing.assert_array_equal(a, b)
    assert abs(a.mean()) < 1e-6 and a.std() == pytest.approx(1.0, abs=1e-6)
    # smooth: neighbouring pixels are far more alike than the field is varied
    assert np.abs(np.diff(a, axis=1)).mean() < 0.2 * a.std()
    assert smooth_noise((7, 5), np.random.default_rng(0), 16.0).shape == (7, 5)  # tiny image


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
        radius=5.0,
        radius_jitter=0.0,
        p_drop_max=1.0,
        min_patch_size=0,
        closing_size=1,
    )
    s = deg(z, normal=n, seed=0)["sparse_depth"]
    near, _ = near_edge_mask(_edges(z, n), z > 0, 5.0)
    assert (s[near] == 0).all()
    np.testing.assert_array_equal(s[~near], z[~near])


@pytest.mark.parametrize("cap", [0.02, 0.05, 0.15])
def test_dropped_share_never_exceeds_cap(cap):
    z, n = _stairs()
    deg = EdgeDropoutDegradation(radius=20.0, p_drop_max=cap)
    for seed in range(6):
        out = deg(z, normal=n, seed=seed)
        assert 0.0 < out["meta"]["hole_fraction"] <= cap + 1e-9
        assert (out["sparse_depth"] == 0).mean() <= cap + 1e-9


def test_cap_does_not_bind_when_band_is_small():
    z, n = _stairs()
    kw = {"radius": 3.0, "radius_jitter": 0.0}
    free = EdgeDropoutDegradation(**kw, p_drop_max=1.0)
    capped = EdgeDropoutDegradation(**kw, p_drop_max=0.9)
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
        EdgeDropoutDegradation(z_step=0)
    for bad in (0, -1, 2, 4):  # the closing square must have an odd side
        with pytest.raises(ValueError):
            EdgeDropoutDegradation(closing_size=bad)
    with pytest.raises(ValueError):
        EdgeDropoutDegradation(min_patch_size=-1)
    with pytest.raises(ValueError):
        EdgeDropoutDegradation(radius=-1.0)
    with pytest.raises(ValueError):
        EdgeDropoutDegradation(p_drop_max=-0.1)
