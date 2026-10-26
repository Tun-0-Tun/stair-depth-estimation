import numpy as np
import pytest
from scipy import ndimage

from data.degradation import BlockDropoutDegradation, build_degradation
from data.degradation.block_dropout import (
    _unit_normals,
    edge_pixel_mask,
    merge_small_patches,
    surface_patches,
)


def _two_planes(h=96, w=128):
    """Left: near plane facing the camera. Right: far plane tilted 60 deg."""
    z = np.empty((h, w), np.float32)
    n = np.zeros((h, w, 3), np.float32)
    half = w // 2
    z[:, :half], n[:, :half, 2] = 1.5, 1.0
    t = np.radians(60)
    z[:, half:], n[:, half:, 1], n[:, half:, 2] = 3.0, np.sin(t), np.cos(t)
    return z, n


def _labels(z, n, cell=1e6):
    valid = z > 0
    return surface_patches(
        z.astype(np.float64),
        _unit_normals(n, z.shape),
        valid,
        15.0,
        0.05,
        cell,
        np.random.default_rng(0),
    )


def test_seeded():
    z, n = _two_planes()
    deg = BlockDropoutDegradation()
    a = deg(z, normal=n, seed=1)["sparse_depth"]
    np.testing.assert_array_equal(a, deg(z, normal=n, seed=1)["sparse_depth"])
    assert any(not np.array_equal(a, deg(z, normal=n, seed=s)["sparse_depth"]) for s in range(2, 8))


def test_output_contract():
    z, n = _two_planes()
    z[:10] = 0.0
    out = BlockDropoutDegradation(p0=0.9)(z, normal=n, seed=0)
    s, lr = out["sparse_depth"], out["lr_depth"]
    assert s.shape == z.shape and s.dtype == np.float32
    np.testing.assert_array_equal(s, lr)
    assert np.isfinite(s).all() and (s >= 0).all()
    assert (s[z == 0] == 0).all()
    kept = s > 0
    np.testing.assert_array_equal(s[kept], z[kept])  # holes only remove, never alter
    assert out["meta"]["calibrated"] is False
    assert out["meta"]["hole_fraction"] == pytest.approx(1 - kept[z > 0].mean())


@pytest.mark.parametrize("sources", [("surface",), ("edge",), ("surface", "edge")])
def test_sources_run(sources):
    z, n = _two_planes()
    out = BlockDropoutDegradation(patch_sources=sources, p0=0.9, p_max=0.9)(z, normal=n, seed=3)
    assert (out["sparse_depth"] == 0).any()


def test_surface_patches_follow_planes():
    z, n = _two_planes()
    lab = _labels(z, n)
    half = z.shape[1] // 2
    assert set(np.unique(lab[:, :half])).isdisjoint(np.unique(lab[:, half:]))
    assert len(np.unique(lab)) == 2  # one patch per plane (huge cell: no grid cutting)


def test_large_surface_is_cut_by_grid():
    z, n = _two_planes()
    assert len(np.unique(_labels(z, n, cell=20))) > 4


def test_edge_detector_fires_on_step_not_on_plane():
    z, n = _two_planes()
    valid = z > 0
    edge = edge_pixel_mask(z.astype(np.float64), _unit_normals(n, z.shape), valid, 15.0, 0.05, 3)
    half = z.shape[1] // 2
    assert edge[:, half - 1 : half + 1].all()
    assert not edge[:, : half - 3].any() and not edge[:, half + 3 :].any()
    flat = np.full_like(z, 2.0)
    n_flat = np.zeros_like(n)
    n_flat[..., 2] = 1.0
    assert not edge_pixel_mask(flat.astype(np.float64), n_flat, flat > 0, 15.0, 0.05, 3).any()


def test_small_patches_merge_into_nearest_large():
    lab = np.zeros((40, 40), np.int64)
    lab[:, :20] = 1  # large
    lab[5:7, 22:24] = 2  # small, 2 px from the large patch
    lab[30:32, 35:37] = 3  # small and far
    out, merged = merge_small_patches(lab, min_area_px=16, merge_dist_px=4.0)
    assert merged == 1
    assert (out[5:7, 22:24] == out[0, 0]).all()
    assert out[30, 35] != out[0, 0] and out[30, 35] > 0
    out, _ = merge_small_patches(lab, min_area_px=16, merge_dist_px=4.0, discard_small=True)
    assert (out[30:32, 35:37] == 0).all()


def test_dropped_region_is_coherent():
    # no roughening: a dropped surface patch never straddles the plane boundary
    z, n = _two_planes()
    deg = BlockDropoutDegradation(
        patch_sources=("surface",),
        cell_px=(1e6, 1e6),
        grow_px=0,
        ragged_px=0.0,
        fade_px=0.0,
        island_frac=0.0,
        p0=0.5,
    )
    half = z.shape[1] // 2
    for seed in range(10):
        hole = deg(z, normal=n, seed=seed)["sparse_depth"] == 0
        assert hole[:, :half].all() or not hole[:, :half].any()
        assert hole[:, half:].all() or not hole[:, half:].any()


def test_registered_and_config():
    import yaml

    with open("configs/degradation/block_dropout.yaml") as f:
        deg = build_degradation(yaml.safe_load(f))
    assert isinstance(deg, BlockDropoutDegradation)


def test_rejects_bad_config():
    with pytest.raises(ValueError):
        BlockDropoutDegradation(patch_sources=("nope",))
    with pytest.raises(ValueError):
        BlockDropoutDegradation(window=4)


def _stairs(h=240, w=320):
    """Floor ramp plus three risers/treads: depth steps and tilted normals."""
    z = np.tile(np.linspace(1.0, 4.0, h, dtype=np.float32)[:, None], (1, w))
    n = np.zeros((h, w, 3), np.float32)
    n[..., 1], n[..., 2] = 0.3, 0.95
    for i, r in enumerate((80, 130, 180)):
        z[r : r + 25] += 0.6 * (i + 1)
        n[r : r + 25] = (0.0, 0.0, 1.0)
    return z, n


def test_holes_are_organic_not_rectangular():
    z, n = _stairs()
    deg = BlockDropoutDegradation(p0=0.6, p_max=0.9)
    seen = 0
    for seed in range(6):
        hole = deg(z, normal=n, seed=seed)["sparse_depth"] == 0
        lab, _ = ndimage.label(hole)
        for i, sl in enumerate(ndimage.find_objects(lab), start=1):
            area = int((lab[sl] == i).sum())
            if area < 400:
                continue
            seen += 1
            bbox = (sl[0].stop - sl[0].start) * (sl[1].stop - sl[1].start)
            assert area / bbox < 0.9  # an axis-aligned block fills ~100% of its bbox
    assert seen > 0


def test_borders_are_speckled():
    z, n = _two_planes(160, 160)
    kw = {
        "patch_sources": ("surface",),
        "cell_px": (1e6, 1e6),
        "p0": 0.9,
        "p_max": 0.9,
        "grow_px": 0,
    }
    soft = BlockDropoutDegradation(**kw, fade_px=4.0, island_frac=0.1)
    hard = BlockDropoutDegradation(**kw, fade_px=0.0, island_frac=0.0)
    s = soft(z, normal=n, seed=0)["sparse_depth"] == 0
    hd = hard(z, normal=n, seed=0)["sparse_depth"] == 0
    n_soft = ndimage.label(s)[1] + ndimage.label(~s)[1]
    n_hard = ndimage.label(hd)[1] + ndimage.label(~hd)[1]
    assert n_soft > n_hard


def test_voronoi_cut_is_seeded_and_warped():
    z, n = _two_planes()
    valid, un = z > 0, _unit_normals(n, z.shape)

    def cut(seed, warp):
        return surface_patches(
            z.astype(np.float64), un, valid, 15.0, 0.05, 20.0, np.random.default_rng(seed), warp
        )

    np.testing.assert_array_equal(cut(1, 10.0), cut(1, 10.0))
    assert not np.array_equal(cut(1, 10.0), cut(2, 10.0))
    assert not np.array_equal(cut(1, 0.0), cut(1, 10.0))
