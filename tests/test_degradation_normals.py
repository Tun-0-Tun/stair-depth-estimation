"""BaseDegradation resolves normals: dataset's, else estimated, ``force_normals`` overrides."""

import numpy as np
import pytest

from data.degradation import AxialDegradation, IdentityDegradation, LateralDegradation
from data.norm import build_normal_map


def _tilted_plane(deg, h=100, w=100, z0=2.0, fy=570.0):
    """Plane whose normal is ``deg`` degrees from the optical axis (tilted about x)."""
    s = np.tan(np.radians(deg))
    v = np.arange(h, dtype=np.float64)[:, None] - (h - 1) / 2
    return np.broadcast_to(z0 / (1.0 - s * v / fy), (h, w)).astype(np.float32)


def _normal(h, w, deg):
    t = np.radians(deg)
    n = np.zeros((h, w, 3), np.float32)
    n[..., 1], n[..., 2] = np.sin(t), np.cos(t)
    return n


@pytest.fixture
def estimator():
    return build_normal_map({"type": "cross"})


def test_provided_normal_is_used_and_returned(estimator):
    gt, n = _tilted_plane(40), _normal(100, 100, 10)
    out = AxialDegradation()(gt, seed=0, normal=n, normal_estimator=estimator)
    assert out["meta"]["normal_source"] == "provided"
    np.testing.assert_array_equal(out["normals"], n)


def test_missing_normal_is_estimated_from_gt(estimator):
    gt = _tilted_plane(40)
    out = AxialDegradation()(gt, seed=0, normal_estimator=estimator)
    assert out["meta"]["normal_source"] == "estimated:CrossProductNormals"
    assert out["normals"].shape == (100, 100, 3)
    # estimated normal at the centre is ~40 deg from the optical axis
    nz = np.abs(out["normals"][50, 50, 2])
    assert np.degrees(np.arccos(nz)) == pytest.approx(40, abs=1.5)
    # and the noise is as large as with the exact normal
    exact = AxialDegradation()(gt, seed=0, normal=_normal(100, 100, 40))["sparse_depth"] - gt
    est = out["sparse_depth"] - gt
    assert est.std() == pytest.approx(exact.std(), rel=0.1)


def test_force_normals_ignores_the_provided_map(estimator):
    gt, n = _tilted_plane(40), _normal(100, 100, 10)
    out = AxialDegradation()(gt, seed=0, normal=n, normal_estimator=estimator, force_normals=True)
    assert out["meta"]["normal_source"].startswith("estimated")
    assert not np.array_equal(out["normals"], n)


def test_no_normal_and_no_estimator_raises():
    gt = _tilted_plane(40)
    with pytest.raises(ValueError, match="--normals"):
        AxialDegradation()(gt, seed=0)
    with pytest.raises(ValueError, match="force-normals"):
        AxialDegradation()(gt, seed=0, normal=_normal(100, 100, 10), force_normals=True)


def test_estimator_intrinsics_come_from_the_normals_config():
    gt = _tilted_plane(40)
    right = build_normal_map({"type": "cross", "fx": 570.0})
    wrong = build_normal_map({"type": "cross", "fx": 300.0})
    a = AxialDegradation()(gt, seed=0, normal_estimator=right)["normals"][50, 50, 2]
    b = AxialDegradation()(gt, seed=0, normal_estimator=wrong)["normals"][50, 50, 2]
    assert abs(np.degrees(np.arccos(abs(a))) - np.degrees(np.arccos(abs(b)))) > 5


def test_lateral_uses_the_same_resolution(estimator):
    gt = _tilted_plane(40)
    out = LateralDegradation()(gt, seed=0, normal_estimator=estimator)
    assert out["meta"]["normal_source"].startswith("estimated")
    assert "normals" in out


def test_degradations_without_normals_ignore_the_arguments(estimator):
    gt = np.full((8, 8), 1.0, np.float32)
    out = IdentityDegradation()(gt, normal=_normal(8, 8, 0), normal_estimator=estimator)
    np.testing.assert_array_equal(out["sparse_depth"], gt)
    assert "normals" not in out and "normal_source" not in out["meta"]


def test_loader_adds_normals_to_the_sample(estimator):
    from data.loaders import build_dataset
    from utils.config import load_config

    ds = build_dataset(
        dict(load_config("dataset", "synthetic_stairs")),
        degradation=AxialDegradation(),
        force_degradation=True,
        normals=estimator,
        max_samples=1,
    )
    s = ds[0]
    assert s["normals"].shape == (*s["gt_depth"].shape, 3)
    assert s["meta"]["normal_source"] == "estimated:CrossProductNormals"
