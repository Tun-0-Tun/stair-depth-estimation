"""SequentialDegradation: presets applied one after another."""

from __future__ import annotations

import numpy as np
import pytest

from data.degradation import SequentialDegradation, build_degradation
from utils.config import load_config


def _gt() -> np.ndarray:
    return np.linspace(0.8, 5.0, 64 * 64, dtype=np.float32).reshape(64, 64)


def _normals(shape=(64, 64)) -> np.ndarray:
    n = np.zeros((*shape, 3), dtype=np.float32)
    n[..., 2] = 1.0  # fronto-parallel: a valid unit normal everywhere
    return n


class _CountingEstimator:
    """Stands in for a normal estimator; counts how often it is called."""

    def __init__(self):
        self.calls = 0

    def __call__(self, gt):
        self.calls += 1
        return {"normals": _normals(gt.shape)}


def test_single_step_matches_the_preset_applied_directly():
    # projector_shadow is deterministic, so the seed cannot matter
    gt = _gt()
    direct = build_degradation(dict(load_config("degradation", "projector_shadow")))
    seq = SequentialDegradation(steps=["projector_shadow"])

    assert np.array_equal(seq(gt, seed=3)["sparse_depth"], direct(gt, seed=3)["sparse_depth"])


def test_same_seed_same_output_different_seed_different_output():
    gt = _gt()
    seq = SequentialDegradation(steps=["lateral_noise", "projector_shadow"])
    n = _normals()

    a = seq(gt, normal=n, seed=7)["sparse_depth"]
    b = seq(gt, normal=n, seed=7)["sparse_depth"]
    c = seq(gt, normal=n, seed=8)["sparse_depth"]

    assert np.array_equal(a, b)
    assert not np.array_equal(a, c)


def test_steps_are_applied_in_order():
    # projector_shadow only removes depth (shadows); reversing the order changes the result
    gt = _gt()
    ab = SequentialDegradation(steps=["projector_shadow", "bicubic_x4"])(gt, seed=0)
    ba = SequentialDegradation(steps=["bicubic_x4", "projector_shadow"])(gt, seed=0)

    assert ab["meta"]["step0_projector_shadow"]["degradation"] == "ProjectorShadowDegradation"
    assert ba["meta"]["step0_bicubic_x4"]["degradation"] == "BicubicSRDegradation"
    assert not np.array_equal(ab["sparse_depth"], ba["sparse_depth"])


def test_unknown_preset_lists_the_available_ones():
    with pytest.raises(KeyError, match="available"):
        SequentialDegradation(steps=["no_such_preset"])


@pytest.mark.parametrize("steps", [[], None, "lateral_noise", [3]])
def test_bad_step_list_is_rejected(steps):
    with pytest.raises(ValueError):
        SequentialDegradation(steps=steps)


def test_noop_preset_is_rejected():
    with pytest.raises(ValueError, match="no-op"):
        SequentialDegradation(steps=["none"])


def test_nested_sequence_is_rejected():
    with pytest.raises(ValueError, match="nesting"):
        SequentialDegradation(steps=["sequence_lateral_projector_axial"])


def test_uses_normals_follows_the_steps():
    assert SequentialDegradation(steps=["projector_shadow"]).uses_normals is False
    assert SequentialDegradation(steps=["axial_noise"]).uses_normals is True
    assert SequentialDegradation(steps=["projector_shadow", "lateral_noise"]).uses_normals is True


def test_normals_are_estimated_once_and_shared():
    est = _CountingEstimator()
    seq = SequentialDegradation(steps=["lateral_noise", "axial_noise"])

    out = seq(_gt(), seed=0, normal_estimator=est)

    assert est.calls == 1, "the composite must resolve normals once, not once per step"
    assert out["meta"]["normal_source"].startswith("estimated:")


def test_provided_normals_skip_the_estimator():
    est = _CountingEstimator()
    seq = SequentialDegradation(steps=["axial_noise"])

    out = seq(_gt(), normal=_normals(), seed=0, normal_estimator=est)

    assert est.calls == 0
    assert out["meta"]["normal_source"] == "provided"


def test_meta_namespaces_each_step_and_reports_calibration():
    seq = SequentialDegradation(steps=["projector_shadow", "axial_noise"])
    out = seq(_gt(), normal=_normals(), seed=0)

    assert out["meta"]["steps"] == ["projector_shadow", "axial_noise"]
    assert "step0_projector_shadow" in out["meta"]
    assert "step1_axial_noise" in out["meta"]
    assert out["meta"]["calibrated"] is False  # both presets are uncalibrated


def test_output_is_finite_non_negative_and_gt_shaped():
    gt = _gt()
    out = SequentialDegradation(steps=["lateral_noise", "projector_shadow", "axial_noise"])(
        gt, normal=_normals(), seed=1
    )
    sparse = out["sparse_depth"]

    assert sparse.shape == gt.shape and sparse.dtype == np.float32
    assert np.isfinite(sparse).all()
    assert (sparse >= 0).all()


def test_builds_from_yaml_type():
    deg = build_degradation({"type": "sequential", "steps": ["projector_shadow", "bicubic_x4"]})
    assert isinstance(deg, SequentialDegradation)
    assert deg.steps == ["projector_shadow", "bicubic_x4"]
