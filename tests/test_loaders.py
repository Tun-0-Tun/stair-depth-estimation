"""Smoke tests: every loader must honour the sample contract.

These are cheap and boring on purpose.  They catch the class of bug that is
otherwise invisible -- a loader that returns millimetres instead of metres, or
uint8 RGB, or a sparse mask that disagrees with the sparse map.  Such a bug does
not crash; it just makes one column of the final table wrong.

Loaders whose data is not on this machine are skipped, not failed: the suite has
to be green on a laptop.  ``synthetic_stairs`` needs nothing and therefore
always runs, which is why it exists.
"""

from __future__ import annotations

import numpy as np
import pytest

from data.degradation import build_degradation
from data.loaders import DATASET_REGISTRY, build_dataset, validate_sample
from utils.config import CONFIG_ROOT, load_config


def _try_build(key: str, **overrides):
    """Build a loader from its committed config, or skip if the data is absent."""
    cfg = dict(load_config("dataset", key))
    # Datasets that ship no degraded input (NYUv2, MinJiang, synthetic) need a
    # sensor model to produce one; the ones that do ship one ignore it.
    degradation = build_degradation(dict(load_config("degradation", "astra2_uncalibrated")))
    try:
        return build_dataset(cfg, degradation=degradation, **overrides)
    except (FileNotFoundError, RuntimeError) as exc:
        pytest.skip(f"{key}: data not available here ({str(exc).splitlines()[0]})")


# --------------------------------------------------------------------------
# the synthetic dataset always runs -- it is the contract's reference example
# --------------------------------------------------------------------------


@pytest.fixture(scope="module")
def synth():
    deg = build_degradation({"type": "active_stereo_astra2"})
    return build_dataset(
        {
            "loader": "synthetic_stairs",
            "n_samples": 3,
            "height": 96,
            "width": 128,
            "min_depth": 0.3,
            "max_depth": 8.0,
            "strict": False,
        },
        degradation=deg,
    )


def test_synthetic_sample_matches_the_contract(synth):
    validate_sample(synth[0], "synthetic_stairs[0]")


def test_len_and_indexing_agree(synth):
    assert len(synth) == 3
    ids = {synth[i]["meta"]["sample_id"] for i in range(len(synth))}
    assert len(ids) == 3


def test_loader_is_deterministic(synth):
    """Same index, same bytes -- otherwise two teammates cannot compare numbers."""
    a, b = synth[1], synth[1]
    for key in ("rgb", "gt_depth", "sparse_depth", "lr_depth"):
        assert np.array_equal(a[key], b[key]), f"{key} changed between two reads"


def test_degradation_is_seeded_per_sample(synth):
    """Different samples get different noise, but each one is reproducible."""
    s0, s1 = synth[0], synth[1]
    assert not np.array_equal(s0["sparse_depth"], s1["sparse_depth"])


def test_depth_is_in_metres_not_millimetres(synth):
    """A loader that forgot to divide by 1000 fails here, not three weeks later."""
    gt = synth[0]["gt_depth"]
    valid = gt[gt > 0]
    assert valid.max() < 100.0, "depth looks like millimetres; divide by depth_unit_per_metre"


def test_sparse_input_actually_has_holes(synth):
    """The Astra 2 model must invalidate *something*, or it is not modelling anything."""
    s = synth[0]
    assert 0.0 < s["meta"]["sparsity"] < 1.0
    assert s["sparse_mask"].sum() > 0


def test_both_input_forms_are_always_present(synth):
    """An adapter must never have to ask which dataset it is looking at."""
    s = synth[0]
    assert s["sparse_depth"].shape == s["gt_depth"].shape
    assert s["lr_depth"].ndim == 2 and s["lr_depth"].size > 0


def test_target_size_resizes_rgb_and_gt_together():
    deg = build_degradation({"type": "identity"})
    ds = build_dataset(
        {
            "loader": "synthetic_stairs",
            "n_samples": 1,
            "height": 96,
            "width": 128,
            "target_size": [48, 64],
            "strict": False,
        },
        degradation=deg,
    )
    s = ds[0]
    assert s["rgb"].shape[:2] == (48, 64)
    assert s["gt_depth"].shape == (48, 64)
    validate_sample(s, "resized")


def test_identity_degradation_round_trips_exactly():
    """Sanity anchor: input == GT must give a perfect score through the real code."""
    from metrics.depth_metrics import MetricConfig, compute_depth_metrics

    ds = build_dataset(
        {
            "loader": "synthetic_stairs",
            "n_samples": 1,
            "height": 64,
            "width": 64,
            "min_depth": 0.3,
            "max_depth": 8.0,
            "strict": False,
        },
        degradation=build_degradation({"type": "identity"}),
    )
    s = ds[0]
    m = compute_depth_metrics(
        s["sparse_depth"], s["gt_depth"], s["mask"], MetricConfig(min_depth=0.3, max_depth=8.0)
    )
    assert m["rmse"] == pytest.approx(0.0, abs=1e-6)
    assert m["delta1"] == 1.0


# --------------------------------------------------------------------------
# real datasets: run if present, skip if not
# --------------------------------------------------------------------------


@pytest.mark.parametrize("key", sorted(set(DATASET_REGISTRY) - {"synthetic_stairs"}))
def test_real_loader_contract(key):
    """Runs where the shared data folder has the dataset; skips where it does not."""
    ds = _try_build(key, max_samples=2, strict=True)
    assert len(ds) > 0, f"{key}: built but empty"
    validate_sample(ds[0], f"{key}[0]")


def test_every_registered_dataset_has_a_config():
    """A dataset you cannot select from a config does not exist for the benchmark."""
    configs = {p.stem for p in (CONFIG_ROOT / "dataset").glob("*.yaml")}
    missing = set(DATASET_REGISTRY) - configs
    assert not missing, f"no configs/dataset/*.yaml for: {sorted(missing)}"


# --------------------------------------------------------------------------
# degradation models
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "cfg",
    [
        {"type": "active_stereo_astra2"},
        {"type": "dtof_sim", "zones_h": 8, "zones_w": 8},
        {"type": "bicubic_sr", "scale": 4},
        {"type": "random_sparse", "n_samples": 100},
        {"type": "identity"},
    ],
    ids=lambda c: c["type"],
)
def test_degradation_contract(cfg):
    gt = np.linspace(0.8, 5.0, 64 * 64, dtype=np.float32).reshape(64, 64)
    deg = build_degradation(dict(cfg))

    out_a = deg(gt, seed=7)
    out_b = deg(gt, seed=7)
    sparse = out_a["sparse_depth"]

    assert sparse.shape == gt.shape and sparse.dtype == np.float32
    assert np.isfinite(sparse).all()
    assert (sparse >= 0).all(), "0 means 'no measurement'; negatives are meaningless"
    assert np.array_equal(sparse, out_b["sparse_depth"]), "same seed must give the same input"
    assert "degradation" in out_a["meta"]

    # `identity` and a noise-free `bicubic_sr` are deterministic by construction:
    # they never draw from the rng, so the seed cannot change their output.
    if cfg["type"] not in ("identity", "bicubic_sr"):
        assert not np.array_equal(sparse, deg(gt, seed=8)["sparse_depth"])


def test_astra2_degradation_produces_structured_not_random_holes():
    """Occlusion shadows sit next to depth edges; salt-and-pepper does not.

    The distinguishing property: invalid pixels must be spatially clustered, so
    the mean size of a connected invalid region is well above one pixel.
    """
    from scipy import ndimage

    gt = np.full((96, 128), 3.0, dtype=np.float32)
    gt[:, 64:] = 1.2  # one big vertical depth discontinuity
    deg = build_degradation({"type": "active_stereo_astra2"})
    sparse = deg(gt, seed=0)["sparse_depth"]

    holes = sparse == 0
    assert holes.any(), "an active-stereo model with no invalid pixels models nothing"
    _labels, n = ndimage.label(holes)
    assert n > 0
    assert holes.sum() / n > 3.0, "holes look uncorrelated; the occlusion model is not firing"


def test_dtof_sim_writes_at_most_one_value_per_zone():
    gt = np.linspace(0.5, 3.5, 96 * 96, dtype=np.float32).reshape(96, 96)
    deg = build_degradation({"type": "dtof_sim", "zones_h": 8, "zones_w": 8})
    out = deg(gt, seed=3)
    assert out["meta"]["n_zones_reported"] <= 64
    assert 0 < (out["sparse_depth"] > 0).mean() < 1.0
