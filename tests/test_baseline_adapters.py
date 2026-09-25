"""Adapter smoke tests: one sample through each baseline.

What these can and cannot check on a machine with no weights and no CUDA:

* **Always checked** -- the adapter contract (shape, dtype, units, the fact that
  ``predict`` refuses to run before ``load``), the availability report, and the
  registry/config wiring.  These catch the majority of adapter bugs, which are
  plumbing bugs.
* **Checked when the environment allows it** -- a real forward pass through
  DEPTHOR / DuCos.  Those tests skip unless the weights and dependencies are
  present, so the suite stays green on a laptop and turns into a real
  integration test on the GPU box.

Run the full thing on the GPU machine before every release:
``pytest tests/ -v`` there must show the ``ducos``/``depthor`` cases running,
not skipping.
"""

from __future__ import annotations

import numpy as np
import pytest

from baselines.base import BASELINE_REGISTRY, build_baseline
from data.degradation import build_degradation
from data.loaders import build_dataset
from utils.config import CONFIG_ROOT, load_config

#: Baselines that need no weights and no CUDA -- the ones CI can really run.
DEPENDENCY_FREE = ("nn_fill", "bicubic")
#: Baselines wrapping third_party code; run only where they are installed.
UPSTREAM = ("depthor", "ducos", "wave")


@pytest.fixture(scope="module")
def sample():
    ds = build_dataset(
        {
            "loader": "synthetic_stairs",
            "n_samples": 1,
            "height": 96,
            "width": 128,
            "min_depth": 0.3,
            "max_depth": 8.0,
            "strict": False,
        },
        degradation=build_degradation({"type": "active_stereo_astra2"}),
    )
    return ds[0]


# --------------------------------------------------------------------------
# contract
# --------------------------------------------------------------------------


@pytest.mark.parametrize("key", DEPENDENCY_FREE)
def test_adapter_returns_dense_metric_depth(key, sample):
    model = build_baseline(
        {"adapter": key, "device": "cpu", "min_depth": 0.3, "max_depth": 8.0}
    ).load()
    pred = model.predict_sample(sample)

    assert pred.shape == sample["gt_depth"].shape, "must predict at the RGB resolution"
    assert pred.dtype == np.float32
    assert np.isfinite(pred).all(), "a non-finite prediction would silently poison the metrics"
    assert (pred >= 0.3).all() and (pred <= 8.0).all(), "output must be clamped to the depth range"
    assert (pred > 0).all(), "dense means dense: no holes in the output"


@pytest.mark.parametrize("key", DEPENDENCY_FREE)
def test_predict_before_load_is_an_error(key, sample):
    model = build_baseline({"adapter": key, "device": "cpu"})
    with pytest.raises(RuntimeError, match="load"):
        model.predict(sample["rgb"], sample["sparse_depth"])


@pytest.mark.parametrize("key", DEPENDENCY_FREE)
def test_adapter_rejects_uint8_rgb(key, sample):
    """Passing 0-255 RGB is the classic mistake; it must fail, not silently work."""
    model = build_baseline({"adapter": key, "device": "cpu"}).load()
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        model.predict(sample["rgb"] * 255.0, sample["sparse_depth"])


@pytest.mark.parametrize("key", DEPENDENCY_FREE)
def test_adapter_rejects_a_wrong_shape(key, sample):
    model = build_baseline({"adapter": key, "device": "cpu"}).load()
    with pytest.raises(ValueError, match="must be"):
        model.predict(sample["rgb"][..., 0], sample["sparse_depth"])


def test_predict_sample_feeds_the_modality_the_method_wants(sample):
    """A completion model must get the sparse map, an SR model the LR map."""
    completion = build_baseline({"adapter": "nn_fill", "device": "cpu"}).load()
    sr = build_baseline({"adapter": "bicubic", "device": "cpu"}).load()
    assert completion.input_modality == "sparse"
    assert sr.input_modality == "lr"
    assert completion.predict_sample(sample).shape == sample["gt_depth"].shape
    assert sr.predict_sample(sample).shape == sample["gt_depth"].shape


def test_reference_baselines_beat_a_constant_prediction(sample):
    """A floor that is not better than a constant is not a floor."""
    from metrics.depth_metrics import MetricConfig, compute_depth_metrics

    cfg = MetricConfig(min_depth=0.3, max_depth=8.0)
    model = build_baseline(
        {"adapter": "nn_fill", "device": "cpu", "min_depth": 0.3, "max_depth": 8.0}
    ).load()
    pred = model.predict_sample(sample)
    const = np.full_like(pred, float(np.mean(sample["gt_depth"][sample["mask"]])))

    got = compute_depth_metrics(pred, sample["gt_depth"], sample["mask"], cfg)
    baseline = compute_depth_metrics(const, sample["gt_depth"], sample["mask"], cfg)
    assert got["rmse"] < baseline["rmse"]


# --------------------------------------------------------------------------
# wiring
# --------------------------------------------------------------------------


def test_every_model_config_names_a_registered_adapter():
    import baselines  # noqa: F401  (populate the registry)
    import models  # noqa: F401

    for path in sorted((CONFIG_ROOT / "model").glob("*.yaml")):
        cfg = load_config("model", path.stem)
        adapter = cfg.get("adapter", cfg.get("name"))
        assert adapter in BASELINE_REGISTRY, f"{path.name}: unknown adapter {adapter!r}"


def test_availability_reports_are_actionable():
    """An unavailable baseline must say what to install/download, not just fail."""
    for key in UPSTREAM:
        cls = BASELINE_REGISTRY[key]
        avail = cls.availability(None)
        if not avail.ok:
            assert avail.reasons, f"{key}: unavailable with no reason given"
            assert avail.instructions, f"{key}: unavailable with no fix suggested"
            assert key in avail.report(key)


def test_depthor_plus_plus_refuses_to_masquerade_as_v1():
    """The SOW names DEPTHOR++. Its code is unreleased. That must be loud."""
    cls = BASELINE_REGISTRY["depthor_plus_plus"]
    avail = cls.availability(None)
    assert not avail.ok
    assert "not publicly released" in " ".join(avail.reasons)

    model = build_baseline({"adapter": "depthor_plus_plus", "device": "cpu"})
    with pytest.raises(RuntimeError, match="no public implementation"):
        model.load()


def test_ours_refuses_to_run_untrained_by_default():
    model = build_baseline({"adapter": "ours", "device": "cpu"})
    with pytest.raises(RuntimeError, match="no checkpoint"):
        model.load()


def test_ours_runs_end_to_end_with_an_explicit_random_init(sample):
    """Plumbing check for our own model. The numbers are meaningless by design."""
    model = build_baseline(
        {
            "adapter": "ours",
            "device": "cpu",
            "min_depth": 0.3,
            "max_depth": 8.0,
            "allow_random_init": True,
        }
    ).load()
    pred = model.predict_sample(sample)
    assert pred.shape == sample["gt_depth"].shape
    assert np.isfinite(pred).all()


# --------------------------------------------------------------------------
# real upstream inference -- runs only where the environment supports it
# --------------------------------------------------------------------------


@pytest.mark.parametrize("key", UPSTREAM)
def test_upstream_adapter_real_inference(key, sample):
    cls = BASELINE_REGISTRY[key]
    cfg = dict(load_config("model", key))
    avail = cls.availability(cfg.get("checkpoint"), **cfg)
    if not avail.ok:
        pytest.skip(f"{key} not runnable here:\n{avail.report(key)}")

    model = build_baseline(cfg, min_depth=0.3, max_depth=8.0).load(cfg.get("checkpoint"))
    pred = model.predict_sample(sample)

    assert pred.shape == sample["gt_depth"].shape
    assert pred.dtype == np.float32
    assert np.isfinite(pred).all()
    assert float(pred.min()) >= 0.3 and float(pred.max()) <= 8.0
    # a real model must not collapse to a constant
    assert float(pred.std()) > 1e-3
