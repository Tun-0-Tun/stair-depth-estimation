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


def test_model_depth_range_is_independent_of_the_eval_window():
    """A checkpoint's depth range must not follow the dataset being scored.

    DEPTHOR decodes depth as a weighted sum of bin centres spanning
    [min_depth, max_depth]. Letting a narrow evaluation window re-space those
    bins does not raise anything -- it just returns wrong depths.
    """
    from baselines.base import model_depth_range
    from utils.config import load_config

    # stated by the model config -> used verbatim, whatever the eval window is
    depthor = dict(load_config("model", "depthor"))
    assert depthor["min_depth"] == 0.001, "depthor.yaml must pin the checkpoint's bin grid"
    assert model_depth_range(depthor, 0.2, 8.0) == (0.001, 10.0)
    assert model_depth_range(depthor, 0.0, 1.01) == (0.001, 10.0)

    # not stated -> fall back to the eval window, as every other baseline does
    assert model_depth_range({"adapter": "bicubic"}, 0.2, 8.0) == (0.2, 8.0)


def test_depthor_minjiang_experiment_uses_the_centre_footprint():
    """dtof_l5 hands DEPTHOR ~1700 px per zone instead of 1 and it over-predicts 2-4x."""
    from utils.config import load_experiment

    cfg = load_experiment("depthor_minjiang")
    assert cfg["degradation"]["type"] == "dtof_sim"
    assert cfg["degradation"]["fill_ratio"] == 0.0, "must be the one-pixel-per-zone variant"


# --------------------------------------------------------------------------
# local_bilateral: the two pieces that fail silently if they are wrong
# --------------------------------------------------------------------------


def test_local_affine_fit_recovers_exact_coefficients_per_region():
    """The per-superpixel fit is a one-pass bincount rewrite of a Python loop.

    If it is wrong the output is still a plausible depth map, so pin it against
    data built to have an exact answer: two regions, two different (s, t).
    """
    import numpy as np

    from baselines.local_bilateral import _affine_per_label

    h = w = 16
    labels = np.zeros((h, w), dtype=np.int64)
    labels[:, w // 2 :] = 1
    rel = np.linspace(0.1, 2.0, h * w).reshape(h, w)

    truth = {0: (3.0, 0.5), 1: (-1.5, 4.0)}
    depth = np.where(labels == 0, truth[0][0] * rel + truth[0][1], truth[1][0] * rel + truth[1][1])

    valid = np.zeros((h, w), dtype=bool)
    valid[::2, ::2] = True  # a sparse subset, as a real sensor would give

    s, t = _affine_per_label(rel, depth, labels, valid, 2, min_pixels=4, fallback=(0.0, 0.0))
    for k, (s_k, t_k) in truth.items():
        assert s[k] == pytest.approx(s_k, rel=1e-6)
        assert t[k] == pytest.approx(t_k, rel=1e-6)

    # A region with too few points must take the global fallback, not a wild fit.
    lonely = np.zeros((h, w), dtype=bool)
    lonely[0, 0] = True
    s2, t2 = _affine_per_label(rel, depth, labels, lonely, 2, min_pixels=4, fallback=(9.0, 8.0))
    assert (s2 == 9.0).all() and (t2 == 8.0).all()


def test_local_bilateral_smoothing_keeps_a_step_and_leaves_flat_fields_alone():
    """Edge-preserving means a sharp change in (s, t) survives the smoother."""
    import numpy as np

    from baselines.local_bilateral import _smooth_bilateral

    flat_s = np.full((32, 32), 2.0)
    flat_t = np.full((32, 32), 0.5)
    out_s, out_t = _smooth_bilateral(flat_s, flat_t, 6.0, 0.25, 10)
    assert np.allclose(out_s, 2.0) and np.allclose(out_t, 0.5)

    step_s = np.where(np.arange(32)[None, :] < 16, 1.0, 5.0) * np.ones((32, 1))
    step_t = np.zeros((32, 32))
    sm_s, _ = _smooth_bilateral(step_s, step_t, 6.0, 0.25, 10)
    # Far from the seam the values are untouched; a plain Gaussian would bleed.
    assert sm_s[16, 2] == pytest.approx(1.0, abs=1e-3)
    assert sm_s[16, 29] == pytest.approx(5.0, abs=1e-3)
    assert sm_s.min() >= 1.0 - 1e-6 and sm_s.max() <= 5.0 + 1e-6
