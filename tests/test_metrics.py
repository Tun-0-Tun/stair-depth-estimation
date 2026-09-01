"""Metric tests against answers computed by hand.

These are the most important tests in the repository.  Every number in the
final report goes through ``compute_depth_metrics``; if it is wrong, everything
downstream is wrong and nothing else would catch it.  So the expected values
here are derived analytically in the comments, not copied from a run.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from metrics.depth_metrics import (
    METRIC_NAMES,
    DepthMetricAccumulator,
    MetricConfig,
    compute_depth_metrics,
)

# A 2x2 frame with depths chosen so the arithmetic is exact and easy to check.
GT = np.array([[1.0, 2.0], [4.0, 8.0]], dtype=np.float32)
CFG = MetricConfig(min_depth=0.1, max_depth=100.0)


def test_perfect_prediction_is_exactly_zero_error():
    m = compute_depth_metrics(GT, GT, config=CFG)
    assert m["absrel"] == 0.0
    assert m["rmse"] == 0.0
    assert m["rmse_log"] == 0.0
    assert m["mae"] == 0.0
    assert m["sqrel"] == 0.0
    assert m["silog"] == 0.0
    assert m["delta1"] == m["delta2"] == m["delta3"] == 1.0
    assert m["n_valid"] == 4


def test_uniform_ten_percent_overestimate():
    """pred = 1.1 * gt.

    absrel  = mean(|1.1g - g| / g)            = 0.1 exactly
    mae     = mean(0.1 * g)                   = 0.1 * 3.75      = 0.375
    rmse    = sqrt(mean((0.1 g)^2))
              mean(g^2) = (1 + 4 + 16 + 64)/4 = 21.25
                                              = 0.1 * sqrt(21.25) = 0.46098...
    sqrel   = mean((0.1 g)^2 / g) = 0.01 * mean(g) = 0.01 * 3.75 = 0.0375
    rmselog = |log(1.1)|                      = 0.0953101798...
    silog   = 0 (the error is a constant factor, i.e. scale-invariant)
    delta   = max(1.1, 1/1.1) = 1.1 < 1.25    -> all three are 1.0
    """
    pred = GT * 1.1
    m = compute_depth_metrics(pred, GT, config=CFG)
    assert m["absrel"] == pytest.approx(0.1, abs=1e-6)
    assert m["mae"] == pytest.approx(0.375, abs=1e-6)
    assert m["rmse"] == pytest.approx(0.1 * math.sqrt(21.25), abs=1e-6)
    assert m["sqrel"] == pytest.approx(0.0375, abs=1e-6)
    assert m["rmse_log"] == pytest.approx(abs(math.log(1.1)), abs=1e-6)
    assert m["silog"] == pytest.approx(0.0, abs=1e-4)
    assert m["delta1"] == 1.0


def test_delta_threshold_is_strict_and_counts_the_right_pixels():
    """Ratios 1.0, 1.2, 1.25, 2.0 against thresholds 1.25 / 1.5625 / 1.953125.

    delta uses a strict ``ratio < threshold``, so a pixel sitting exactly on the
    threshold does NOT count.  That matches the reference implementations;
    getting it wrong shifts delta1 by 1/N, which is invisible on a large dataset
    and just as wrong.
    """
    gt = np.array([[1.0, 1.0], [1.0, 1.0]], dtype=np.float32)
    pred = np.array([[1.0, 1.2], [1.25, 2.0]], dtype=np.float32)
    m = compute_depth_metrics(pred, gt, config=CFG)
    assert m["delta1"] == pytest.approx(0.5)  # 1.0, 1.2          (1.25 is excluded)
    assert m["delta2"] == pytest.approx(0.75)  # + 1.25 < 1.5625
    assert m["delta3"] == pytest.approx(0.75)  # 2.0 > 1.953125, still excluded


def test_ratio_is_symmetric_under_and_over_estimation():
    gt = np.full((2, 2), 2.0, dtype=np.float32)
    over = compute_depth_metrics(gt * 1.2, gt, config=CFG)
    under = compute_depth_metrics(gt / 1.2, gt, config=CFG)
    assert over["delta1"] == under["delta1"] == 1.0
    # float32 storage of gt/1.2 costs a few ulps, hence 1e-6 rather than exact
    assert over["rmse_log"] == pytest.approx(under["rmse_log"], abs=1e-6)


def test_invalid_gt_pixels_are_excluded():
    """Zeros and out-of-range ground truth must not contribute.

    gt = [[1, 0], [4, 200]] with max_depth=100 leaves pixels {1, 4} valid.
    pred is exact on those two and wildly wrong on the excluded ones, so a
    correct implementation still reports zero error.
    """
    gt = np.array([[1.0, 0.0], [4.0, 200.0]], dtype=np.float32)
    pred = np.array([[1.0, 50.0], [4.0, 1.0]], dtype=np.float32)
    m = compute_depth_metrics(pred, gt, config=CFG)
    assert m["n_valid"] == 2
    assert m["rmse"] == 0.0
    assert m["valid_ratio"] == pytest.approx(0.5)


def test_extra_mask_is_intersected_with_the_range_mask():
    mask = np.array([[True, True], [False, False]])
    pred = GT.copy()
    pred[1, 1] = 100.0  # masked out, must not matter
    m = compute_depth_metrics(pred, GT, mask=mask, config=CFG)
    assert m["n_valid"] == 2
    assert m["rmse"] == 0.0


def test_predictions_are_clamped_not_dropped():
    """A NaN/inf prediction is a failure and must be penalised, not ignored."""
    pred = np.array([[np.nan, np.inf], [4.0, 8.0]], dtype=np.float32)
    cfg = MetricConfig(min_depth=0.1, max_depth=10.0, clamp_pred=True)
    m = compute_depth_metrics(pred, GT, config=cfg)
    assert m["n_valid"] == 4  # nothing dropped
    assert m["rmse"] > 0  # and the bad pixels cost us


def test_no_valid_pixels_returns_nan_not_zero():
    gt = np.zeros((2, 2), dtype=np.float32)
    m = compute_depth_metrics(gt, gt, config=CFG)
    assert m["n_valid"] == 0
    assert all(math.isnan(m[name]) for name in METRIC_NAMES)


def test_shape_mismatch_raises():
    with pytest.raises(ValueError, match="same shape"):
        compute_depth_metrics(np.zeros((2, 3), np.float32), GT, config=CFG)


def test_eigen_crop_requires_the_nyu_resolution():
    cfg = MetricConfig(crop="eigen")
    with pytest.raises(ValueError, match="480x640"):
        compute_depth_metrics(GT, GT, config=cfg)

    big = np.full((480, 640), 2.0, dtype=np.float32)
    m = compute_depth_metrics(big, big, config=cfg)
    assert m["n_valid"] == (471 - 45) * (601 - 41)


def test_border6_crop_drops_a_six_pixel_frame():
    a = np.full((20, 20), 3.0, dtype=np.float32)
    m = compute_depth_metrics(a, a, config=MetricConfig(crop="border6"))
    assert m["n_valid"] == 8 * 8


def test_unknown_crop_is_rejected_at_config_time():
    with pytest.raises(ValueError, match="unknown crop"):
        MetricConfig(crop="middlebury")


def test_accumulator_image_pooling_averages_per_image():
    """Two frames with different valid-pixel counts.

    Frame A: 4 valid pixels, absrel 0.1.  Frame B: 1 valid pixel, absrel 0.5.
    Image pooling -> (0.1 + 0.5) / 2 = 0.3, regardless of the pixel counts.
    """
    acc = DepthMetricAccumulator(CFG, pooling="image")
    acc.update(GT * 1.1, GT)
    gt_b = np.array([[2.0, 0.0], [0.0, 0.0]], dtype=np.float32)
    acc.update(np.full((2, 2), 3.0, np.float32), gt_b)
    out = acc.compute()
    assert out["n_samples"] == 2
    assert out["absrel"] == pytest.approx((0.1 + 0.5) / 2, abs=1e-6)


def test_accumulator_pixel_pooling_weights_by_pixel_count():
    acc = DepthMetricAccumulator(CFG, pooling="pixel")
    acc.update(GT * 1.1, GT)
    gt_b = np.array([[2.0, 0.0], [0.0, 0.0]], dtype=np.float32)
    acc.update(np.full((2, 2), 3.0, np.float32), gt_b)
    out = acc.compute()
    # 4 pixels at 0.1 and 1 pixel at 0.5 -> (4*0.1 + 1*0.5) / 5 = 0.18
    assert out["absrel"] == pytest.approx(0.18, abs=1e-6)


def test_accumulator_skips_and_counts_empty_frames():
    acc = DepthMetricAccumulator(CFG)
    acc.update(GT, GT)
    acc.update(np.zeros((2, 2), np.float32), np.zeros((2, 2), np.float32))
    out = acc.compute()
    assert out["n_samples"] == 1
    assert out["n_skipped"] == 1
    assert not math.isnan(out["rmse"])


def test_singleton_axes_are_accepted():
    m = compute_depth_metrics(GT[None], GT[..., None], config=CFG)
    assert m["n_valid"] == 4


def test_metric_config_rejects_an_inverted_range():
    with pytest.raises(ValueError, match="min_depth < max_depth"):
        MetricConfig(min_depth=5.0, max_depth=1.0)


def test_runtime_measurement_reports_a_plausible_fps():
    from metrics.runtime_metrics import RuntimeConfig, measure_runtime

    calls = {"n": 0}

    def fn():
        calls["n"] += 1

    rt = measure_runtime(fn, device="cpu", config=RuntimeConfig(warmup=2, iters=5))
    assert calls["n"] == 7
    assert rt.iters == 5
    assert rt.fps > 0
    assert rt.latency_ms_median >= 0
    assert rt.device_info["device"] == "cpu"
