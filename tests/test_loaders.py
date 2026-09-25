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


def test_nyuv2_mat_layout_transposes_matlab_axes(tmp_path):
    """MATLAB is column-major, so h5py hands back (N, 3, W, H) / (N, W, H).

    Getting the transpose wrong yields a plausible-looking but rotated image
    and a depth map that does not line up with it -- no crash, just a quietly
    wrong column in the table.  Pin the axes with a non-square asymmetric probe.
    """
    h5py = pytest.importorskip("h5py")

    n, h, w = 2, 4, 6
    depths = np.arange(n * w * h, dtype=np.float32).reshape(n, w, h)
    images = np.arange(n * 3 * w * h, dtype=np.uint8).reshape(n, 3, w, h)

    root = tmp_path / "nyu_depth_v2"
    root.mkdir()
    with h5py.File(root / "nyu_depth_v2_labeled.mat", "w") as f:
        f["images"], f["depths"], f["rawDepths"] = images, depths, depths

    ds = build_dataset(
        {"loader": "nyuv2", "root": str(root), "layout": "mat"},
        degradation=build_degradation({"type": "identity"}),
    )
    assert len(ds) == n

    sample = ds[1]
    assert sample["rgb"].shape == (h, w, 3)
    assert sample["gt_depth"].shape == (h, w)
    assert np.array_equal(sample["gt_depth"], depths[1].T)
    assert np.allclose(sample["rgb"], images[1].transpose(2, 1, 0) / 255.0)
    assert sample["meta"]["sparse_source"] == "kinect_raw"
    validate_sample(sample, "nyuv2")


def _write_depth_enhance_pair(root, stem, h=70, w=100):
    """One scene in the Depth_Enh flat layout: uint8 RGB + uint8 depth."""
    from PIL import Image

    rng = np.random.default_rng(abs(hash(stem)) % 2**32)
    Image.fromarray(rng.integers(0, 256, (h, w, 3), dtype=np.uint8)).save(
        root / f"{stem}_output_color.png"
    )
    Image.fromarray(rng.integers(1, 256, (h, w), dtype=np.uint8)).save(
        root / f"{stem}_output_depth.png"
    )


def test_depth_enhance_pairs_by_name_and_modcrops(tmp_path):
    """Lu/Middlebury pair two independently sorted lists -- and crop to /16.

    Both are silent failure modes: a mismatched pair scores depth against the
    wrong photo, and a missing modcrop scores a different region than every
    published number for these sets.
    """
    pytest.importorskip("PIL")

    root = tmp_path / "Middlebury"
    root.mkdir()
    for stem in ("Middlebury_01", "Middlebury_02", "Middlebury_03"):
        _write_depth_enhance_pair(root, stem)

    ds = build_dataset(
        {"loader": "middlebury", "root": str(root)},
        degradation=build_degradation({"type": "bicubic_sr", "scale": 4}),
    )
    assert len(ds) == 3

    sample = ds[0]
    assert sample["gt_depth"].shape == (64, 96), "70x100 must be cropped to a multiple of 16"
    assert sample["rgb"].shape == (64, 96, 3)
    assert 0.0 < float(sample["gt_depth"].max()) <= 1.0, "depth is uint8/255, not metres"
    assert sample["meta"]["dataset"] == "middlebury", "name comes from the directory"
    assert "NOT metres" in sample["meta"]["depth_unit"]
    validate_sample(sample, "middlebury")

    # A stray extra RGB must fail loudly, not shift every pair by one.
    (root / "Middlebury_04_output_color.png").write_bytes(
        (root / "Middlebury_01_output_color.png").read_bytes()
    )
    with pytest.raises(FileNotFoundError, match="files but"):
        build_dataset({"loader": "middlebury", "root": str(root)})


def test_hypersim_ray_distance_becomes_planar_depth():
    """Hypersim stores distance to the optical centre, not Z.

    Silent failure mode: skip the conversion and every frame reads a few per
    cent deep towards the corners, systematically, with nothing to notice.
    Pin it against the closed-form pinhole result.
    """
    from data.loaders.hypersim import _planar_scale

    h, w, focal = 24, 32, 40.0
    # M_cam_from_uv for a pinhole: u,v in [-1,1] span the frame, camera looks -z.
    m = np.array([[w / 2, 0.0, 0.0], [0.0, h / 2, 0.0], [0.0, 0.0, -focal]])
    scale = _planar_scale(m, h, w)

    assert scale.shape == (h, w)
    assert (scale <= 1.0 + 1e-6).all(), "planar depth can never exceed the ray distance"

    # Same thing written directly: planar = d * f / ||(x, y, f)|| in pixels.
    x = np.linspace(-w / 2 + 0.5, w / 2 - 0.5, w)[None, :]
    y = np.linspace(-h / 2 + 0.5, h / 2 - 0.5, h)[:, None]
    expected = focal / np.sqrt(x**2 + y**2 + focal**2)
    assert np.allclose(scale, expected, atol=1e-6)

    # The centre of the frame looks straight down the axis: no correction there,
    # while a corner is measurably shorter.
    assert scale[h // 2, w // 2] == pytest.approx(scale.max(), rel=1e-3)
    assert scale[0, 0] < 0.95
