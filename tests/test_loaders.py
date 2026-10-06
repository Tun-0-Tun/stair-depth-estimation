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
from data.loaders.base import BaseDepthDataset, DatasetInfo
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


class _NativeSparse(BaseDepthDataset):
    """Tiny in-memory dataset that ships a real sparse input, like VOID."""

    info = DatasetInfo(name="native_sparse", modality="sparse", min_depth=0.3, max_depth=8.0)

    def _build_index(self):
        return [{"id": "0"}, {"id": "1"}]

    def _load_raw(self, record):
        gt = np.full((32, 48), 2.0, np.float32)
        sparse = np.zeros_like(gt)
        sparse[::8, ::8] = 5.0  # deliberately different from GT
        return {
            "rgb": np.zeros((32, 48, 3), np.float32),
            "gt_depth": gt,
            "sparse_depth": sparse,
            "sample_id": record["id"],
        }


def test_native_input_is_used_by_default():
    ds = _NativeSparse(root=".", degradation=build_degradation({"type": "identity"}), strict=False)
    s = ds[0]
    assert s["meta"]["input_modality"] == "sparse"
    assert s["meta"]["forced_degradation"] is False
    assert float(s["sparse_depth"].max()) == 5.0


def test_force_degradation_replaces_native_input_with_degraded_gt():
    ds = _NativeSparse(
        root=".",
        degradation=build_degradation({"type": "identity"}),
        force_degradation=True,
        strict=False,
    )
    s = ds[0]
    validate_sample(s, "forced")
    assert s["meta"]["input_modality"] == "synthetic"
    assert s["meta"]["forced_degradation"] is True
    assert np.array_equal(s["sparse_depth"], s["gt_depth"])  # identity on GT, not the 5.0 map
    assert np.array_equal(ds[1]["sparse_depth"], ds[1]["sparse_depth"])


def test_force_degradation_without_a_model_fails_early():
    with pytest.raises(ValueError, match="force_degradation"):
        _NativeSparse(root=".", degradation=None, force_degradation=True, strict=False)


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
        {"type": "projector_shadow"},
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

    # `identity`, a noise-free `bicubic_sr` and `projector_shadow` are deterministic
    # by construction: they never draw from the rng, so the seed cannot change them.
    if cfg["type"] not in ("identity", "bicubic_sr", "projector_shadow"):
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


def _shadow(gt, **params):
    deg = build_degradation({"type": "projector_shadow", **params})
    out = deg(gt.astype(np.float32), seed=0)
    return (out["sparse_depth"] == 0) & (gt > 0), out["meta"]


def test_projector_shadow_no_self_shadowing_on_planes():
    """A plane never shadows itself: no acne on a wall or on a floor seen at an angle."""
    wall = np.full((96, 128), 2.0)
    for t in [(0.04, 0.0, 0.0), (0.0, -0.04, 0.0), (0.03, -0.03, 0.01)]:
        holes, meta = _shadow(wall, offset_t=t)
        assert not holes.any() and meta["shadow_ratio"] == 0.0, t

    # a real floor plane 1 m below the camera (y down): Z = fy * 1.0 / (v - cy),
    # with the horizon above the image so every row sees the floor (~0.9-10 m)
    f, cy = 100.0, -10.0
    v = np.arange(96, dtype=np.float64)[:, None]
    floor = np.repeat(f * 1.0 / (v - cy), 128, axis=1)
    for t in [(0.04, 0.0, 0.0), (0.0, -0.04, 0.0)]:
        holes, _ = _shadow(floor, offset_t=t, fx=f, cy=cy)
        assert not holes.any(), t


def test_projector_shadow_matches_stereo_occlusion_on_a_vertical_edge():
    """Horizontal offset + vertical edge: the band the existing IR-camera model predicts."""
    from data.degradation.active_stereo_astra2 import _occlusion_shadow

    f, b, z_bg, z_fg = 570.0, 0.04, 3.0, 1.2
    gt = np.full((64, 128), z_bg)
    gt[:, 64:] = z_fg  # foreground on the right
    holes, _ = _shadow(gt, offset_t=(b, 0.0, 0.0), fx=f)

    expected = f * b * (1 / z_fg - 1 / z_bg)  # ~11.4 px
    assert not holes[:, 64:].any(), "the nearer surface is never in shadow"
    width = holes[:, :64].sum(axis=1)
    assert np.all(np.abs(width - expected) <= 1.0), (width.min(), width.max(), expected)
    assert holes[:, 64 - int(expected) : 64].all(), "the band sits right next to the edge"

    ref = _occlusion_shadow(gt, f * b, gt > 0)
    iou = (holes & ref).sum() / max((holes | ref).sum(), 1)
    assert iou > 0.9, iou


def test_projector_shadow_holes_below_a_nosing_need_a_vertical_offset():
    """Horizontal edge, near surface on top: holes appear below it only for ty < 0."""
    z_bg, z_fg = 3.0, 1.2
    gt = np.full((96, 128), z_bg)
    gt[:48] = z_fg  # the nosing / near step occupies the upper half

    above, _ = _shadow(gt, offset_t=(0.0, -0.04, 0.0))
    assert above[:48].sum() == 0
    assert above[48:52].all(), "projector above the camera: a band right below the edge"

    below, _ = _shadow(gt, offset_t=(0.0, 0.04, 0.0))
    assert not below.any(), "projector below the camera: the edge casts nothing downwards"

    side, _ = _shadow(gt, offset_t=(0.04, 0.0, 0.0))
    assert not side.any(), "a horizontal offset casts no shadow across a horizontal edge"


def test_projector_shadow_fov_clip_cuts_a_border_band():
    f, tx, z = 570.0, 0.04, 2.0
    wall = np.full((48, 96), z)

    holes, _ = _shadow(wall, offset_t=(tx, 0.0, 0.0), fx=f, fov_clip=True)
    width = holes.sum(axis=1)
    assert np.all(np.abs(width - f * tx / z) <= 1.0), (width.min(), f * tx / z)
    assert holes[:, :10].all() and not holes[:, 20:].any(), "the band is on the left border"

    holes, _ = _shadow(wall, offset_t=(tx, 0.0, 0.0), fx=f, fov_clip=False)
    assert not holes.any()


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


def test_dtof_footprint_is_exactly_what_fill_ratio_asks_for():
    """fill_ratio must be able to write a single pixel per zone.

    That is DEPTHOR's own convention, and feeding it the 0.6-footprint variant
    instead gives ~1700 pixels per zone -- three orders of magnitude denser than
    its training input, which silently sends the model back to its depth prior.
    """
    gt = np.full((64, 64), 2.0, dtype=np.float32)

    centre = build_degradation(
        {"type": "dtof_sim", "zones_h": 8, "zones_w": 8, "fill_ratio": 0.0, "zone_dropout": 0.0}
    )(gt, seed=0)
    n_zones = centre["meta"]["n_zones_reported"]
    assert (centre["sparse_depth"] > 0).sum() == n_zones, "fill_ratio=0 must write 1 px per zone"

    # And a wide footprint writes the requested patch, not one pixel more.
    wide = build_degradation(
        {"type": "dtof_sim", "zones_h": 8, "zones_w": 8, "fill_ratio": 0.5, "zone_dropout": 0.0}
    )(gt, seed=0)
    zone_h = zone_w = 64 // 8
    per_zone = round(zone_h * 0.5) * round(zone_w * 0.5)
    assert (wide["sparse_depth"] > 0).sum() == wide["meta"]["n_zones_reported"] * per_zone


def test_zero_byte_file_raises_oserror_not_something_else(tmp_path):
    """The run loop forgives OSError only, so an unreadable frame must raise one.

    An interrupted copy onto the shared data folder leaves zero-byte files --
    MinJiang has two out of 909. PIL's UnidentifiedImageError is an OSError, so
    scripts/run_baseline.py counts the frame and carries on; if a loader ever
    wrapped it in something else, a 909-frame run would die at frame 901 again.
    """
    pytest.importorskip("PIL")

    root = tmp_path / "Middlebury"
    root.mkdir()
    _write_depth_enhance_pair(root, "Middlebury_01")
    (root / "Middlebury_02_output_color.png").write_bytes(b"")
    (root / "Middlebury_02_output_depth.png").write_bytes(b"")

    ds = build_dataset(
        {"loader": "middlebury", "root": str(root)},
        degradation=build_degradation({"type": "bicubic_sr", "scale": 4}),
    )
    assert len(ds) == 2
    validate_sample(ds[0], "good frame still loads")
    with pytest.raises(OSError):
        ds[1]


def test_rgbd_stair_decodes_per_frame_range_and_drops_wide_frames(tmp_path):
    """8-bit depth is only metric through the extrinsics range; get it wrong and
    every frame is off by a different factor, which no metric flags."""
    Image = pytest.importorskip("PIL.Image")

    base = tmp_path / "test"
    for d in ("images", "depthes", "extrinsicses"):
        (base / d).mkdir(parents=True)
    v = np.array([[0, 255], [51, 102]], dtype=np.uint8)
    for n, rng in ((0, "1000 3550"), (1, "0 65535")):  # frame 1: stray far pixel
        Image.fromarray(np.zeros((2, 2, 3), np.uint8)).save(base / "images" / f"color_{n}.png")
        Image.fromarray(v).save(base / "depthes" / f"Depth_{n}.png")
        (base / "extrinsicses" / f"Extrinsics_{n}.txt").write_text(f"{rng} 0.4 -9.0 -2.8")

    ds = build_dataset(
        {"loader": "rgbd_stair", "root": str(tmp_path)},
        degradation=build_degradation({"type": "bicubic_sr", "scale": 1}),
    )
    assert len(ds) == 1, "the 65 m-range frame must be dropped"
    gt = ds[0]["gt_depth"]
    assert gt[0, 0] == 0.0, "0 is no measurement, not dmin"
    assert gt[0, 1] == pytest.approx(3.55) and gt[1, 0] == pytest.approx(1.51)
    validate_sample(ds[0], "rgbd_stair")


def test_arkitscenes_pairs_lr_and_gt_in_metres(tmp_path):
    Image = pytest.importorskip("PIL.Image")

    vid = tmp_path / "data" / "upsampling" / "Validation" / "123"
    for d in ("wide", "lowres_depth", "highres_depth"):
        (vid / d).mkdir(parents=True)
    Image.fromarray(np.zeros((16, 16, 3), np.uint8)).save(vid / "wide" / "123_1.0.png")
    Image.fromarray(np.full((2, 2), 1500, np.uint16)).save(vid / "lowres_depth" / "123_1.0.png")
    Image.fromarray(np.full((16, 16), 1500, np.uint16)).save(vid / "highres_depth" / "123_1.0.png")

    s = build_dataset({"loader": "arkitscenes", "root": str(tmp_path)})[0]
    assert s["lr_depth"].shape == (2, 2) and s["lr_depth"].max() == pytest.approx(1.5)
    assert s["gt_depth"].max() == pytest.approx(1.5) and s["meta"]["lr_scale"] == 8
    validate_sample(s, "arkitscenes")


def test_resize_sparse_keeps_every_zone_point():
    """Nearest sampling keeps a point only if it sits on the sampled grid; on a 3x
    shrink that is 1 pixel in 9, so off-grid zones vanish and DEPTHOR silently
    runs on fewer points."""
    from utils.misc import resize_depth, resize_sparse

    big = np.zeros((1440, 1920), np.float32)
    # zone centres, one pixel off the 3x grid
    big[91::180, 121::240] = np.arange(1, 65, dtype=np.float32).reshape(8, 8)
    small = resize_sparse(big, (480, 640))
    assert (small > 0).sum() == 64 and sorted(small[small > 0]) == list(range(1, 65))
    assert (resize_depth(big, (480, 640), "nearest") > 0).sum() < 64, "the trap this avoids"


def test_simulate_from_sensor_replaces_the_native_input(tmp_path):
    """DEPTHOR on ARKitScenes: 64 zones from the real LiDAR, GT stays the laser."""
    Image = pytest.importorskip("PIL.Image")

    vid = tmp_path / "data" / "upsampling" / "Validation" / "1"
    for d in ("wide", "lowres_depth", "highres_depth"):
        (vid / d).mkdir(parents=True)
    Image.fromarray(np.zeros((96, 128, 3), np.uint8)).save(vid / "wide" / "1_1.png")
    Image.fromarray(np.full((12, 16), 1500, np.uint16)).save(vid / "lowres_depth" / "1_1.png")
    Image.fromarray(np.full((96, 128), 3000, np.uint16)).save(vid / "highres_depth" / "1_1.png")

    deg = build_degradation(dict(load_config("degradation", "dtof_l5_center")))
    s = build_dataset(
        {"loader": "arkitscenes", "root": str(tmp_path), "simulate_from": "sensor"}, degradation=deg
    )[0]
    pts = s["sparse_depth"][s["sparse_mask"]]
    assert 0 < pts.size <= 64, "one pixel per zone, not the dense LiDAR map"
    assert np.allclose(pts, 1.5, atol=0.2), "zones come from the sensor (1.5 m), not the GT (3 m)"
    assert s["gt_depth"].max() == pytest.approx(3.0) and s["meta"]["simulated_from"] == "sensor"
    validate_sample(s, "arkitscenes simulate_from")


def test_nyuv2_mat_range_and_simulate_from_gt_give_the_dsr_protocol(tmp_path):
    """DuCos/WAVE on NYU: last 449 frames, LR input bicubic from GT -- not from rawDepths."""
    h5py = pytest.importorskip("h5py")

    with h5py.File(tmp_path / "nyu_depth_v2_labeled.mat", "w") as f:
        f["images"] = np.zeros((6, 3, 16, 8), np.uint8)  # MATLAB order: (N, 3, W, H)
        f["depths"] = np.full((6, 16, 8), 2.0, np.float32)
        f["rawDepths"] = np.full((6, 16, 8), 9.0, np.float32)

    ds = build_dataset(
        {
            "loader": "nyuv2",
            "root": str(tmp_path),
            "layout": "mat",
            "mat_range": [4, 6],
            "simulate_from": "gt",
        },
        degradation=build_degradation({"type": "bicubic_sr", "scale": 4}),
    )
    assert [r["id"] for r in ds.records] == ["nyu_0004", "nyu_0005"]
    assert np.allclose(ds[0]["lr_depth"], 2.0), "input must come from the GT, not rawDepths"


def test_arkitscenes_skips_frames_whose_rgb_is_rotated_against_the_depth(tmp_path):
    """A portrait RGB beside a landscape depth cannot be aligned; it must be dropped and
    counted, not fail the whole run at frame 3000 (seen on the server)."""
    Image = pytest.importorskip("PIL.Image")

    vid = tmp_path / "data" / "upsampling" / "Validation" / "1"
    for d in ("wide", "lowres_depth", "highres_depth"):
        (vid / d).mkdir(parents=True)
    for name, rgb_hw in (("1_1.0", (96, 128)), ("1_2.0", (128, 96))):  # second RGB is portrait
        Image.fromarray(np.zeros((*rgb_hw, 3), np.uint8)).save(vid / "wide" / f"{name}.png")
        Image.fromarray(np.full((12, 16), 1500, np.uint16)).save(
            vid / "lowres_depth" / f"{name}.png"
        )
        Image.fromarray(np.full((96, 128), 1500, np.uint16)).save(
            vid / "highres_depth" / f"{name}.png"
        )

    ds = build_dataset({"loader": "arkitscenes", "root": str(tmp_path)})
    assert [r["id"] for r in ds.records] == ["1/1_1.0"]
