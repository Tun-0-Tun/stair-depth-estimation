"""Degradation + dataset + 3 training steps on a synthetic ``ours`` folder (CPU, tiny frames)."""
import json
import os

import numpy as np
import pytest

pytest.importorskip("pytorch_wavelets")
from synth_ours import make  # noqa: E402

from data.degradation import AstraHardDegradation, masked_downsample  # noqa: E402


def test_masked_downsample_marks_holes():
    d = np.ones((16, 16), np.float32) * 2
    d[:8, :8] = 0
    lr, v = masked_downsample(d, 8)
    assert not v[0, 0] and v[1, 1] and lr[0, 0] == 0 and lr[1, 1] == pytest.approx(2)


def test_severity_orders_holes_and_test_is_deterministic():
    rng = np.random.default_rng(0)
    gt = (rng.uniform(1, 4, (264, 344))).astype(np.float32)
    rgb = rng.random((264, 344, 3)).astype(np.float32)
    ctx = {"rgb": rgb, "sample_id": "s1"}
    inv = {}
    for sev in ("light", "medium", "strong"):
        d = AstraHardDegradation(severity=sev, scale=8)
        a, b = d(gt, seed=3, context=ctx), d(gt, seed=3, context=ctx)
        assert np.array_equal(a["sparse_depth"], b["sparse_depth"])
        inv[sev] = a["meta"]["input_invalid_frac"]
        assert a["lr_depth"].shape == (33, 43) and (a["lr_depth"] > 0).any()
    assert inv["light"] < inv["medium"] < inv["strong"], inv
    t = AstraHardDegradation(severity="strong", scale=8, stochastic=True)
    x = t(gt, seed=3, context=ctx)["sparse_depth"]
    t.epoch = 1
    assert not np.array_equal(x, t(gt, seed=3, context=ctx)["sparse_depth"])


def test_switches_off_means_no_added_holes():
    gt = np.full((64, 64), 2.0, np.float32)
    d = AstraHardDegradation(severity="light", edge_holes=False, glare_holes=False, blob_holes=False,
                             depth_noise=False, flying=False, far_dropout=False, base={"dropout_base": 0.0, "occlusion": False})
    o = d(gt, seed=0, context={"rgb": np.zeros((64, 64, 3), np.float32), "sample_id": "x"})
    assert o["meta"]["input_invalid_frac"] == 0.0


@pytest.mark.parametrize("fusion,edge", [("spatial", "gaussian"), ("stairnetv2_orig", "binary")])
def test_train_smoke_end_to_end(tmp_path, monkeypatch, fusion, edge):
    from PIL import Image

    from data.stairnet.cache import save_masks
    from models.aprime_runner import load_cfg, train

    root = make(tmp_path / "ours" / "sample")
    for f in root.glob("s*_rgb.png"):  # fake StairNet masks from the synthetic colours
        rgb = np.asarray(Image.open(f))
        cls = np.where(rgb[..., 0] > 120, 1, np.where(rgb[..., 2] > 100, 2, 0)).astype(np.uint8)
        save_masks(tmp_path / "ours" / "stairnet_cache", f.stem[:-4], cls, np.ones(cls.shape, np.float32))
    split = {"train": [1, 2, 3, 4], "val": [5], "test": [6]}
    (tmp_path / "split.json").write_text(json.dumps(split))
    monkeypatch.setenv("STAIR_DATA_ROOT", str(tmp_path))
    cfg = load_cfg("configs/aprime/_base.yaml", [
        f"data.split_file={tmp_path / 'split.json'}", "data.crop=null", "data.target_size=null",
        "model.checkpoint=null", f"model.fusion={fusion}", f"loss.edge_weight={edge}", "loss.alpha=2.0",
        "train.workers=0", "train.batch_size=2", "device=cpu"])
    cfg["name"] = "smoke"
    train(cfg, tmp_path / "out", smoke=4)
    rows = [json.loads(l) for l in (tmp_path / "out" / "train_log.jsonl").read_text().splitlines()]
    rows = [r for r in rows if "total" in r]
    assert all(np.isfinite(r["total"]) for r in rows) and rows[-1]["total"] < rows[0]["total"]


def test_block_center_sparse_keeps_one_point_per_valid_block():
    from data.degradation import masked_downsample
    from data.degradation.astra_hard import block_center_sparse

    d = np.full((32, 48), 2.0, np.float32)
    d[:8, :8] = 0  # a hole block
    sp = block_center_sparse(d, 8)
    lr, valid = masked_downsample(d, 8)
    assert (sp > 0).sum() == valid.sum() == 4 * 6 - 1
    assert sp[12, 12] == pytest.approx(2.0) and sp[4, 4] == 0 and sp[0, 0] == 0


def test_zone_center_sparse_one_point_per_valid_zone():
    from data.degradation.astra_hard import zone_center_sparse

    d = np.full((80, 160), 3.0, np.float32)
    d[:10, :20] = 0  # the first zone is a hole
    d[40:44, 60:80] = 1.0  # 40 % of one zone is near: the median must ignore it
    sp = zone_center_sparse(d, (8, 8))
    assert (sp > 0).sum() == 63 and sp[5, 10] == 0
    assert np.allclose(sp[sp > 0], 3.0)


def test_sensor_chain_preset_runs_through_the_pipeline(tmp_path, monkeypatch):
    """The main degradation (lateral noise -> projector shadow -> axial noise at x8) feeds A' and the baseline input."""
    from PIL import Image

    from data.aprime_dataset import APrimeDataset
    from data.stairnet.cache import save_masks
    from models.aprime_runner import load_cfg, train

    root = make(tmp_path / "ours" / "sample")
    for f in root.glob("s*_rgb.png"):
        rgb = np.asarray(Image.open(f))
        cls = np.where(rgb[..., 0] > 120, 1, np.where(rgb[..., 2] > 100, 2, 0)).astype(np.uint8)
        save_masks(tmp_path / "ours" / "stairnet_cache", f.stem[:-4], cls, np.ones(cls.shape, np.float32))
    (tmp_path / "split.json").write_text(json.dumps({"train": [1, 2, 3, 4], "val": [5], "test": [6]}))
    monkeypatch.setenv("STAIR_DATA_ROOT", str(tmp_path))
    cfg = load_cfg("configs/aprime/_base.yaml", [
        f"data.split_file={tmp_path / 'split.json'}", "data.crop=null", "data.target_size=null", "model.checkpoint=null",
        "degradation.preset=sequence_bicubic_x8_lateral_projector_axial", "train.workers=0", "device=cpu",
        "data.intrinsics=[150,150,88,64]"])
    cfg["name"] = "smoke_chain"
    ds = APrimeDataset(cfg, "train", train=True, with_raw=True)
    it = ds[0]
    h, w = it["raw"]["gt_depth"].shape
    assert it["lr"].shape == (1, h // 8, w // 8) and it["raw"]["sparse_depth"].shape == (h // 8, w // 8)
    assert 0.0 < float((it["lr_valid"] > 0).float().mean()) <= 1.0
    a = ds[0]["lr"].clone()
    ds.set_epoch(1)  # training: a new reproducible corruption every epoch
    assert not np.allclose(a.numpy(), ds[0]["lr"].numpy())
    train(cfg, tmp_path / "out", smoke=3)


def _chain_cfg(tmp_path, monkeypatch, extra_overrides=()):
    from PIL import Image

    from data.stairnet.cache import save_masks
    from models.aprime_runner import load_cfg

    root = make(tmp_path / "ours" / "sample")
    for f in root.glob("s*_rgb.png"):
        rgb = np.asarray(Image.open(f))
        cls = np.where(rgb[..., 0] > 120, 1, np.where(rgb[..., 2] > 100, 2, 0)).astype(np.uint8)
        save_masks(tmp_path / "ours" / "stairnet_cache", f.stem[:-4], cls, np.ones(cls.shape, np.float32))
    (tmp_path / "split.json").write_text(json.dumps({"train": [1, 2, 3, 4], "val": [5], "test": [6]}))
    monkeypatch.setenv("STAIR_DATA_ROOT", str(tmp_path))
    return load_cfg("configs/aprime/_base.yaml", [
        f"data.split_file={tmp_path / 'split.json'}", "data.crop=null", "data.target_size=null", "model.checkpoint=null",
        "degradation.preset=sequence_bicubic_x8_lateral_projector_axial", "train.workers=0", "device=cpu",
        "data.intrinsics=[150,150,88,64]", "loss.edge_weight=gaussian", "loss.use_planar_loss=true", *extra_overrides])


def test_stair_gate_switches_the_stair_terms_per_frame(tmp_path, monkeypatch):
    from data.aprime_dataset import APrimeDataset

    cfg = _chain_cfg(tmp_path, monkeypatch)
    plain = APrimeDataset(cfg, "train", train=True)
    ids = [r["id"] for r in plain.base.records]
    on = {i: bool(plain[k]["tread"].any()) for k, i in enumerate(ids)}
    assert any(on.values()), "the synthetic frames must have tread pixels"
    keep = next(i for i, v in on.items() if v)
    gate_file = tmp_path / "gate.json"
    gate_file.write_text(json.dumps({i: {"stair_like_strict": i == keep} for i in ids}))
    cfg["data"]["stair_gate_file"] = str(gate_file)
    gated = APrimeDataset(cfg, "train", train=True)
    for k, i in enumerate(ids):
        it = gated[k]
        if i == keep:
            assert it["tread"].any() and it["E"].any()
        else:
            assert not (it["tread"].any() or it["riser"].any() or it["E"].any() or it["band"].any() or it["ignore"].any())


def test_extra_source_is_mixed_in_and_trains(tmp_path, monkeypatch):
    from data.aprime_dataset import MixedDataset, build_train
    from models.aprime_runner import train

    cfg = _chain_cfg(tmp_path, monkeypatch)
    cfg["data"]["extra"] = [{"dataset": "ours", "stair_terms": "none", "normals_K": [150, 150, 88, 64], "max_frames": 4,
                             "loader_cfg": {"root": "ours/sample", "crop": None, "target_size": None}}]
    ds = build_train(cfg)
    assert isinstance(ds, MixedDataset) and len(ds) == len(ds.datasets[0]) + 4 and len(ds.datasets[1]) == 4
    assert ds.datasets[1].base.force_degradation  # the chain, never the sensor input an extra dataset may ship
    ds.set_epoch(1)
    extra_item = ds[len(ds) - 1]
    assert not (extra_item["tread"].any() or extra_item["E"].any())  # no stair terms on that source
    train(cfg, tmp_path / "out", smoke=3)
