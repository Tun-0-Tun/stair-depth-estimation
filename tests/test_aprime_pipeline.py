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
