#!/usr/bin/env python3
"""Put several methods side by side on the same frames: what goes in, what comes out, what the GT is.

    uv run --group baselines python scripts/compare_methods.py --dataset ours --frames s100 s304 s507
    uv run --group baselines python scripts/compare_methods.py --dataset ours -n 6 \
        --methods prior_da:random_sparse_500 omni_dc:random_sparse_500 dornet:bicubic_x4

A method is ``<model config>:<degradation config>`` (``configs/model``, ``configs/degradation``) and runs
with that model's default checkpoint.  One PNG per frame, three rows:

    inputs   RGB | GT | one panel per distinct degradation (500 points are dilated so they are visible)
    outputs  one prediction per method, title = method, input, AbsRel / RMSE / delta1 on this frame
    error    (pred - gt) / gt, red = too far, blue = too near, grey = no GT

Depth panels share one colour scale per frame (2nd-98th percentile of the GT).  Numbers come from
``metrics/depth_metrics.py``; everything else here is a picture.  Output: ``outputs/compare/<tag>/``.
For ``ours`` the GT is the sensor's own depth, so this shows how well a method undoes a
degradation of it, not how accurate the sensor is.
"""

from __future__ import annotations

import argparse
import csv
import gc
import sys
from pathlib import Path

import _bootstrap
import numpy as np

from baselines import build_baseline
from baselines.base import model_depth_range
from data.degradation import build_degradation
from data.loaders import build_dataset
from metrics.depth_metrics import MetricConfig, compute_depth_metrics
from utils.config import load_config
from utils.misc import resize_depth, set_seed

DEFAULT_METHODS = (
    "prior_da:random_sparse_500",
    "omni_dc:random_sparse_500",
    "dornet:bicubic_x4",
    "ducos:bicubic_x4",
    "wave:bicubic_x8",
    "promptda:bicubic_x8",
)


def purge_third_party() -> None:
    """Forget every module and ``sys.path`` entry that came from ``third_party/``.

    Upstream repos reuse top-level names (OMNI-DC and DuCos both have a ``model`` package, DORNet and
    DuCos both have ``utils`` / ``data``), so a second method would import the first one's modules.  Safe
    here because each model has already predicted every frame by the time this runs.
    """
    tp = str(_bootstrap.REPO_ROOT / "third_party")
    doomed = []
    for name, mod in list(sys.modules.items()):
        # Decide first, delete after: a namespace package recomputes its __path__ from its
        # parent in sys.modules, so deleting while looking raises KeyError.
        try:
            places = [getattr(mod, "__file__", None), *list(getattr(mod, "__path__", None) or [])]
        except Exception:  # torch.classes and friends: never ours
            continue
        if any(p and str(p).startswith(tp) for p in places):
            doomed.append(name)
    for name in doomed:
        sys.modules.pop(name, None)
    sys.path[:] = [p for p in sys.path if not str(p).startswith(tp)]


def input_panel(sample: dict, kind: str) -> np.ndarray:
    """The map a method of this kind receives, at GT size, 0 = nothing."""
    from scipy.ndimage import maximum_filter

    h, w = sample["gt_depth"].shape
    if kind == "sparse":
        sp = sample["sparse_depth"]
        return maximum_filter(sp, size=7) if 0 < int((sp > 0).sum()) < 5000 else sp
    return resize_depth(sample["lr_depth"], (h, w), mode="nearest")


def pick_frames(dataset, ids: list[str] | None, n: int) -> list[int]:
    index = {dataset.record_id(r): i for i, r in enumerate(dataset.records)}
    if ids:
        missing = [s for s in ids if s not in index]
        if missing:
            raise SystemExit(f"not in the dataset: {missing}; first ids: {list(index)[:5]}")
        return [index[s] for s in ids]
    return sorted({int(i) for i in np.linspace(0, len(dataset) - 1, n)})


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--dataset", default="ours", help="configs/dataset/<name>.yaml")
    ap.add_argument("--methods", nargs="+", default=list(DEFAULT_METHODS), help="model:degradation")
    ap.add_argument("--frames", nargs="*", help="sample ids, e.g. s100 s304 (default: -n spread)")
    ap.add_argument(
        "-n", type=int, default=6, help="evenly spread frames when --frames is not given"
    )
    ap.add_argument("--device", default="auto")
    ap.add_argument("--tag", default=None, help="output folder name (default: the dataset)")
    ap.add_argument(
        "--err", type=float, default=0.25, help="error colour range, relative (0.25 = +-25%%)"
    )
    args = ap.parse_args()

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    set_seed(0)
    dataset_cfg = dict(load_config("dataset", args.dataset))
    specs = [m.split(":", 1) for m in args.methods]
    if any(len(s) != 2 for s in specs):
        raise SystemExit("every method must look like model:degradation, e.g. dornet:bicubic_x4")

    # one dataset per distinct degradation: same records, same seeded inputs
    datasets = {}
    for _, deg in specs:
        if deg not in datasets:
            datasets[deg] = build_dataset(
                dict(dataset_cfg),
                degradation=build_degradation(dict(load_config("degradation", deg))),
            )
    ref = next(iter(datasets.values()))
    frames = pick_frames(ref, args.frames, args.n)
    samples = {deg: [ds[i] for i in frames] for deg, ds in datasets.items()}
    metric_cfg = MetricConfig(min_depth=ref.min_depth, max_depth=ref.max_depth, crop="none")
    ids = [ref.record_id(ref.records[i]) for i in frames]
    print(f"[frames] {ids}")

    results = []  # (model, degradation, kind, [pred per frame], [metrics per frame])
    for name, deg in specs:
        cfg = dict(load_config("model", name))
        dcfg = dict(load_config("degradation", deg))
        if "scale" in cfg and "scale" in dcfg and int(cfg["scale"]) != int(dcfg["scale"]):
            raise SystemExit(
                f"{name}: model scale {cfg['scale']} but degradation {deg} is x{dcfg['scale']}; "
                "pick the matching pair (the weights are per scale)"
            )
        lo, hi = model_depth_range(cfg, metric_cfg.min_depth, metric_cfg.max_depth)
        model = build_baseline(cfg, device=args.device, min_depth=lo, max_depth=hi)
        model.load(cfg.get("checkpoint"))
        preds, mets = [], []
        for s in samples[deg]:
            p = model.predict_sample(s)
            preds.append(p)
            mets.append(compute_depth_metrics(p, s["gt_depth"], s["mask"], metric_cfg))
        results.append((name, deg, model.input_modality, preds, mets))
        print(f"[{name}:{deg}] mean absrel {np.mean([m['absrel'] for m in mets]):.4f}")
        del model
        gc.collect()
        import contextlib

        import torch

        with contextlib.suppress(Exception):
            torch.cuda.empty_cache()
        purge_third_party()

    out = Path(_bootstrap.REPO_ROOT) / "outputs" / "compare" / (args.tag or args.dataset)
    out.mkdir(parents=True, exist_ok=True)
    cmap = plt.get_cmap("turbo").copy()
    cmap.set_bad("black")
    overlay = cmap.copy()
    overlay.set_bad((0, 0, 0, 0))
    ecmap = plt.get_cmap("RdBu_r").copy()
    ecmap.set_bad("#bbbbbb")

    degs = list(datasets)
    kind_of = {deg: next(r[2] for r in results if r[1] == deg) for deg in degs}
    ncols = max(2 + len(degs), len(results))
    rows_csv = []
    for fi, sid in enumerate(ids):
        base = samples[degs[0]][fi]
        gt = np.where(base["mask"], base["gt_depth"], np.nan)
        v = gt[np.isfinite(gt)]
        vmin, vmax = (np.percentile(v, 2), np.percentile(v, 98)) if v.size else (0.0, 1.0)

        fig, axes = plt.subplots(3, ncols, figsize=(3.4 * ncols, 7.6), squeeze=False)
        axes[0, 0].imshow(np.clip(base["rgb"], 0, 1))
        axes[0, 0].set_title("RGB", fontsize=8)
        axes[0, 1].imshow(gt, cmap=cmap, vmin=vmin, vmax=vmax)
        axes[0, 1].set_title("GT (для ours: глубина сенсора)", fontsize=8)
        for j, deg in enumerate(degs):
            s = samples[deg][fi]
            inp = input_panel(s, kind_of[deg])
            axes[0, 2 + j].imshow(np.clip(s["rgb"], 0, 1) * 0.35)
            axes[0, 2 + j].imshow(
                np.where(inp > 0, inp, np.nan), cmap=overlay, vmin=vmin, vmax=vmax
            )
            axes[0, 2 + j].set_title(f"вход: {deg}", fontsize=8)
        for j, (name, deg, _, preds, mets) in enumerate(results):
            s = samples[deg][fi]
            m = mets[fi]
            axes[1, j].imshow(preds[fi], cmap=cmap, vmin=vmin, vmax=vmax)
            axes[1, j].set_title(
                f"{name} [{deg}]\nAbsRel {m['absrel']:.3f}  RMSE {m['rmse']:.3f}  δ1 {m['delta1']:.3f}",
                fontsize=8,
            )
            rel = np.where(
                s["mask"], (preds[fi] - s["gt_depth"]) / np.maximum(s["gt_depth"], 1e-6), np.nan
            )
            axes[2, j].imshow(rel, cmap=ecmap, vmin=-args.err, vmax=args.err)
            rows_csv.append(
                {
                    "sample_id": sid,
                    "model": name,
                    "degradation": deg,
                    **{k: m[k] for k in ("absrel", "rmse", "delta1")},
                }
            )
        for a in axes.ravel():
            a.axis("off")
        meta = base["meta"]
        tags = " | ".join(
            f"{k}: {meta[k]}"
            for k in ("stairs_id", "view", "illumination", "occlusion")
            if meta.get(k) is not None
        )
        fig.suptitle(
            f"{sid}  {tags}\nряд 3: (pred−gt)/gt, красный = дальше, синий = ближе, серый = нет GT, ±{args.err:.0%}",
            fontsize=9,
        )
        fig.tight_layout(rect=(0, 0, 1, 0.94))
        fig.savefig(out / f"{sid}.png", dpi=80)
        plt.close(fig)
        print(f"  {out / (sid + '.png')}")

    with (out / "metrics.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows_csv[0]))
        w.writeheader()
        w.writerows(rows_csv)
    print(f"[done] {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
