#!/usr/bin/env python3
"""Put several methods side by side on the same frames: what goes in, what comes out, what the GT is.

    uv run --group baselines python scripts/compare_methods.py --dataset ours --frames s100 s304 s507
    uv run --group baselines python scripts/compare_methods.py --dataset ours -n 6 \
        --methods prior_da:random_sparse_500 omni_dc:random_sparse_500 dornet:bicubic_x4
    # a dataset with a real sensor input: degradation "none" (the dataset's own input is used)
    uv run --group baselines python scripts/compare_methods.py --dataset void_stairs -n 6 \
        --methods prior_da:none omni_dc:none nn_fill:none
    # ... or throw that input away and simulate one from the GT
    uv run --group baselines python scripts/compare_methods.py --dataset void_stairs \
        --methods dornet:bicubic_x4 --set dataset.simulate_from=gt

A method is ``<model config>:<degradation config>[@lr_scale]`` (``configs/model``, ``configs/degradation``) and
runs with that model's default checkpoint.  For a degradation without a scale (``astra2_uncalibrated``) SR
methods get its output hole-filled and downsampled by the model's own scale (or ``@4`` / ``@8``).  One PNG per frame:

    row 1    RGB | GT | one prediction per method; each title = method, input, AbsRel / RMSE / delta1 here
    row 2    what each method received, under its prediction (500 points are dilated so they are visible);
             --no-inputs drops it
    row 3    --error: (pred - gt) / gt, red = too far, blue = too near, grey = no GT
             --mask-pred blanks the predictions where the GT is missing (the default shows what the
             model draws there, which is invented depth)

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
from utils.config import apply_overrides, load_config
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
    ap.add_argument(
        "--set",
        nargs="*",
        default=[],
        metavar="dataset.key=value",
        help="override dataset options, e.g. dataset.simulate_from=gt dataset.frame_stride=20",
    )
    ap.add_argument(
        "--no-inputs",
        dest="inputs",
        action="store_false",
        help="drop the row with each method's input",
    )
    ap.add_argument("--error", action="store_true", help="add a row with the signed relative error")
    ap.add_argument(
        "--mask-pred", action="store_true", help="blank predictions where the GT is missing"
    )
    args = ap.parse_args()

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    set_seed(0)
    dataset_cfg = dict(load_config("dataset", args.dataset))
    if args.set:
        bad = [o for o in args.set if not o.startswith("dataset.")]
        if bad:
            raise SystemExit(f"--set only takes dataset.* keys, got {bad}")
        dataset_cfg = apply_overrides({"dataset": dataset_cfg}, args.set)["dataset"]
    # method = model:degradation[@lr_scale]. A degradation that has no scale of its own (astra2_*, ...)
    # hands SR methods a hole-filled map downsampled by the dataset's lr_scale; that must match the
    # weights (x4 / x8), so it is taken from the model config unless given with @.
    specs, deg_of = [], {}  # specs: (model, key); deg_of: key -> (degradation, lr_scale | None)
    for m in args.methods:
        try:
            name, rest = m.split(":", 1)
        except ValueError:
            raise SystemExit(
                "every method must look like model:degradation[@scale], e.g. dornet:bicubic_x4"
            ) from None
        deg, _, at = rest.partition("@")
        dcfg = load_config("degradation", deg)
        mscale = dict(load_config("model", name)).get("scale")
        if at and "scale" in dcfg:
            raise SystemExit(f"{m}: {deg} already fixes the scale (x{dcfg['scale']}); drop @{at}")
        lr_scale = int(at) if at else (int(mscale) if mscale and "scale" not in dcfg else None)
        key = f"{deg}@{lr_scale}" if lr_scale else deg
        specs.append((name, key))
        deg_of[key] = (deg, lr_scale)

    # one dataset per distinct input: same records, same seeded inputs
    datasets = {}
    for _, key in specs:
        if key not in datasets:
            deg, lr_scale = deg_of[key]
            cfg_d = dict(dataset_cfg, **({"lr_scale": lr_scale} if lr_scale else {}))
            datasets[key] = build_dataset(
                cfg_d, degradation=build_degradation(dict(load_config("degradation", deg)))
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
        dcfg = dict(load_config("degradation", deg_of[deg][0]))
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
        ok = [m["absrel"] for m in mets if m["n_valid"] > 0]
        print(
            f"[{name}:{deg}] mean absrel {np.mean(ok) if ok else float('nan'):.4f} over {len(ok)}/{len(mets)} frames"
        )
        del model
        gc.collect()
        import contextlib

        import torch

        with contextlib.suppress(Exception):
            torch.cuda.empty_cache()
        purge_third_party()

    empty = [ids[i] for i, m in enumerate(results[0][4]) if m["n_valid"] == 0]
    if empty:
        print(
            f"[warn] no GT pixel inside the {metric_cfg.min_depth:g}-{metric_cfg.max_depth:g} m window "
            f"in {len(empty)} frame(s), scored as nan and left out of the means: {empty}"
        )

    out = Path(_bootstrap.REPO_ROOT) / "outputs" / "compare" / (args.tag or args.dataset)
    out.mkdir(parents=True, exist_ok=True)
    cmap = plt.get_cmap("turbo").copy()
    cmap.set_bad("black")
    overlay = cmap.copy()
    overlay.set_bad((0, 0, 0, 0))
    ecmap = plt.get_cmap("RdBu_r").copy()
    ecmap.set_bad("#bbbbbb")

    degs = list(datasets)
    ncols = 2 + len(results)
    nrows = 1 + int(args.inputs) + int(args.error)
    rows_csv = []
    for fi, sid in enumerate(ids):
        base = samples[degs[0]][fi]
        gt = np.where(base["mask"], base["gt_depth"], np.nan)
        v = gt[np.isfinite(gt)]
        vmin, vmax = (np.percentile(v, 2), np.percentile(v, 98)) if v.size else (0.0, 1.0)

        fig, axes = plt.subplots(
            nrows, ncols, figsize=(3.0 * ncols, 2.6 * nrows + 0.9), squeeze=False
        )
        # row 0: RGB | GT | one prediction per method
        axes[0, 0].imshow(np.clip(base["rgb"], 0, 1))
        axes[0, 0].set_title("RGB", fontsize=8)
        axes[0, 1].imshow(gt, cmap=cmap, vmin=vmin, vmax=vmax)
        axes[0, 1].set_title("GT (для ours: глубина сенсора)", fontsize=8)
        for j, (name, deg, kind, preds, mets) in enumerate(results):
            s = samples[deg][fi]
            m = mets[fi]
            col = 2 + j
            pred = preds[fi]
            if args.mask_pred:
                pred = np.where(s["mask"], pred, np.nan)
            axes[0, col].imshow(pred, cmap=cmap, vmin=vmin, vmax=vmax)
            axes[0, col].set_title(
                f"{name} [{deg}]\nAbsRel {m['absrel']:.3f}  RMSE {m['rmse']:.3f}  δ1 {m['delta1']:.3f}"
                + ("" if m["n_valid"] > 0 else "\n(нет GT в окне глубины)"),
                fontsize=8,
            )
            row = 1
            if args.inputs:  # what this very method received, under its prediction
                inp = input_panel(s, kind)
                axes[row, col].imshow(np.clip(s["rgb"], 0, 1) * 0.35)
                axes[row, col].imshow(
                    np.where(inp > 0, inp, np.nan), cmap=overlay, vmin=vmin, vmax=vmax
                )
                axes[row, col].set_title("вход метода", fontsize=7)
                row += 1
            if args.error:
                rel = np.where(
                    s["mask"], (preds[fi] - s["gt_depth"]) / np.maximum(s["gt_depth"], 1e-6), np.nan
                )
                axes[row, col].imshow(rel, cmap=ecmap, vmin=-args.err, vmax=args.err)
                axes[row, col].set_title("(pred−gt)/gt", fontsize=7)
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
            f"{sid}  {tags}"
            + (
                f"\nошибка: красный = дальше, синий = ближе, серый = нет GT, ±{args.err:.0%}"
                if args.error
                else ""
            )
            + ("\nчёрное в GT = нет значения сенсора" if not args.mask_pred else ""),
            fontsize=9,
        )
        fig.tight_layout(rect=(0, 0, 1, 0.94))
        fname = sid.replace("/", "__") + ".png"  # ids like ai_001_003/cam_00/0012 contain slashes
        fig.savefig(out / fname, dpi=80)
        plt.close(fig)
        print(f"  {out / fname}")

    with (out / "metrics.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows_csv[0]))
        w.writeheader()
        w.writerows(rows_csv)
    print(f"[done] {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
