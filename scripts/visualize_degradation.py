#!/usr/bin/env python3
"""Plot what a degradation does to ground-truth depth, sample by sample.

One row per sample: GT | degraded input | error (degraded - GT) | [model prediction].

    uv run --group notebooks python scripts/visualize_degradation.py \
        --dataset synthetic_stairs --degradation projector_shadow -n 5

``--degradation`` names a preset in ``configs/degradation/``.  Datasets that ship
a real sensor input (void_stairs, zju_l5, hammer, ...) ignore it and show their
native input.  matplotlib lives in the ``notebooks`` dependency group.

The error panel is a per-pixel picture only.  Aggregate numbers (RMSE, AbsRel,
...) come from ``metrics/depth_metrics.py`` and nowhere else.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import _bootstrap
import numpy as np

from utils.misc import resize_depth

DEFAULT_OUT_DIR = _bootstrap.REPO_ROOT / "outputs" / "degradation_viz"  # gitignored


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--dataset", required=True, help="dataset config in configs/dataset/")
    p.add_argument(
        "--degradation",
        default=None,
        help="degradation preset in configs/degradation/ (needed for GT-only datasets)",
    )
    p.add_argument("-n", type=int, default=5, help="number of samples (default: 5)")
    p.add_argument(
        "--model", default=None, help="model whose prediction to add (NOT IMPLEMENTED YET)"
    )
    p.add_argument(
        "--input",
        choices=("sparse", "lr"),
        default="sparse",
        help="which input to show: sparse_depth (completion) or lr_depth (SR)",
    )
    p.add_argument("-o", "--out", type=Path, default=None, help="output PNG path")
    p.add_argument("--show", action="store_true", help="open a window instead of only saving")
    return p.parse_args(argv)


def load_samples(dataset: str, degradation: str | None, n: int) -> tuple[list[dict], bool]:
    """Return the first ``n`` samples and whether their input is uncalibrated."""
    from data.degradation import build_degradation
    from data.loaders import build_dataset
    from utils.config import load_config

    deg = None
    if degradation is not None:
        deg = build_degradation(dict(load_config("degradation", degradation)))
    ds = build_dataset(dict(load_config("dataset", dataset)), degradation=deg, max_samples=n)

    uncalibrated = False
    if deg is not None and ds.info.modality != "gt_only":
        print(
            f"note: {dataset} ships a real sensor input; --degradation {degradation} is ignored",
            file=sys.stderr,
        )
    elif deg is not None:
        # the loader drops the degradation's meta, so ask the model directly
        probe = deg(np.ones((4, 4), dtype=np.float32), seed=0)["meta"]
        uncalibrated = probe.get("calibrated") is False
    return [ds[i] for i in range(len(ds))], uncalibrated


def predict(model: str, sample: dict[str, Any]) -> np.ndarray:
    """Dense (H, W) prediction in metres.  Hook for later -- not implemented yet."""
    raise NotImplementedError(f"--model {model}: model prediction is not implemented yet")


def degraded_input(sample: dict[str, Any], which: str) -> tuple[np.ndarray, np.ndarray]:
    """Return (input as stored, input resized to the GT grid)."""
    x = sample["sparse_depth"] if which == "sparse" else sample["lr_depth"]
    # nearest: interpolation would invent depths between surfaces and smear holes
    return x, resize_depth(x, sample["gt_depth"].shape, mode="nearest")


def _holes_as_nan(depth: np.ndarray) -> np.ndarray:
    return np.where(depth > 0, depth, np.nan)


def plot(samples: list[dict], args: argparse.Namespace, uncalibrated: bool = False):
    import matplotlib

    if not args.show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n_cols = 4 if args.model else 3
    fig, axes = plt.subplots(
        len(samples), n_cols, figsize=(4.2 * n_cols, 3.4 * len(samples)), squeeze=False
    )
    depth_cmap = matplotlib.colormaps["turbo"].copy()
    depth_cmap.set_bad("0.35")  # holes / no measurement: grey
    err_cmap = matplotlib.colormaps["RdBu_r"].copy()
    err_cmap.set_bad("0.35")

    for row, s in zip(axes, samples, strict=True):
        gt = s["gt_depth"]
        shown, resized = degraded_input(s, args.input)
        # one depth scale per row, so GT and input are directly comparable
        valid_gt = gt[gt > 0]
        vmin, vmax = np.percentile(valid_gt, [1, 99]) if valid_gt.size else (0.0, 1.0)

        both = (gt > 0) & (resized > 0)
        err = np.where(both, resized - gt, np.nan)
        emax = float(np.nanpercentile(np.abs(err), 99)) if both.any() else 0.0
        emax = max(emax, 1e-3)

        sid = s["meta"]["sample_id"]
        panels = [
            (_holes_as_nan(gt), depth_cmap, (vmin, vmax), f"GT [{sid}]", "m"),
            (
                _holes_as_nan(shown),
                depth_cmap,
                (vmin, vmax),
                f"{args.input} input {shown.shape[1]}x{shown.shape[0]}, "
                f"holes {100 * (1 - (shown > 0).mean()):.1f}%",
                "m",
            ),
            (err, err_cmap, (-emax, emax), "error: input - GT (grey = no pair)", "m"),
        ]
        for ax, (img, cmap, (lo, hi), title, unit) in zip(row[:3], panels, strict=True):
            im = ax.imshow(img, cmap=cmap, vmin=lo, vmax=hi, interpolation="nearest")
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02, label=unit)
            ax.set_title(title, fontsize=9)

        if args.model:
            ax = row[3]
            try:
                pred = predict(args.model, s)
                im = ax.imshow(pred, cmap=depth_cmap, vmin=vmin, vmax=vmax)
                fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02, label="m")
                ax.set_title(f"prediction: {args.model}", fontsize=9)
            except NotImplementedError:
                ax.text(
                    0.5,
                    0.5,
                    "model prediction\nnot implemented yet",
                    ha="center",
                    va="center",
                    transform=ax.transAxes,
                )
                ax.set_title(f"prediction: {args.model}", fontsize=9)

        for ax in row:
            ax.set_xticks([])
            ax.set_yticks([])

    fig.suptitle(
        f"{args.dataset} / {args.degradation or 'native input'}"
        + ("  (uncalibrated)" if uncalibrated else ""),
        fontsize=11,
    )
    fig.tight_layout()
    return fig


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.n < 1:
        raise SystemExit("-n must be >= 1")
    if args.model:
        print(f"note: --model {args.model}: prediction is not implemented yet", file=sys.stderr)

    try:
        samples, uncalibrated = load_samples(args.dataset, args.degradation, args.n)
    except (FileNotFoundError, RuntimeError, ValueError, KeyError) as e:
        # the loaders' messages already say what to fix (missing data, no --degradation)
        raise SystemExit(f"error: {e}") from None
    fig = plot(samples, args, uncalibrated)

    out = args.out or DEFAULT_OUT_DIR / f"{args.dataset}_{args.degradation or 'native'}.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=110)
    print(f"saved {out}")
    if args.show:
        import matplotlib.pyplot as plt

        plt.show()
    return 0


if __name__ == "__main__":
    sys.exit(main())
