#!/usr/bin/env python3
"""Overlay the two halves of ``plane_edge_mask`` on RGB and GT depth.

``plane_edge_mask`` marks a pixel as an edge if a depth step (``z_jump_mask``) OR a
normal turn (``normal_jump_mask``) fires.  This script plots each half on its own, in a
different colour, so the two thresholds in ``configs/degradation/edge_dropout.yaml``
(``z_epsilon`` and ``normal_thresh``) can be tuned while watching the output.

    uv run python scripts/visualise_edge_jumps.py --dataset zju_l5 -n 3 \
        --config edge_dropout --depth-jump --normal-jump

Three columns per sample: RGB + overlay, GT depth + overlay, mask-only on black.
Red = depth jumps, blue = normal jumps; overlapping pixels become black on the mask
panel.  Edges are computed on the clean GT depth -- no degradation is applied.
``--config`` is the only source of edge parameters (``z_epsilon, z_step,
normal_thresh, normal_step, normal_median, fx, fy, cx, cy, far_side_only``); unused
keys in the YAML are ignored.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import _bootstrap
import numpy as np

from utils.intrinsics import resolve_intrinsics

DEFAULT_OUT_DIR = _bootstrap.REPO_ROOT / "outputs" / "edge_viz"  # gitignored

DEPTH_RGB = (1.0, 0.0, 0.0)  # red
NORMAL_RGB = (30 / 255, 144 / 255, 1.0)  # dodger blue
OVERLAY_ALPHA = 0.50  # transparency on the RGB and depth panels
DILATE_ITERATIONS = 2  # thicken the thin edges so they are visible


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--dataset", required=True, help="dataset config in configs/dataset/")
    p.add_argument(
        "--config",
        required=True,
        help="preset in configs/degradation/ (typically 'edge_dropout')",
    )
    p.add_argument(
        "--normals", default="gt_cross", help="preset in configs/normals/ (default: gt_cross)"
    )
    p.add_argument("--depth-jump", action="store_true", help="overlay z_jump_mask (red)")
    p.add_argument("--normal-jump", action="store_true", help="overlay normal_jump_mask (blue)")
    p.add_argument("-n", type=int, default=3, help="number of samples (default: 3)")
    p.add_argument("-o", "--out", type=Path, default=None, help="output PNG path")
    p.add_argument("--show", action="store_true", help="open a window instead of only saving")
    args = p.parse_args(argv)
    if not (args.depth_jump or args.normal_jump):
        p.error("at least one of --depth-jump / --normal-jump is required")
    if args.n < 1:
        p.error("-n must be >= 1")
    return args


def load_samples(dataset: str, n: int) -> list[dict[str, Any]]:
    from data.degradation import IdentityDegradation
    from data.loaders import build_dataset
    from utils.config import load_config

    ds = build_dataset(
        dict(load_config("dataset", dataset)), degradation=IdentityDegradation(), max_samples=n
    )
    return [ds[i] for i in range(len(ds))]


def _dilated(mask: np.ndarray | None, shape: tuple[int, int]) -> np.ndarray:
    from scipy import ndimage

    if mask is None:
        return np.zeros(shape, dtype=bool)
    return ndimage.binary_dilation(mask, iterations=DILATE_ITERATIONS)


def combined_overlay(
    depth_mask: np.ndarray | None,
    normal_mask: np.ndarray | None,
    shape: tuple[int, int],
    alpha: float,
) -> np.ndarray:
    """RGBA overlay for the RGB and GT-depth panels.

    Depth-only pixels are red, normal-only blue.  Where both fire, the pixels are
    split in a one-pixel checkerboard so both colours stay visible without producing
    a third colour (red + blue additively would read as magenta).
    """
    d_bool = _dilated(depth_mask, shape)
    m_bool = _dilated(normal_mask, shape)

    yy, xx = np.indices(shape)
    checker = ((yy + xx) % 2).astype(bool)
    overlap = d_bool & m_bool
    show_depth = (d_bool & ~m_bool) | (overlap & checker)
    show_normal = (m_bool & ~d_bool) | (overlap & ~checker)

    rgba = np.zeros((*shape, 4), dtype=np.float32)
    rgba[show_depth] = (*DEPTH_RGB, alpha)
    rgba[show_normal] = (*NORMAL_RGB, alpha)
    return rgba


def mask_panel_image(
    depth_mask: np.ndarray | None,
    normal_mask: np.ndarray | None,
    shape: tuple[int, int],
) -> np.ndarray:
    """(H, W, 3) image for the mask-only panel: white background, red / blue / black.

    Depth-only is red, normal-only blue, overlap black.  No transparency: this panel
    exists to show the raw masks without the RGB or depth map behind them.
    """
    d_bool = _dilated(depth_mask, shape)
    m_bool = _dilated(normal_mask, shape)
    only_d = d_bool & ~m_bool
    only_m = m_bool & ~d_bool
    overlap = d_bool & m_bool

    img = np.ones((*shape, 3), dtype=np.float32)
    img[only_d] = DEPTH_RGB
    img[only_m] = NORMAL_RGB
    img[overlap] = (0.0, 0.0, 0.0)
    return img


def plot(samples: list[dict], cfg: dict, args: argparse.Namespace):
    import matplotlib

    if not args.show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from data.norm import build_normal_map
    from data.norm.edge_mask import _median_normals, normal_jump_mask, z_jump_mask
    from utils.config import load_config

    estimator = build_normal_map(dict(load_config("normals", args.normals)))
    if estimator is None:
        raise SystemExit(f"error: normals preset {args.normals!r} is 'none'")

    depth_cmap = matplotlib.colormaps["turbo"].copy()
    depth_cmap.set_bad("0.35")

    fig, axes = plt.subplots(len(samples), 3, figsize=(13.5, 3.4 * len(samples)), squeeze=False)

    for row, s in zip(axes, samples, strict=True):
        rgb = np.clip(s["rgb"], 0.0, 1.0)
        z = s["gt_depth"].astype(np.float64)
        valid = np.isfinite(z) & (z > 0)
        h, w = z.shape

        fx, fy, cx, cy = resolve_intrinsics(cfg, s, h, w)
        normals = estimator(s["gt_depth"], rgb, intrinsics=(fx, fy, cx, cy))["normals"]
        n = _median_normals(normals, int(cfg.get("normal_median", 5)))

        depth_mask = None
        normal_mask = None
        if args.depth_jump:
            depth_mask = valid & z_jump_mask(
                z,
                n,
                valid,
                fx=fx,
                fy=fy,
                cx=cx,
                cy=cy,
                z_epsilon=float(cfg["z_epsilon"]),
                z_step=int(cfg["z_step"]),
                far_side_only=bool(cfg.get("far_side_only", False)),
            )
        if args.normal_jump:
            normal_mask = valid & normal_jump_mask(
                n, float(cfg["normal_thresh"]), int(cfg["normal_step"])
            )

        valid_vals = z[valid]
        vmin, vmax = np.percentile(valid_vals, [1, 99]) if valid_vals.size else (0.0, 1.0)
        depth_img = np.where(valid, z, np.nan)

        overlay = combined_overlay(depth_mask, normal_mask, (h, w), OVERLAY_ALPHA)

        row[0].imshow(rgb, interpolation="nearest")
        row[0].imshow(overlay, interpolation="nearest")
        im = row[1].imshow(
            depth_img, cmap=depth_cmap, vmin=vmin, vmax=vmax, interpolation="nearest"
        )
        fig.colorbar(im, ax=row[1], fraction=0.046, pad=0.02, label="m")
        row[1].imshow(overlay, interpolation="nearest")
        row[2].imshow(mask_panel_image(depth_mask, normal_mask, (h, w)), interpolation="nearest")

        parts: list[str] = []
        if args.depth_jump:
            parts.append(
                f"z_epsilon={cfg['z_epsilon']} z_step={cfg['z_step']}"
                + (" far" if cfg.get("far_side_only") else "")
            )
        if args.normal_jump:
            parts.append(
                f"normal_thresh={cfg['normal_thresh']}° step={cfg['normal_step']}"
                f" median={cfg.get('normal_median', 5)}"
            )
        row[0].set_title(f"RGB [{s['meta']['sample_id']}]", fontsize=9)
        row[1].set_title("GT depth", fontsize=9)
        row[2].set_title(" | ".join(parts), fontsize=9)

        for ax in row:
            ax.set_xticks([])
            ax.set_yticks([])

    labels: list[str] = []
    if args.depth_jump:
        labels.append("red = depth jump")
    if args.normal_jump:
        labels.append("blue = normal jump")
    fig.suptitle(f"{args.dataset} / {args.config}   ({', '.join(labels)})", fontsize=11)
    fig.tight_layout()
    return fig


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)

    from utils.config import load_config

    try:
        cfg = dict(load_config("degradation", args.config))
        samples = load_samples(args.dataset, args.n)
    except (FileNotFoundError, RuntimeError, ValueError, KeyError) as e:
        raise SystemExit(f"error: {e}") from None

    fig = plot(samples, cfg, args)

    halves = []
    if args.depth_jump:
        halves.append("depth")
    if args.normal_jump:
        halves.append("normal")
    stem = f"{args.dataset}_{args.config}_{'_'.join(halves)}_jumps"
    out = args.out or DEFAULT_OUT_DIR / f"{stem}.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=110)
    print(f"saved {out}")
    if args.show:
        import matplotlib.pyplot as plt

        plt.show()
    return 0


if __name__ == "__main__":
    sys.exit(main())
