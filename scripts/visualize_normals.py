#!/usr/bin/env python3
"""Plot normal maps built from ground-truth depth, sample by sample.

One row per sample.  Two layouts, chosen by whether the samples carry reference normals
(``sample["gt_normals"]``, which no loader provides yet):

    no reference normals:  RGB | GT depth | normals predicted from GT depth
    reference normals:     RGB | GT depth | GT normals | predicted normals | angular error

    uv run --group notebooks python scripts/visualize_normals.py \
        --dataset synthetic_stairs --normals gt_cross -n 5

``--normals`` names a preset in ``configs/normals/``.  Normals are predicted from GT depth
alone: no sensor degradation is applied and none can be chosen.  (Loaders of GT-only datasets
refuse to build without *some* degradation object, so the script hands them ``identity`` --
a pass-through that only fills the unused input fields; ``gt_depth`` is never touched.)
Intrinsics come from ``sample["meta"]["intrinsics"]`` when the loader provides them,
otherwise from the preset -- those are placeholders, so the title says ``uncalibrated``.
matplotlib lives in the ``notebooks`` dependency group.

Colour code: n = (x right, y down, z forward) -> RGB = (n + 1) / 2. A surface facing the
camera (0, 0, -1) is olive (0.5, 0.5, 0); a tread/floor (normal up = -y, tilted toward the
camera) shifts toward red-orange, and a surface tilted right/left shifts the red channel.
Invalid pixels (holes, depth edges, image border) are grey.  This is a per-pixel picture only;
no aggregate normal-quality numbers are computed here.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import _bootstrap
import numpy as np

DEFAULT_OUT_DIR = _bootstrap.REPO_ROOT / "outputs" / "normals_viz"  # gitignored


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--dataset", required=True, help="dataset config in configs/dataset/")
    p.add_argument(
        "--normals", default="gt_cross", help="preset in configs/normals/ (default: gt_cross)"
    )
    p.add_argument("-n", type=int, default=5, help="number of samples (default: 5)")
    p.add_argument("-o", "--out", type=Path, default=None, help="output PNG path")
    p.add_argument("--show", action="store_true", help="open a window instead of only saving")
    return p.parse_args(argv)


def load_samples(dataset: str, n: int) -> list[dict[str, Any]]:
    from data.degradation import IdentityDegradation
    from data.loaders import build_dataset
    from utils.config import load_config

    # no degradation of GT: identity only satisfies the GT-only loaders' constructor check
    ds = build_dataset(
        dict(load_config("dataset", dataset)), degradation=IdentityDegradation(), max_samples=n
    )
    return [ds[i] for i in range(len(ds))]


def normals_to_rgb(normals: np.ndarray, mask: np.ndarray, invalid: float = 0.35) -> np.ndarray:
    """Encode unit normals as RGB in [0, 1]; invalid pixels become grey."""
    img = np.clip((normals + 1.0) * 0.5, 0.0, 1.0)
    return np.where(mask[..., None], img, invalid).astype(np.float32)


def intrinsics_of(sample: dict[str, Any]) -> tuple[float, ...] | None:
    k = sample["meta"].get("intrinsics")
    return None if k is None else tuple(float(v) for v in k)


def gt_normals_of(sample: dict[str, Any]) -> np.ndarray | None:
    """Dataset-provided normals (``sample["gt_normals"]``, (H, W, 3) unit, camera frame), if any.

    No loader ships them yet; this is the hook for one that does.  Not the same thing as
    ``gt_depth``: these are an independent reference, ``pred`` below is built from the depth.
    """
    n = sample.get("gt_normals")
    return None if n is None else np.asarray(n, dtype=np.float32)


def angular_error_deg(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Per-pixel angle between two unit-normal maps, degrees (a picture, not a metric)."""
    cos = np.clip(np.sum(a * b, axis=-1), -1.0, 1.0)
    return np.degrees(np.arccos(cos))


def plot(samples: list[dict], estimator, args: argparse.Namespace):
    """Without reference normals: RGB | GT depth | predicted normals.

    With them: RGB | GT depth | GT normals | predicted normals | angular error.
    """
    import matplotlib

    if not args.show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    has_ref = all(gt_normals_of(s) is not None for s in samples)
    n_cols = 5 if has_ref else 3
    fig, axes = plt.subplots(
        len(samples), n_cols, figsize=(4.2 * n_cols, 3.4 * len(samples)), squeeze=False
    )
    depth_cmap = matplotlib.colormaps["turbo"].copy()
    depth_cmap.set_bad("0.35")
    err_cmap = matplotlib.colormaps["magma"].copy()
    err_cmap.set_bad("0.35")

    uncalibrated = False
    for row, s in zip(axes, samples, strict=True):
        gt, rgb = s["gt_depth"], np.clip(s["rgb"], 0.0, 1.0)
        out = estimator(gt, rgb, intrinsics=intrinsics_of(s))
        uncalibrated |= out["meta"]["intrinsics_source"] != "call"
        # judge normals only where the metrics would judge depth too
        mask = out["normal_mask"] & s["mask"]
        pred_img = normals_to_rgb(out["normals"], mask)

        valid = gt[gt > 0]
        vmin, vmax = np.percentile(valid, [1, 99]) if valid.size else (0.0, 1.0)

        row[0].imshow(rgb, interpolation="nearest")
        row[0].set_title(f"RGB [{s['meta']['sample_id']}]", fontsize=9)
        im = row[1].imshow(
            np.where(gt > 0, gt, np.nan), cmap=depth_cmap, vmin=vmin, vmax=vmax,
            interpolation="nearest",
        )  # fmt: skip
        fig.colorbar(im, ax=row[1], fraction=0.046, pad=0.02, label="m")
        row[1].set_title("GT depth", fontsize=9)

        if has_ref:
            ref = gt_normals_of(s)
            ref_valid = mask & (np.linalg.norm(ref, axis=-1) > 0.5)
            row[2].imshow(normals_to_rgb(ref, ref_valid), interpolation="nearest")
            row[2].set_title("GT normals", fontsize=9)
            row[3].imshow(pred_img, interpolation="nearest")
            row[3].set_title(f"predicted normals, valid {100 * mask.mean():.1f}%", fontsize=9)
            both = mask & ref_valid
            err = np.where(both, angular_error_deg(out["normals"], ref), np.nan)
            emax = float(np.nanpercentile(err, 99)) if both.any() else 0.0
            im = row[4].imshow(
                err, cmap=err_cmap, vmin=0.0, vmax=max(emax, 1.0), interpolation="nearest"
            )
            fig.colorbar(im, ax=row[4], fraction=0.046, pad=0.02, label="deg")
            row[4].set_title("angular error (grey = no pair)", fontsize=9)
        else:
            row[2].imshow(pred_img, interpolation="nearest")
            row[2].set_title(f"normals from GT depth, valid {100 * mask.mean():.1f}%", fontsize=9)

        for ax in row:
            ax.set_xticks([])
            ax.set_yticks([])

    fig.suptitle(
        f"{args.dataset} / {args.normals}" + ("  (uncalibrated)" if uncalibrated else ""),
        fontsize=11,
    )
    fig.tight_layout()
    return fig


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.n < 1:
        raise SystemExit("-n must be >= 1")

    from data.norm import build_normal_map
    from utils.config import load_config

    try:
        estimator = build_normal_map(dict(load_config("normals", args.normals)))
        if estimator is None:
            raise SystemExit(f"error: normals preset {args.normals!r} is 'none'")
        samples = load_samples(args.dataset, args.n)
    except (FileNotFoundError, RuntimeError, ValueError, KeyError) as e:
        raise SystemExit(f"error: {e}") from None
    fig = plot(samples, estimator, args)

    out = args.out or DEFAULT_OUT_DIR / f"{args.dataset}_{args.normals}.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=110)
    print(f"saved {out}")
    if args.show:
        import matplotlib.pyplot as plt

        plt.show()
    return 0


if __name__ == "__main__":
    sys.exit(main())
