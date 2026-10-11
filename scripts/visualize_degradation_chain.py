#!/usr/bin/env python3
"""Plot a chain of degradations applied one after another to ground-truth depth.

One row per sample: RGB | normal map | GT | after step 1 | ... | after step N | error.
The normal map column appears only if at least one step needs normals; the error panel
is the final result minus GT.

    uv run --group notebooks python scripts/visualize_degradation_chain.py \
        --dataset synthetic_stairs -n 3
    uv run --group notebooks python scripts/visualize_degradation_chain.py \
        --dataset synthetic_stairs --degradations projector_shadow axial_noise

``--degradations`` names presets in ``configs/degradation/``; the order given is the order
of application (default: lateral_noise projector_shadow axial_noise).  Each step takes the
``sparse_depth`` of the previous one as its input.  The normal map is resolved once, from the
clean GT (the dataset's own if it has one, else ``--normals``), and shared by all steps.

Always starts from clean GT: a native sensor input of the dataset is not used.
matplotlib lives in the ``notebooks`` dependency group.  Pictures only: aggregate numbers
come from ``metrics/depth_metrics.py``.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import _bootstrap
import numpy as np
from visualize_normals import PALETTES, normals_to_rgb

from utils.misc import resize_depth

DEFAULT_OUT_DIR = _bootstrap.REPO_ROOT / "outputs" / "degradation_viz"  # gitignored
DEFAULT_CHAIN = ("lateral_noise", "projector_shadow", "axial_noise")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--dataset", required=True, help="dataset config in configs/dataset/")
    p.add_argument(
        "--degradations",
        nargs="+",
        default=list(DEFAULT_CHAIN),
        metavar="NAME",
        help="presets in configs/degradation/, applied in this order "
        f"(default: {' '.join(DEFAULT_CHAIN)})",
    )
    p.add_argument(
        "--normals",
        default="gt_cross",
        help="normal-map method in configs/normals/ used when the dataset has no normals "
        "(default: gt_cross)",
    )
    p.add_argument(
        "--force-normals",
        action="store_true",
        help="compute normals with --normals even if the dataset provides them",
    )
    p.add_argument(
        "--palette",
        choices=PALETTES,
        default="standard",
        help="colour coding of the normal map (default: standard)",
    )
    p.add_argument("-n", type=int, default=3, help="number of samples (default: 3)")
    p.add_argument("--seed", type=int, default=0, help="base seed (default: 0)")
    p.add_argument("-o", "--out", type=Path, default=None, help="output PNG path")
    p.add_argument("--show", action="store_true", help="open a window instead of only saving")
    return p.parse_args(argv)


def run_chain(args: argparse.Namespace) -> tuple[list[dict[str, Any]], bool]:
    """Return one record per sample (rgb, gt, steps, normals) and whether normals are shown."""
    from data.degradation import IdentityDegradation, build_degradation
    from data.loaders import build_dataset
    from data.norm import build_normal_map
    from utils.config import load_config

    chain = [
        (name, build_degradation(dict(load_config("degradation", name))))
        for name in args.degradations
    ]
    for name, deg in chain:
        if deg is None:
            raise SystemExit(f"error: degradation {name!r} is a no-op ('none'); nothing to chain")
    need_normals = any(getattr(d, "uses_normals", False) for _, d in chain)
    estimator = (
        build_normal_map(dict(load_config("normals", args.normals))) if need_normals else None
    )

    # clean GT only: identity merely satisfies the GT-only loaders' constructor check
    ds = build_dataset(
        dict(load_config("dataset", args.dataset)),
        degradation=IdentityDegradation(),
        force_degradation=True,
        max_samples=args.n,
    )

    records = []
    for i in range(len(ds)):
        s = ds[i]
        gt = s["gt_depth"]

        normals = None
        if need_normals:
            native = None if args.force_normals else ds._load_raw(ds.records[i]).get("normal")
            if native is not None:
                normals = np.asarray(native, dtype=np.float32)
                if normals.shape[:2] != gt.shape:  # loader resized GT (target_size)
                    normals = np.stack(
                        [resize_depth(normals[..., c], gt.shape, mode="nearest") for c in range(3)],
                        -1,
                    )
            else:
                normals = estimator(gt)["normals"]

        steps = [{"name": "GT", "depth": gt, "calibrated": True}]
        prev = gt
        for k, (name, deg) in enumerate(chain):
            kwargs = {"normal": normals} if getattr(deg, "uses_normals", False) else {}
            # distinct seed per step, so steps do not share an RNG stream
            out = deg(prev, seed=args.seed * 1_000_003 + i * 101 + k, **kwargs)
            prev = out["sparse_depth"]
            steps.append(
                {
                    "name": name,
                    "depth": prev,
                    "calibrated": out["meta"].get("calibrated") is not False,
                }
            )
        records.append(
            {
                "id": s["meta"]["sample_id"],
                "rgb": s["rgb"],
                "gt": gt,
                "steps": steps,
                "normals": normals,
            }
        )
    return records, need_normals


def plot(records: list[dict[str, Any]], with_normals: bool, args: argparse.Namespace):
    import matplotlib

    if not args.show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # RGB | [normals] | GT | step 1..N | error (last step - GT)
    n_steps = len(records[0]["steps"])
    n_cols = 1 + int(with_normals) + n_steps + 1
    fig, axes = plt.subplots(
        len(records), n_cols, figsize=(3.6 * n_cols, 3.0 * len(records)), squeeze=False
    )
    cmap = matplotlib.colormaps["turbo"].copy()
    cmap.set_bad("0.35")  # holes / no measurement: grey
    err_cmap = matplotlib.colormaps["RdBu_r"].copy()
    err_cmap.set_bad("0.35")

    for row, r in zip(axes, records, strict=True):
        gt = r["gt"]
        valid_gt = gt[gt > 0]
        vmin, vmax = np.percentile(valid_gt, [1, 99]) if valid_gt.size else (0.0, 1.0)

        row[0].imshow(np.clip(r["rgb"], 0.0, 1.0), interpolation="nearest")
        row[0].set_title(f"RGB [{r['id']}]", fontsize=9)

        first_depth = 1
        if with_normals:
            first_depth = 2
            mask = (np.linalg.norm(r["normals"], axis=-1) > 1e-6) & (gt > 0)
            row[1].imshow(normals_to_rgb(r["normals"], mask, args.palette), interpolation="nearest")
            row[1].set_title("normal map (clean GT)", fontsize=9)

        for k, (ax, st) in enumerate(
            zip(row[first_depth : first_depth + n_steps], r["steps"], strict=True)
        ):
            d = st["depth"]
            im = ax.imshow(
                np.where(d > 0, d, np.nan),
                cmap=cmap,
                vmin=vmin,
                vmax=vmax,
                interpolation="nearest",
            )
            title = st["name"] if k == 0 else f"{k}. {st['name']}"
            ax.set_title(f"{title}, holes {100 * (1 - (d > 0).mean()):.1f}%", fontsize=9)
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02, label="m")

        final = r["steps"][-1]["depth"]
        both = (gt > 0) & (final > 0)
        err = np.where(both, final - gt, np.nan)
        emax = max(float(np.nanpercentile(np.abs(err), 99)) if both.any() else 0.0, 1e-3)
        im = row[-1].imshow(err, cmap=err_cmap, vmin=-emax, vmax=emax, interpolation="nearest")
        fig.colorbar(im, ax=row[-1], fraction=0.046, pad=0.02, label="m")
        row[-1].set_title("error: final - GT (grey = no pair)", fontsize=9)

        for ax in row:
            ax.set_xticks([])
            ax.set_yticks([])

    uncal = any(not st["calibrated"] for r in records for st in r["steps"])
    fig.suptitle(
        f"{args.dataset}: "
        + " -> ".join(args.degradations)
        + ("  (uncalibrated)" if uncal else ""),
        fontsize=11,
    )
    fig.tight_layout()
    return fig


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.n < 1:
        raise SystemExit("-n must be >= 1")

    try:
        records, with_normals = run_chain(args)
    except (FileNotFoundError, RuntimeError, ValueError, KeyError) as e:
        raise SystemExit(f"error: {e}") from None
    fig = plot(records, with_normals, args)

    out = args.out or DEFAULT_OUT_DIR / f"{args.dataset}_chain_{'+'.join(args.degradations)}.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=110)
    print(f"saved {out}")
    if args.show:
        import matplotlib.pyplot as plt

        plt.show()
    return 0


if __name__ == "__main__":
    sys.exit(main())
