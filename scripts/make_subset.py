#!/usr/bin/env python3
"""Cut a small subset out of a dataset and save it to a directory.

Handy for sharing a few frames, debugging on a laptop without the full shared
folder, or looking at samples in a notebook.

    uv run python scripts/make_subset.py --dataset void_stairs --output_dir outputs/subsets/void -n 5
    uv run python scripts/make_subset.py --dataset minjiang --degradation astra2_uncalibrated \
        --output_dir outputs/subsets/minjiang -n 10

Samples go through the loader, so what is saved is the sample contract of
``data/loaders/base.py`` (depth in metres, both input forms), not the raw
upstream files -- the dataset-specific traps (uint16/256, min-max packs, ...)
are already resolved.  Output layout::

    <output_dir>/
        <sample_id>.npz   rgb, gt_depth, sparse_depth, sparse_mask, lr_depth, mask
        <sample_id>.json  meta
        split.json        the chosen ids, in the data/splits/ format
        manifest.json     dataset, degradation, seed, uncalibrated flag

``split.json`` reproduces the same subset from the full dataset with
``dataset.split_file=<output_dir>/split.json``.  Keep the output out of git
(``outputs/`` is ignored).
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path

import _bootstrap  # noqa: F401
import numpy as np

ARRAY_KEYS = ("rgb", "gt_depth", "sparse_depth", "sparse_mask", "lr_depth", "mask")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--dataset", required=True, help="dataset config in configs/dataset/")
    p.add_argument("--output_dir", required=True, type=Path, help="where to write the subset")
    p.add_argument("-n", type=int, default=5, help="number of samples (default: 5)")
    p.add_argument(
        "--degradation",
        default=None,
        help="degradation preset in configs/degradation/ (needed for GT-only datasets)",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=0,
        help="seed for picking samples; -1 takes the first n in index order (default: 0)",
    )
    p.add_argument("--overwrite", action="store_true", help="allow a non-empty output_dir")
    args = p.parse_args(argv)
    if args.n < 1:
        p.error("-n must be >= 1")
    return args


def safe_name(sample_id: str) -> str:
    """Sample ids look like ``stairs0/000123``; flatten them into a file name."""
    return sample_id.replace("/", "__").replace("\\", "__")


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)

    from data.degradation import build_degradation
    from data.loaders import build_dataset
    from utils.config import load_config

    out: Path = args.output_dir
    if out.exists() and any(out.iterdir()) and not args.overwrite:
        print(f"error: {out} is not empty; pass --overwrite to write into it", file=sys.stderr)
        return 1
    out.mkdir(parents=True, exist_ok=True)

    deg = None
    if args.degradation is not None:
        deg = build_degradation(dict(load_config("degradation", args.degradation)))
    ds = build_dataset(dict(load_config("dataset", args.dataset)), degradation=deg)

    uncalibrated = False
    if deg is not None and ds.info.modality != "gt_only":
        print(
            f"note: {args.dataset} ships a real sensor input; "
            f"--degradation {args.degradation} is ignored",
            file=sys.stderr,
        )
    elif deg is not None:
        # the loader drops the degradation's meta, so ask the model directly
        probe = deg(np.ones((4, 4), dtype=np.float32), seed=0)["meta"]
        uncalibrated = probe.get("calibrated") is False

    n = min(args.n, len(ds))
    if n < args.n:
        print(f"note: {args.dataset} has only {len(ds)} samples, taking all", file=sys.stderr)
    if args.seed < 0:
        indices = list(range(n))
    else:
        # sorted, so the subset reads in the dataset's own order
        indices = sorted(np.random.default_rng(args.seed).choice(len(ds), n, replace=False))

    ids = []
    for idx in indices:
        sample = ds[int(idx)]
        sid = sample["meta"]["sample_id"]
        # the base loader falls back to the index when a loader sets no id;
        # the split file needs the record id, which is what split_file matches
        rid = ds.record_id(ds.records[int(idx)])
        name = safe_name(rid)
        np.savez_compressed(out / f"{name}.npz", **{k: sample[k] for k in ARRAY_KEYS})
        (out / f"{name}.json").write_text(
            json.dumps(sample["meta"], indent=1, default=str), encoding="utf-8"
        )
        ids.append(rid)
        gt = sample["gt_depth"]
        valid = gt[sample["mask"]]
        rng_txt = f"{valid.min():.2f}-{valid.max():.2f} m" if valid.size else "no valid GT"
        print(f"  {sid:40s} {gt.shape}  depth {rng_txt}")

    ds.write_split_file(out / "split.json", ids)
    manifest = {
        "dataset": args.dataset,
        "split": ds.split,
        "n": len(ids),
        "seed": args.seed,
        "degradation": args.degradation if ds.info.modality == "gt_only" else None,
        "notes": "uncalibrated" if uncalibrated else "",
        "ids": ids,
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1), encoding="utf-8")

    print(f"\nwrote {len(ids)} samples of {args.dataset} to {out}")
    if uncalibrated:
        print("note: input comes from an uncalibrated sensor model (marked in manifest.json)")
    if args.dataset == "synthetic_stairs":
        print("note: synthetic_stairs is for CI only -- no number from it is reportable")
    return 0


if __name__ == "__main__":
    sys.exit(main())
