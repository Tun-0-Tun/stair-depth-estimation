#!/usr/bin/env python3
"""Check the shared dataset folder is laid out the way the loaders expect.

Datasets are big and shared between several people, so they live in one
external folder that everyone points ``STAIR_DATA_ROOT`` at.  Nothing here
downloads anything -- it tells you whether what you have is in the right shape,
and names the exact missing path when it is not.

    export STAIR_DATA_ROOT=/path/to/shared/data
    python scripts/check_data.py              # check everything
    python scripts/check_data.py void_stairs  # check one
    python scripts/check_data.py --load       # also open one sample per dataset

Layouts and download links: docs/datasets.md
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence

import _bootstrap  # noqa: F401

from utils.misc import DATA_ROOT_ENV, data_root

#: dataset key -> (root dir under STAIR_DATA_ROOT, paths that must exist inside it)
EXPECTED: dict[str, tuple[str, tuple[str, ...]]] = {
    "void_stairs": (
        "VOID",
        (
            "data/stairs0/image",
            "data/stairs0/sparse_depth",
            "data/stairs0/ground_truth",
            "data/stairs0/K.txt",
        ),
    ),
    "minjiang": ("MinJiang-Dataset", ("STAIRS/CAM1_RGB", "STAIRS/CAM1_GDEP")),
    # layout='mat' (the default). The h5 pack lives in nyudepthv2/val, the DSR
    # npy pack in test_images_v2.npy -- check those by hand if you switch.
    "nyuv2": ("nyu_depth_v2", ("nyu_depth_v2_labeled.mat",)),
    "zju_l5": ("ZJUL5", ("data.json",)),
    "hammer": ("HAMMER", ()),
}


def check(key: str, load: bool = False) -> bool:
    rel, expect = EXPECTED[key]
    root = data_root() / rel
    ok = root.is_dir()
    print(f"[{'OK ' if ok else 'MISSING'}] {key:14s} {root}")
    if not ok:
        return False

    for e in expect:
        present = (root / e).exists()
        ok &= present
        print(f"           {'+' if present else '-'} {e}")

    if ok and load:
        try:
            from data.degradation import build_degradation
            from data.loaders import build_dataset, validate_sample
            from utils.config import load_config

            # GT-only datasets need a sensor model to produce an input; the
            # rest ignore it. Any model will do here, we only check the layout.
            deg = build_degradation(dict(load_config("degradation", "astra2_uncalibrated")))
            ds = build_dataset(dict(load_config("dataset", key)), degradation=deg, max_samples=1)
            validate_sample(ds[0], key)
            gt = ds[0]["gt_depth"]
            valid = gt[gt > 0]
            print(
                f"           loaded 1 sample: {gt.shape}, "
                f"depth {valid.min():.2f}-{valid.max():.2f} m, "
                f"sparsity {ds[0]['meta']['sparsity']:.3f}"
            )
        except Exception as exc:
            print(f"           LOAD FAILED: {type(exc).__name__}: {str(exc).splitlines()[0]}")
            ok = False
    return ok


def main(argv: Sequence[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("datasets", nargs="*", default=None, help=f"any of {sorted(EXPECTED)}")
    p.add_argument("--load", action="store_true", help="also open one sample per dataset")
    args = p.parse_args(argv)

    print(f"{DATA_ROOT_ENV} = {data_root()}\n")
    keys = args.datasets or sorted(EXPECTED)
    unknown = [k for k in keys if k not in EXPECTED]
    if unknown:
        p.error(f"unknown dataset(s) {unknown}; expected {sorted(EXPECTED)}")

    results = {k: check(k, args.load) for k in keys}
    missing = [k for k, v in results.items() if not v]
    print(
        f"\n{len(results) - len(missing)}/{len(results)} ready"
        + (f"; missing or broken: {missing}" if missing else "")
    )
    print("Layouts and download links: docs/datasets.md")
    return 1 if missing else 0


if __name__ == "__main__":
    sys.exit(main())
