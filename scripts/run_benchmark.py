#!/usr/bin/env python3
"""Run a matrix of (dataset x method) evaluations and emit one table.

Usage
-----
::

    # everything that can actually run here, on the datasets present on disk
    python scripts/run_benchmark.py --auto

    # an explicit matrix
    python scripts/run_benchmark.py --datasets zju_l5 hammer --models depthor ducos nn_fill

    # a list of experiment files (each one fully specifies its own protocol)
    python scripts/run_benchmark.py --experiments depthor_zju_l5 ducos_rgbdd_real

Outputs
-------
``experiments/results/benchmark.csv``   one appended row per (dataset, method)
``experiments/results/table_<stamp>.md``  the dataset x method x metric table
``experiments/results/table_<stamp>.csv`` the same, machine-readable

A cell that could not be produced is **not** silently dropped: it appears with
the reason (missing weights, missing data, unimplemented loader), because "we
did not run it" and "it scored badly" must never look the same in a report.
"""

from __future__ import annotations

import argparse
import itertools
import time
import traceback
from collections.abc import Sequence
from typing import Any

import _bootstrap  # noqa: F401
import run_baseline

from metrics.depth_metrics import METRIC_NAMES, PRIMARY_METRICS
from utils.config import CONFIG_ROOT
from utils.results import RESULTS_DIR, write_markdown_table


def _names(group: str) -> list[str]:
    return sorted(
        p.stem for p in (CONFIG_ROOT / group).glob("*.yaml") if not p.stem.startswith("_")
    )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--datasets", nargs="*", default=None, help=f"any of {_names('dataset')}")
    p.add_argument("--models", nargs="*", default=None, help=f"any of {_names('model')}")
    p.add_argument("--degradation", default=None, help="one degradation for the whole matrix")
    p.add_argument("--experiments", nargs="*", default=None, help=f"any of {_names('experiment')}")
    p.add_argument("--auto", action="store_true", help="skip cells that cannot run here")
    p.add_argument("--limit", type=int, default=None, help="samples per cell")
    p.add_argument("--device", default=None)
    p.add_argument("--metrics", nargs="*", default=[*PRIMARY_METRICS, "fps"])
    p.add_argument("--dry-run", action="store_true", help="print the matrix and exit")
    return p.parse_args(argv)


def _cell_args(dataset: str, model: str, args: argparse.Namespace) -> list[str]:
    argv = ["--dataset", dataset, "--model", model]
    if args.degradation:
        argv += ["--degradation", args.degradation]
    if args.limit is not None:
        argv += ["--limit", str(args.limit)]
    if args.device:
        argv += ["--device", args.device]
    return argv


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    stamp = time.strftime("%Y%m%d-%H%M%S")

    jobs: list[tuple[str, str, list[str]]] = []
    if args.experiments:
        for exp in args.experiments:
            jobs.append(
                (
                    exp,
                    exp,
                    ["--experiment", exp]
                    + (["--limit", str(args.limit)] if args.limit is not None else []),
                )
            )
    else:
        datasets = args.datasets or ["synthetic_stairs"]
        models = args.models or ["nn_fill", "bicubic"]
        for ds, mdl in itertools.product(datasets, models):
            jobs.append((ds, mdl, _cell_args(ds, mdl, args)))

    print(f"benchmark matrix: {len(jobs)} cell(s)")
    for ds, mdl, _ in jobs:
        print(f"  - {ds} x {mdl}")
    if args.dry_run:
        return 0

    rows: list[dict[str, Any]] = []
    failures: list[tuple[str, str, str]] = []
    for ds, mdl, cell_argv in jobs:
        print(f"\n{'=' * 70}\n>>> {ds} x {mdl}\n{'=' * 70}")
        try:
            code = run_baseline.main(cell_argv)
            if code != 0:
                reason = "unavailable (see the report above)"
                if args.auto:
                    print(f"[skip] {ds} x {mdl}: {reason}")
                failures.append((ds, mdl, reason))
                continue
        except Exception as exc:
            reason = f"{type(exc).__name__}: {exc}".splitlines()[0][:160]
            print(f"[error] {ds} x {mdl}: {reason}")
            if not args.auto:
                traceback.print_exc()
            failures.append((ds, mdl, reason))
            continue

        row = _last_summary_row()
        if row:
            rows.append(row)

    # ---- assemble the table ------------------------------------------------
    wanted = [m for m in args.metrics if m in METRIC_NAMES or m in ("fps", "latency_ms_median")]
    columns = ["dataset", "method", "n_samples", *wanted, "notes"]
    table_rows: list[dict[str, Any]] = []
    for r in rows:
        table_rows.append({c: r.get(c, "") for c in columns})
    for ds, mdl, reason in failures:
        table_rows.append({"dataset": ds, "method": mdl, "notes": f"NOT RUN: {reason}"})

    md = RESULTS_DIR / f"table_{stamp}.md"
    write_markdown_table(table_rows, columns, md)
    csv_path = RESULTS_DIR / f"table_{stamp}.csv"
    _write_csv(table_rows, columns, csv_path)

    print(f"\n{'=' * 70}")
    print(md.read_text(encoding="utf-8"))
    print(f"table written to {md} and {csv_path}")
    if failures:
        print(f"{len(failures)} cell(s) did not run - they are listed in the table as NOT RUN")
    return 0


def _last_summary_row() -> dict[str, Any] | None:
    """Read back the row ``run_baseline`` just appended to the shared CSV."""
    import csv

    from utils.results import SUMMARY_CSV

    if not SUMMARY_CSV.exists():
        return None
    with SUMMARY_CSV.open(newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    if not rows:
        return None
    row = rows[-1]
    for k, v in list(row.items()):
        if k in METRIC_NAMES or k in ("fps", "latency_ms_median"):
            try:
                row[k] = float(v)
            except (TypeError, ValueError):
                row[k] = float("nan")
    return row


def _write_csv(rows: list[dict[str, Any]], columns: list[str], path) -> None:
    import csv

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=columns, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


if __name__ == "__main__":
    raise SystemExit(main())
