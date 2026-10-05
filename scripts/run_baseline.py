#!/usr/bin/env python3
"""Run one method on one dataset and print / record the metrics.

Usage
-----
Hydra-style (preferred -- an experiment file is one reproducible row)::

    python scripts/run_baseline.py experiment=smoke
    python scripts/run_baseline.py experiment=depthor_zju_l5 eval.max_samples=20
    python scripts/run_baseline.py model=ducos dataset=rgbdd degradation=none

Argparse-style, for quick interactive work::

    python scripts/run_baseline.py --model depthor --dataset nyuv2 --limit 10

Check whether a baseline can run here at all, without loading anything::

    python scripts/run_baseline.py --model depthor --dataset zju_l5 --check

Every run appends one row to ``experiments/results/benchmark.csv`` (tracked in
git) and writes the full config, environment and per-sample metrics to
``experiments/runs/<timestamp>_<experiment>/``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections.abc import Sequence
from typing import Any

import _bootstrap  # noqa: F401  (must come first: puts the repo root on sys.path)

from baselines.base import BASELINE_REGISTRY, build_baseline, model_depth_range
from data.degradation import build_degradation
from data.loaders import build_dataset
from metrics.depth_metrics import METRIC_NAMES, DepthMetricAccumulator, MetricConfig, format_metrics
from metrics.runtime_metrics import RuntimeConfig, measure_runtime
from utils.config import CONFIG_ROOT, Config, _read_yaml, apply_overrides, load_experiment
from utils.misc import set_seed
from utils.results import RunRecorder, find_summary_row, get_tracker

#: The benchmark.csv row of the last main() call -- run, or found in the cache.
#: run_benchmark reads it instead of "the last line of the CSV", which another
#: run on another GPU may have appended in the meantime.
LAST_ROW: dict[str, Any] | None = None


def config_hash(cfg: Config) -> str:
    """What decides the numbers: data, input, method, weights, protocol, seed.

    Not the device, runtime or notes, and not the experiment's name -- so the
    same floor reached from two experiment files is computed once. Not the code
    either: after a fix that changes results, rerun with --force.
    """

    def clean(x: Any) -> Any:
        if isinstance(x, dict):
            return {
                k: clean(v) for k, v in x.items() if v is not None and k not in ("docs", "notes")
            }
        return x

    d = cfg.to_dict()
    blob = {k: clean(d.get(k)) for k in ("dataset", "degradation", "model", "eval", "seed")}
    return hashlib.sha1(json.dumps(blob, sort_keys=True, default=str).encode()).hexdigest()[:12]


def parse_args(argv: Sequence[str]) -> tuple[argparse.Namespace, list[str]]:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--model", help="model config name (configs/model/*.yaml)")
    p.add_argument("--dataset", help="dataset config name (configs/dataset/*.yaml)")
    p.add_argument("--degradation", help="degradation config name (configs/degradation/*.yaml)")
    p.add_argument("--experiment", help="experiment config name (configs/experiment/*.yaml)")
    p.add_argument("--limit", type=int, help="evaluate at most N samples")
    p.add_argument("--device", help="cpu | cuda | mps | auto")
    p.add_argument("--check", action="store_true", help="only report whether the model can run")
    p.add_argument("--no-runtime", action="store_true", help="skip the FPS measurement")
    p.add_argument("--notes", default="", help="free text stored in the results row")
    p.add_argument(
        "--force", action="store_true", help="rerun even if benchmark.csv has this exact config"
    )
    args, rest = p.parse_known_args(argv)
    overrides = [a for a in rest if "=" in a and not a.startswith("-")]
    unknown = [a for a in rest if a not in overrides]
    if unknown:
        p.error(f"unrecognised arguments: {unknown}")
    return args, overrides


def compose(args: argparse.Namespace, overrides: list[str]) -> Config:
    """Build the config from either the experiment file or bare group choices."""
    named = {
        o.split("=", 1)[0]: o.split("=", 1)[1] for o in overrides if "." not in o.split("=")[0]
    }
    experiment = args.experiment or named.pop("experiment", None)

    # A group override (`dataset=void_stairs`) REPLACES that whole group, so it
    # has to be applied before any key override inside it (`dataset.root=...`),
    # or the key override is silently discarded.
    group_flags = [
        f"{g}={getattr(args, g)}" for g in ("model", "dataset", "degradation") if getattr(args, g)
    ]
    rest = [o for o in overrides if not o.startswith("experiment=")]

    if experiment:
        return load_experiment(experiment, [*group_flags, *rest])

    if not (args.model or any(o.startswith("model=") for o in rest)):
        raise SystemExit(
            "give either --experiment/experiment=<name> or at least --model and --dataset.\n"
            f"models: {sorted(p.stem for p in (CONFIG_ROOT / 'model').glob('*.yaml'))}\n"
            f"datasets: {sorted(p.stem for p in (CONFIG_ROOT / 'dataset').glob('*.yaml'))}"
        )
    base = _read_yaml(CONFIG_ROOT / "experiment" / "_base.yaml")
    base.setdefault("degradation", {"type": "none"})
    base["experiment_name"] = "adhoc"
    return Config(apply_overrides(base, [*group_flags, *rest]))


def main(argv: Sequence[str] | None = None) -> int:
    args, overrides = parse_args(list(sys.argv[1:] if argv is None else argv))
    cfg = compose(args, overrides)

    if args.device:
        cfg.device = args.device
    if args.limit is not None:
        cfg.eval["max_samples"] = args.limit
    if args.no_runtime:
        cfg.runtime["measure"] = False

    model_cfg = dict(cfg.model)
    dataset_cfg = dict(cfg.dataset)
    experiment_name = cfg.get("experiment_name", "adhoc")

    global LAST_ROW
    LAST_ROW = None
    chash = config_hash(cfg)
    cached = None if (args.force or args.check) else find_summary_row(chash)
    if cached:
        print(
            f"[cached] {experiment_name}: same config already in benchmark.csv "
            f"(row {cached['timestamp']}, commit {cached['git_commit']}, "
            f"{cached['n_samples']} samples). --force to rerun."
        )
        LAST_ROW = cached
        return 0

    # ---- availability: fail early and explain, never mid-forward-pass -----
    key = model_cfg.get("adapter", model_cfg.get("name"))
    if key not in BASELINE_REGISTRY:
        import baselines  # noqa: F401
        import models  # noqa: F401
    cls = BASELINE_REGISTRY.get(key)
    if cls is None:
        raise SystemExit(f"unknown model {key!r}; registered: {sorted(BASELINE_REGISTRY)}")
    avail = cls.availability(model_cfg.get("checkpoint"), **model_cfg)
    print(avail.report(key))
    if args.check:
        return 0 if avail.ok else 1
    if not avail.ok:
        return 1

    set_seed(int(cfg.get("seed", 0)))

    # ---- data -------------------------------------------------------------
    degradation = build_degradation(dict(cfg.degradation) if "degradation" in cfg else None)
    eval_cfg = dict(cfg.get("eval", {}))
    dataset = build_dataset(
        dataset_cfg,
        degradation=degradation,
        **({"max_samples": eval_cfg["max_samples"]} if eval_cfg.get("max_samples") else {}),
    )
    print(f"[data] {dataset.info.name}: {len(dataset)} samples, modality={dataset.info.modality}")

    min_depth = eval_cfg.get("min_depth") or dataset.min_depth
    max_depth = eval_cfg.get("max_depth") or dataset.max_depth
    metric_cfg = MetricConfig(
        min_depth=float(min_depth),
        max_depth=float(max_depth),
        crop=str(eval_cfg.get("crop", "none")),
    )

    # ---- model ------------------------------------------------------------
    # A model's depth range is not the evaluation window.  DEPTHOR's head sums
    # over bins spanning [min_depth, max_depth], so for it those two numbers are
    # a property of the *checkpoint*: scoring on a dataset with a narrower window
    # must not move them, or the learned bin logits decode to the wrong depths.
    # A configs/model/*.yaml may therefore state its own range, and only when it
    # does not do we fall back to the metric window (which keeps every existing
    # config behaving exactly as before).
    m_min, m_max = model_depth_range(model_cfg, metric_cfg.min_depth, metric_cfg.max_depth)
    model = build_baseline(
        model_cfg, device=cfg.get("device", "auto"), min_depth=m_min, max_depth=m_max
    )
    model.load(model_cfg.get("checkpoint"))
    print(f"[model] {model.describe()}")

    recorder = RunRecorder(
        experiment=experiment_name,
        config=cfg.to_dict(),
        tracker=get_tracker(dict(cfg.get("tracking", {})), f"{experiment_name}:{key}"),
    )

    # ---- evaluate ---------------------------------------------------------
    acc = DepthMetricAccumulator(metric_cfg, pooling=str(eval_cfg.get("pooling", "image")))
    unreadable: list[str] = []
    for i in range(len(dataset)):
        try:
            sample = dataset[i]
        except OSError as exc:
            # A single truncated file on the shared data folder must not throw
            # away a run that is twenty minutes in -- an interrupted copy leaves
            # zero-byte files, and MinJiang has two. Count and name them instead;
            # the count reaches the results row, so a short run can never be
            # mistaken for a complete one. Only *reading* is forgiven here: a
            # model or metric failure still stops everything.
            unreadable.append(f"{i}: {exc}")
            if len(unreadable) <= 5:
                print(f"  [skip] unreadable sample {i}: {exc}")
            continue
        pred = model.predict_sample(sample)
        acc.update(pred, sample["gt_depth"], sample["mask"], sample["meta"]["sample_id"])
        if (i + 1) % 50 == 0 or i + 1 == len(dataset):
            print(f"  [{i + 1}/{len(dataset)}] {format_metrics(acc.compute())}")

    if unreadable:
        print(
            f"[data] {len(unreadable)} of {len(dataset)} samples could not be read and were "
            f"skipped; n_unreadable is recorded in the results row"
        )
    if len(unreadable) == len(dataset):
        raise SystemExit("[data] every sample failed to load; nothing was scored")

    summary: dict[str, Any] = acc.compute()
    summary["n_unreadable"] = len(unreadable)

    # ---- runtime ----------------------------------------------------------
    if cfg.get("runtime", {}).get("measure", True) and len(dataset):
        sample = dataset[0]
        depth_in = sample["lr_depth"] if model.input_modality == "lr" else sample["sparse_depth"]
        rt = measure_runtime(
            lambda: model.predict(sample["rgb"], depth_in),
            device=model.device,
            config=RuntimeConfig(
                warmup=int(cfg.runtime.get("warmup", 5)), iters=int(cfg.runtime.get("iters", 20))
            ),
            input_hw=sample["rgb"].shape[:2],
        )
        summary["fps"] = rt.fps
        summary["latency_ms_median"] = rt.latency_ms_median
        summary["runtime"] = rt.as_dict()

    # ---- record -----------------------------------------------------------
    summary.update(
        {
            "dataset": dataset.info.name,
            "split": dataset.split,
            "method": key,
            "checkpoint": model_cfg.get("checkpoint") or "",
            "degradation": (degradation.key if degradation else "none"),
            "seed": cfg.get("seed", 0),
            "device": model.device,
            "metric_config": metric_cfg.to_json(),
            "config_hash": chash,
        }
    )
    recorder.save_per_sample(acc.per_sample)
    row = recorder.finalize(summary, notes=args.notes or str(cfg.get("notes", "")))
    LAST_ROW = row

    print("\n=== result ===")
    print(
        f"{dataset.info.name} x {key}  ({summary['n_samples']} samples, "
        f"{summary.get('n_skipped', 0)} skipped)"
    )
    print(format_metrics(summary, METRIC_NAMES))
    if "fps" in summary:
        print(f"fps={summary['fps']:.2f}  latency_median={summary['latency_ms_median']:.1f} ms")
    print(f"run dir: {recorder.out_dir}")
    print(f"appended to: experiments/results/benchmark.csv (row {row['timestamp']})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
