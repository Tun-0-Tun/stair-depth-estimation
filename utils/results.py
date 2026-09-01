"""Local experiment tracking.

Deliberately boring: a per-run JSON with the full config + environment, a
per-run CSV of per-sample metrics, and one append-only summary CSV that *is*
committed so the three of us always look at the same table.

W&B / MLflow are optional and off by default (``tracking.backend: none``), so
nobody is blocked on having an account.  See ``utils/results.py::get_tracker``.
"""

from __future__ import annotations

import csv
import json
import platform
import subprocess
import time
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from metrics.depth_metrics import METRIC_NAMES
from utils.misc import REPO_ROOT

__all__ = ["RESULTS_DIR", "SUMMARY_CSV", "RunRecorder", "collect_env", "write_markdown_table"]

RESULTS_DIR = REPO_ROOT / "experiments" / "results"
SUMMARY_CSV = RESULTS_DIR / "benchmark.csv"

SUMMARY_COLUMNS: tuple[str, ...] = (
    "timestamp",
    "experiment",
    "dataset",
    "split",
    "method",
    "checkpoint",
    "degradation",
    "seed",
    "device",
    "n_samples",
    "n_skipped",
    *METRIC_NAMES,
    "fps",
    "latency_ms_median",
    "git_commit",
    "metric_config",
    "notes",
)


def _git(*args: str) -> str | None:
    try:
        return (
            subprocess.check_output(["git", *args], cwd=REPO_ROOT, stderr=subprocess.DEVNULL)
            .decode()
            .strip()
        )
    except Exception:
        return None


def collect_env() -> dict[str, Any]:
    """Everything needed to explain why a number changed between two runs."""
    env: dict[str, Any] = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "git_commit": _git("rev-parse", "HEAD"),
        "git_dirty": bool(_git("status", "--porcelain")),
        "git_submodules": _git("submodule", "status"),
    }
    try:
        import numpy
        import torch

        env["torch"] = torch.__version__
        env["numpy"] = numpy.__version__
        env["cuda"] = torch.version.cuda if torch.cuda.is_available() else None
    except ImportError:
        pass
    return env


@dataclass
class RunRecorder:
    """Collect one run's artefacts and append it to the shared summary table."""

    experiment: str
    config: Mapping[str, Any]
    out_dir: Path | None = None
    tracker: Any = None

    def __post_init__(self) -> None:
        stamp = time.strftime("%Y%m%d-%H%M%S")
        self.timestamp = stamp
        self.out_dir = Path(
            self.out_dir or (REPO_ROOT / "experiments" / "runs" / f"{stamp}_{self.experiment}")
        )
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.env = collect_env()
        (self.out_dir / "config.json").write_text(
            json.dumps({"config": dict(self.config), "env": self.env}, indent=2, default=str),
            encoding="utf-8",
        )

    def save_per_sample(self, rows: Sequence[Mapping[str, Any]]) -> Path | None:
        if not rows:
            return None
        path = self.out_dir / "per_sample.csv"
        cols = list(dict.fromkeys(k for r in rows for k in r))
        with path.open("w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=cols)
            w.writeheader()
            w.writerows(rows)
        return path

    def finalize(self, summary: Mapping[str, Any], notes: str = "") -> dict[str, Any]:
        """Write the run summary and append one row to the committed table."""
        row = dict.fromkeys(SUMMARY_COLUMNS, "")
        row.update({k: v for k, v in summary.items() if k in SUMMARY_COLUMNS})
        row["timestamp"] = self.timestamp
        row["experiment"] = self.experiment
        row["git_commit"] = (self.env.get("git_commit") or "")[:12] + (
            "-dirty" if self.env.get("git_dirty") else ""
        )
        row["notes"] = notes

        (self.out_dir / "summary.json").write_text(
            json.dumps(dict(summary), indent=2, default=str), encoding="utf-8"
        )
        append_summary_row(row)
        if self.tracker is not None:
            self.tracker.log(dict(summary))
            self.tracker.finish()
        return row


def append_summary_row(row: Mapping[str, Any], path: Path | None = None) -> Path:
    """Append one row to the shared, git-tracked benchmark table."""
    path = Path(path or SUMMARY_CSV)
    path.parent.mkdir(parents=True, exist_ok=True)
    new = not path.exists()
    with path.open("a", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(SUMMARY_COLUMNS), extrasaction="ignore")
        if new:
            w.writeheader()
        w.writerow({c: row.get(c, "") for c in SUMMARY_COLUMNS})
    return path


def write_markdown_table(
    rows: Iterable[Mapping[str, Any]],
    columns: Sequence[str],
    path: Path,
    float_fmt: str = "{:.4f}",
) -> Path:
    """Render a dataset x method x metric table as Markdown for the docs/PR."""

    def cell(v: Any) -> str:
        if isinstance(v, float):
            return "n/a" if v != v else float_fmt.format(v)
        return "" if v is None else str(v)

    rows = list(rows)
    lines = ["| " + " | ".join(columns) + " |", "|" + "|".join(["---"] * len(columns)) + "|"]
    for r in rows:
        lines.append("| " + " | ".join(cell(r.get(c)) for c in columns) + " |")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def get_tracker(cfg: Mapping[str, Any], run_name: str):
    """Optional W&B / MLflow tracker.  Returns ``None`` for the default backend.

    Configured under ``tracking:`` in the experiment YAML::

        tracking:
          backend: none   # none | wandb | mlflow
          project: stair-depth-estimation

    Neither library is a hard dependency; a missing import is a clear error
    rather than a silent downgrade, so a run that claims to log to W&B does.
    """
    backend = str((cfg or {}).get("backend", "none")).lower()
    if backend in ("none", "", "local"):
        return None
    if backend == "wandb":
        import wandb

        run = wandb.init(
            project=cfg.get("project", "stair-depth-estimation"), name=run_name, config=dict(cfg)
        )

        class _W:
            def log(self, d):
                wandb.log(d)

            def finish(self):
                run.finish()

        return _W()
    if backend == "mlflow":
        import mlflow

        mlflow.set_experiment(cfg.get("project", "stair-depth-estimation"))
        mlflow.start_run(run_name=run_name)

        class _M:
            def log(self, d):
                mlflow.log_metrics({k: v for k, v in d.items() if isinstance(v, (int | float))})

            def finish(self):
                mlflow.end_run()

        return _M()
    raise ValueError(f"unknown tracking backend {backend!r}; expected none | wandb | mlflow")
