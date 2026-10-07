#!/usr/bin/env python3
"""See where a method fails: its worst frames next to the ground truth.

    uv run --group notebooks python scripts/worst_cases.py depthor_minjiang -n 8
    uv run --group notebooks python scripts/worst_cases.py depthor_minjiang wave_minjiang_x8 --by rmse --stats

For each experiment the frames of its latest run
(``experiments/runs/<stamp>_<experiment>/per_sample.csv``) are ranked by one metric,
the model is re-run on the N worst frames only, and one picture is written with a
row per frame:

    RGB | what the model got | prediction | ground truth | signed relative error

The error panel is (pred - gt) / gt clipped to +-50%: red = too far, blue = too
near, so a scale bias (one colour everywhere) reads differently from edge errors
(thin lines along steps) and from missing structure (blobs).  Depth panels share
one colour scale per frame.  Next to the picture goes a CSV of per-frame numbers.

``--stats`` also joins cheap ground-truth covariates (depth range, share of far
pixels, density of depth jumps, share of GT holes) to the per-frame error of ALL
frames and prints rank correlations -- the quick test of "do far scenes /
many steps make the error larger".

Aggregate numbers come from ``metrics/depth_metrics.py`` and nowhere else; the
panels are pictures only.  Output goes to ``outputs/worst_cases/<experiment>/``
(gitignored).  matplotlib lives in the ``notebooks`` dependency group.
"""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path

import _bootstrap
import numpy as np

from baselines import build_baseline
from baselines.base import model_depth_range
from data.degradation import build_degradation
from data.loaders import build_dataset
from metrics.depth_metrics import MetricConfig, compute_depth_metrics
from utils.config import load_experiment
from utils.misc import resize_depth, set_seed

REPO = _bootstrap.REPO_ROOT
HIGHER_IS_BETTER = {"delta1", "delta2", "delta3"}
FAR_M = 4.0  # the dToF simulator drops zones beyond this


def latest_run(experiment: str, runs_dir: Path) -> Path:
    pat = re.compile(r"\d{8}-\d{6}_" + re.escape(experiment))
    runs = sorted(
        p for p in runs_dir.iterdir() if pat.fullmatch(p.name) and (p / "per_sample.csv").exists()
    )
    if not runs:
        raise SystemExit(
            f"no run of {experiment!r} with a per_sample.csv under {runs_dir}. "
            "Run it first (scripts/run_baseline.py experiment=<name>); a cached result has no run folder "
            "on this machine."
        )
    return runs[-1]


def read_per_sample(run: Path) -> list[dict[str, str]]:
    with (run / "per_sample.csv").open(newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def covariates(gt: np.ndarray, lo: float, hi: float) -> dict[str, float]:
    """Cheap per-frame facts about the ground truth alone."""
    valid = np.isfinite(gt) & (gt > lo) & (gt < hi)
    v = gt[valid]
    if v.size == 0:
        return {
            k: float("nan")
            for k in ("gt_median", "gt_p95", "far_frac", "gt_hole_frac", "jump_density")
        }
    jump = []
    for hi_px, lo_px, ok in (
        (gt[:, 1:], gt[:, :-1], valid[:, 1:] & valid[:, :-1]),
        (gt[1:], gt[:-1], valid[1:] & valid[:-1]),
    ):
        rel = np.abs(hi_px - lo_px)[ok] / np.maximum(lo_px[ok], 1e-6)
        jump.append((rel > 0.10).mean() if rel.size else 0.0)  # a >10% step between neighbours
    return {
        "gt_median": float(np.median(v)),
        "gt_p95": float(np.percentile(v, 95)),
        "far_frac": float((v > FAR_M).mean()),
        "gt_hole_frac": float((gt <= 0).mean()),
        "jump_density": float(np.mean(jump)),
    }


def spearman(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    from scipy.stats import spearmanr

    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 5 or np.ptp(x[ok]) == 0:
        return float("nan"), float("nan")
    r = spearmanr(x[ok], y[ok])
    return float(r.statistic), float(r.pvalue)


def panels(sample: dict, pred: np.ndarray, model) -> dict[str, np.ndarray]:
    """Everything that is drawn, as arrays."""
    from scipy.ndimage import maximum_filter

    gt, mask, rgb = sample["gt_depth"], sample["mask"], sample["rgb"]
    h, w = gt.shape
    if model.input_modality == "lr":
        inp = resize_depth(sample["lr_depth"], (h, w), mode="nearest")
    else:
        sp = sample["sparse_depth"]
        n = int((sp > 0).sum())
        inp = (
            maximum_filter(sp, size=9) if 0 < n < 5000 else sp
        )  # a 64-point map is invisible otherwise
    rel = np.where(mask, (pred - gt) / np.maximum(gt, 1e-6), np.nan)
    return {"rgb": rgb, "inp": inp, "pred": pred, "gt": np.where(mask, gt, np.nan), "rel": rel}


def draw(rows: list[dict], title: str, path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n = len(rows)
    fig, axes = plt.subplots(n, 5, figsize=(15.5, 2.55 * n), squeeze=False)
    cmap = plt.get_cmap("turbo").copy()
    cmap.set_bad("black")
    ocmap = (
        cmap.copy()
    )  # the "input" panel is drawn over the dimmed RGB: holes must stay see-through
    ocmap.set_bad((0, 0, 0, 0))
    ecmap = plt.get_cmap("RdBu_r").copy()
    ecmap.set_bad("#bbbbbb")
    for r, row in enumerate(rows):
        p = row["panels"]
        v = p["gt"][np.isfinite(p["gt"])]
        vmin, vmax = (np.percentile(v, 2), np.percentile(v, 98)) if v.size else (0.0, 1.0)
        inp = np.where(p["inp"] > 0, p["inp"], np.nan)
        axes[r, 0].imshow(np.clip(p["rgb"], 0, 1))
        axes[r, 1].imshow(np.clip(p["rgb"], 0, 1) * 0.35)
        axes[r, 1].imshow(inp, cmap=ocmap, vmin=vmin, vmax=vmax)
        axes[r, 2].imshow(p["pred"], cmap=cmap, vmin=vmin, vmax=vmax)
        axes[r, 3].imshow(p["gt"], cmap=cmap, vmin=vmin, vmax=vmax)
        axes[r, 4].imshow(p["rel"], cmap=ecmap, vmin=-0.5, vmax=0.5)
        axes[r, 0].set_title(row["label"], fontsize=7, loc="left")
        for a in axes[r]:
            a.axis("off")
    for c, name in enumerate(
        [
            "RGB",
            "что получила модель",
            "предсказание",
            "эталон",
            "(pred − gt)/gt: красный = дальше, синий = ближе",
        ]
    ):
        axes[0, c].text(
            0.5, 1.28, name, transform=axes[0, c].transAxes, ha="center", fontsize=8, weight="bold"
        )
    fig.suptitle(title, fontsize=10, y=1.0)
    fig.tight_layout(rect=(0, 0, 1, 0.985))
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=85)
    plt.close(fig)


def run_one(experiment: str, args: argparse.Namespace) -> None:
    cfg = load_experiment(experiment)
    run = (
        Path(args.run_dir)
        if args.run_dir
        else latest_run(experiment, REPO / "experiments" / "runs")
    )
    per = read_per_sample(run)
    by = args.by

    finite = [r for r in per if np.isfinite(float(r[by]))]
    # worst first: highest for the "lower is better" metrics, lowest for delta1
    ranked = sorted(finite, key=lambda r: float(r[by]), reverse=by not in HIGHER_IS_BETTER)
    worst = ranked[: args.n]

    set_seed(int(cfg.get("seed", 0)))
    model_cfg = dict(cfg.model)
    degradation = build_degradation(dict(cfg.degradation) if "degradation" in cfg else None)
    dataset = build_dataset(dict(cfg.dataset), degradation=degradation)
    eval_cfg = dict(cfg.get("eval", {}))
    metric_cfg = MetricConfig(
        min_depth=float(eval_cfg.get("min_depth") or dataset.min_depth),
        max_depth=float(eval_cfg.get("max_depth") or dataset.max_depth),
        crop=str(eval_cfg.get("crop", "none")),
    )
    m_min, m_max = model_depth_range(model_cfg, metric_cfg.min_depth, metric_cfg.max_depth)
    model = build_baseline(
        model_cfg, device=args.device or cfg.get("device", "auto"), min_depth=m_min, max_depth=m_max
    )
    model.load(model_cfg.get("checkpoint"))
    index = {dataset.record_id(rec): i for i, rec in enumerate(dataset.records)}

    out = REPO / "outputs" / "worst_cases" / experiment
    rows, table = [], []
    for rank, r in enumerate(worst, 1):
        sid = r["sample_id"]
        if sid not in index:
            print(f"[skip] {sid}: not in the dataset built from the current config")
            continue
        sample = dataset[index[sid]]
        pred = model.predict_sample(sample)
        m = compute_depth_metrics(pred, sample["gt_depth"], sample["mask"], metric_cfg)
        gt, mask = sample["gt_depth"], sample["mask"]
        ratio = float(np.median(pred[mask] / gt[mask])) if mask.any() else float("nan")
        cov = covariates(gt, metric_cfg.min_depth, metric_cfg.max_depth)
        n_pts = (
            int((sample["sparse_depth"] > 0).sum()) if model.input_modality == "sparse" else None
        )
        label = (
            f"#{rank} {sid}\nAbsRel {m['absrel']:.3f}  RMSE {m['rmse']:.3f}  δ1 {m['delta1']:.3f}  "
            f"pred/gt медиана {ratio:.2f}\nэталон {cov['gt_median']:.1f} м (p95 {cov['gt_p95']:.1f}), "
            f">{FAR_M:g} м: {cov['far_frac']:.0%}"
            + (f", точек на входе: {n_pts}" if n_pts is not None else "")
        )
        rows.append({"panels": panels(sample, pred, model), "label": label})
        table.append(
            {
                "rank": rank,
                "sample_id": sid,
                **{k: m[k] for k in ("absrel", "rmse", "delta1", "n_valid")},
                "pred_over_gt_median": ratio,
                "input_points": n_pts,
                **cov,
            }
        )
        print(f"  #{rank} {sid}: absrel {m['absrel']:.4f} rmse {m['rmse']:.4f} pred/gt {ratio:.2f}")

    if rows:
        png = out / f"worst_{by}.png"
        draw(rows, f"{experiment}: {len(rows)} худших кадров по {by} (запуск {run.name})", png)
        with (out / f"worst_{by}.csv").open("w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=list(table[0]))
            w.writeheader()
            w.writerows(table)
        print(f"[{experiment}] {png}")

    if args.stats:
        errs, covs = [], []
        for r in per:
            i = index.get(r["sample_id"])
            if i is None:
                continue
            try:
                raw = dataset._load_raw(dataset.records[i])
            except OSError:
                continue
            errs.append(float(r[by]))
            covs.append(
                covariates(
                    np.nan_to_num(np.asarray(raw["gt_depth"], np.float32)),
                    metric_cfg.min_depth,
                    metric_cfg.max_depth,
                )
            )
        print(
            f"\n[{experiment}] rank correlation of per-frame {by} with ground-truth properties ({len(errs)} frames)"
        )
        print(f"  {'property':28s} {'rho':>7s} {'p':>9s}")
        names = {
            "gt_median": "median depth",
            "gt_p95": "95th percentile depth",
            "far_frac": f"share of pixels > {FAR_M:g} m",
            "jump_density": "density of depth jumps (steps)",
            "gt_hole_frac": "share of GT holes",
        }
        y = np.array(errs)
        for k, name in names.items():
            rho, p = spearman(np.array([c[k] for c in covs]), y)
            print(f"  {name:28s} {rho:7.2f} {p:9.1e}")


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("experiments", nargs="+", help="experiment names (configs/experiment/*.yaml)")
    ap.add_argument("-n", type=int, default=8, help="how many worst frames to draw (default 8)")
    ap.add_argument(
        "--by",
        default="absrel",
        choices=["absrel", "rmse", "rmse_log", "silog", "delta1"],
        help="ranking metric",
    )
    ap.add_argument(
        "--stats",
        action="store_true",
        help="also correlate the per-frame error with ground-truth properties",
    )
    ap.add_argument("--device", default=None)
    ap.add_argument("--run-dir", default=None, help="use this run folder instead of the latest one")
    args = ap.parse_args()
    for exp in args.experiments:
        print(f"== {exp}")
        run_one(exp, args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
