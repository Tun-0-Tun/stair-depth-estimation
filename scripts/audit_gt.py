#!/usr/bin/env python3
"""How good is each dataset's ground truth, measured the same way for all of them.

    uv run python scripts/audit_gt.py                 # every dataset found under $STAIR_DATA_ROOT
    uv run python scripts/audit_gt.py minjiang hammer -n 60

Four numbers per dataset, taken from the GT depth alone (n frames spread evenly):

* **holes**   share of pixels with no value (0 / NaN) -- nothing can be scored there;
* **in range**  share inside the dataset's metric window;
* **step**    typical gap between neighbouring distinct depth values (mm): 1 for
              millimetre PNGs, 17-257 for the per-frame 8-bit rgbd_stair depth;
* **noise**   roughness of flat surfaces: random 7x7 patches that are valid and
              smooth are fitted with a plane and the RMS residual is reported, by
              distance (noise grows with range, so compare at equal distance).
              A laser / synthetic GT sits near zero; a sensor used as GT shows
              its own noise here -- which any model scored against it inherits.

The plane fit also picks up curved surfaces, so read the figures relative to each
other, not as absolute sensor accuracy.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import _bootstrap
import numpy as np

from data.degradation import build_degradation
from data.loaders import build_dataset
from utils.config import load_config

# dataset config -> what its GT is (our reading of the docs, not measured here)
KINDS = {
    "minjiang": "сам датчик",
    "ours": "сам датчик (Astra 2)",
    "rgbd_stair": "сам датчик, 8 бит",
    "void_stairs": "плотный reference (происхождение не проверено)",
    "zju_l5": "стереореконструкция",
    "nyuv2": "Kinect, заполненная глубина",
    "hammer": "лазер",
    "arkitscenes": "лазер (Faro)",
    "hypersim": "синтетика, точная",
}
BINS = ((0.5, 1.5), (1.5, 3.0), (3.0, 6.0))
PATCH = 7


def plane_residual_operator(k: int = PATCH) -> np.ndarray:
    """(k*k, k*k) matrix that maps a patch to its residual from the best-fit plane."""
    yy, xx = np.mgrid[:k, :k]
    a = np.stack([xx.ravel(), yy.ravel(), np.ones(k * k)], axis=1).astype(np.float64)
    return np.eye(k * k) - a @ np.linalg.pinv(a)


def flat_patch_noise(
    gt: np.ndarray, lo: float, hi: float, rng: np.random.Generator, tries: int = 3000
) -> list[tuple[float, float]]:
    """(median depth, RMS plane-fit residual) of random valid, smooth patches."""
    h, w = gt.shape
    if h <= PATCH or w <= PATCH:
        return []
    ys = rng.integers(0, h - PATCH, tries)
    xs = rng.integers(0, w - PATCH, tries)
    off = np.arange(PATCH)
    patches = gt[(ys[:, None, None] + off[None, :, None]), (xs[:, None, None] + off[None, None, :])]
    patches = patches.reshape(tries, -1).astype(np.float64)
    ok = np.isfinite(patches).all(1) & (patches > lo).all(1) & (patches < hi).all(1)
    patches = patches[ok]
    if not len(patches):
        return []
    med = np.median(patches, axis=1)
    smooth = (patches.max(1) - patches.min(1)) < 0.08 * med  # edges and small objects out
    patches, med = patches[smooth], med[smooth]
    rms = np.sqrt(((patches @ plane_residual_operator().T) ** 2).mean(1))
    return list(zip(med.tolist(), rms.tolist(), strict=True))


def step_mm(gt: np.ndarray, lo: float, hi: float) -> float:
    """Median gap between neighbouring distinct valid depth values, in mm."""
    v = np.unique(gt[np.isfinite(gt) & (gt > lo) & (gt < hi)])
    if len(v) < 2:
        return float("nan")
    return float(np.median(np.diff(v.astype(np.float64))) * 1000.0)


def audit_frames(gts: list[np.ndarray], lo: float, hi: float, seed: int = 0) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    holes, in_range, steps, patches = [], [], [], []
    for gt in gts:
        g = np.nan_to_num(np.asarray(gt, dtype=np.float64), nan=0.0, posinf=0.0, neginf=0.0)
        holes.append(float((g <= 0).mean()))
        in_range.append(float(((g > lo) & (g < hi)).mean()))
        steps.append(step_mm(g, lo, hi))
        patches += flat_patch_noise(g, lo, hi, rng)
    out = {
        "frames": len(gts),
        "holes_pct": 100 * float(np.mean(holes)),
        "in_range_pct": 100 * float(np.mean(in_range)),
        "step_mm": float(np.nanmedian(steps)) if np.isfinite(steps).any() else float("nan"),
        "patches": len(patches),
    }
    p = np.array(patches) if patches else np.zeros((0, 2))
    for a, b in BINS:
        sel = (p[:, 0] >= a) & (p[:, 0] < b) if len(p) else np.array([], dtype=bool)
        out[f"noise_mm_{a:g}-{b:g}m"] = (
            float(np.median(p[sel, 1]) * 1000) if sel.sum() >= 100 else float("nan")
        )
    return out


def load_gts(key: str, n: int) -> tuple[list[np.ndarray], float, float]:
    cfg = dict(load_config("dataset", key))
    ds = build_dataset(cfg, degradation=build_degradation({"type": "bicubic_sr", "scale": 4}))
    recs = ds.records[:: max(1, len(ds.records) // n)][:n]
    gts = []
    for r in recs:
        try:
            gts.append(np.asarray(ds._load_raw(r)["gt_depth"], dtype=np.float32))
        except OSError:
            continue  # a truncated file from an interrupted copy
    return gts, ds.min_depth, ds.max_depth


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("datasets", nargs="*", help=f"any of {sorted(KINDS)} (default: all that exist)")
    ap.add_argument("-n", type=int, default=40, help="frames per dataset (default 40)")
    ap.add_argument("--out", default=str(_bootstrap.REPO_ROOT / "outputs" / "audit_gt.csv"))
    args = ap.parse_args()

    rows = []
    for key in args.datasets or list(KINDS):
        try:
            gts, lo, hi = load_gts(key, args.n)
        except (FileNotFoundError, RuntimeError, KeyError) as exc:
            print(f"[skip] {key}: {str(exc).splitlines()[0][:110]}")
            continue
        if not gts:
            print(f"[skip] {key}: no readable frames")
            continue
        rows.append({"dataset": key, "gt": KINDS.get(key, ""), **audit_frames(gts, lo, hi)})
        print(f"[ok]   {key}: {len(gts)} frames")

    if not rows:
        return 1
    cols = list(rows[0])
    print()
    print(
        f"{'dataset':12s} {'GT':28s} {'holes%':>7s} {'range%':>7s} {'step mm':>8s}  "
        + "  ".join(f"noise {a:g}-{b:g} m" for a, b in BINS)
    )
    for r in rows:
        noise = "  ".join(
            f"{r[f'noise_mm_{a:g}-{b:g}m']:>11.1f}"
            if np.isfinite(r[f"noise_mm_{a:g}-{b:g}m"])
            else f"{'-':>11s}"
            for a, b in BINS
        )
        print(
            f"{r['dataset']:12s} {r['gt'][:28]:28s} {r['holes_pct']:7.1f} {r['in_range_pct']:7.1f} {r['step_mm']:8.1f}  {noise}   (мм, {r['patches']} участков)"
        )
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)
    print(f"\n{args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
