"""The *single* implementation of depth-accuracy metrics in this repository.

RULE (enforced in code review, see CONTRIBUTING.md):
    Nobody writes another AbsRel / RMSE / delta implementation anywhere else.
    Everything imports from here.  Baselines, our model, notebooks, ad-hoc
    scripts -- all of them.  If you need a new metric, add it *here* with a
    unit test in ``tests/test_metrics.py``.

Why this matters
----------------
The official baseline repos each use their own convention and they are not
comparable out of the box:

* DEPTHOR (``third_party/depthor/src/utils/utils.py``) reports RMSE in *metres*
  over pixels with ``min_depth < gt < max_depth``.
* DuCos (``third_party/ducos/utils/metric_func.py``) reports RMSE in
  *centimetres* on min-max-normalised depth, after a 6-pixel border crop, and
  its "delta" thresholds are 1.05 / 1.15 / 1.25 rather than 1.25 / 1.25^2 /
  1.25^3.

Numbers produced by those scripts therefore cannot be put in the same table.
This module re-computes everything from raw metric depth in metres.

Conventions
-----------
* All depth is in **metres**, float32/float64, shape ``(H, W)`` or ``(B, H, W)``.
* ``0`` and non-finite values in the ground truth mean "no measurement".
* A pixel counts towards the metrics iff
  ``min_depth < gt < max_depth`` **and** ``gt`` is finite **and** the optional
  user-supplied ``mask`` is True there.
* Predictions are clamped into ``[min_depth, max_depth]`` before evaluation
  (``MetricConfig.clamp_pred``).  This is what every baseline does, and without
  it a single divergent pixel destroys RMSE.
* Default aggregation is **per-image then averaged over images**, which is what
  NYUv2 / ZJU-L5 / RGB-D-D papers report.  Pixel-pooled aggregation is available
  via ``DepthMetricAccumulator(pooling="pixel")`` for ablations.

See ``docs/metrics_protocol.md`` for the full protocol and the cross-check
against published numbers.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass, field

import numpy as np

__all__ = [
    "LOWER_IS_BETTER",
    "METRIC_NAMES",
    "PRIMARY_METRICS",
    "DepthMetricAccumulator",
    "MetricConfig",
    "compute_depth_metrics",
    "format_metrics",
]

#: Every metric this module can produce, in canonical column order.
METRIC_NAMES: tuple[str, ...] = (
    "absrel",
    "rmse",
    "rmse_log",
    "delta1",
    "delta2",
    "delta3",
    "mae",
    "sqrel",
    "silog",
)

#: The four acceptance metrics from the SOW.  ``fps`` is produced by
#: ``metrics.runtime_metrics`` and joined into the same table.
PRIMARY_METRICS: tuple[str, ...] = ("absrel", "rmse", "delta1")

#: Direction of each metric, used by table formatting / regression checks.
LOWER_IS_BETTER: Mapping[str, bool] = {
    "absrel": True,
    "rmse": True,
    "rmse_log": True,
    "mae": True,
    "sqrel": True,
    "silog": True,
    "delta1": False,
    "delta2": False,
    "delta3": False,
}

#: Named evaluation crops.  Values are ``(top, bottom, left, right)`` pixel
#: counts to remove, or a callable-free special case handled in ``_apply_crop``.
_CROPS: Mapping[str, str] = {
    "none": "no crop, full frame",
    "eigen": "Eigen NYUv2 crop, rows 45:471 cols 41:601 of a 480x640 frame",
    "border6": "drop a 6-pixel border, the convention of the DSR literature",
}


@dataclass(frozen=True)
class MetricConfig:
    """Evaluation protocol.  Serialise this next to every result you report.

    Attributes
    ----------
    min_depth, max_depth:
        Valid ground-truth range in metres.  Pixels outside it are excluded from
        the mask, and predictions are clamped into it.  Per-dataset values live
        in ``configs/dataset/*.yaml`` -- do not hard-code them elsewhere.
    clamp_pred:
        Clamp predictions into ``[min_depth, max_depth]`` before scoring.
    crop:
        One of ``METRIC_CROPS``.  ``"none"`` by default; NYUv2 comparisons
        against published tables need ``"eigen"``.
    eps:
        Numerical floor for divisions and logarithms.
    min_valid_pixels:
        Images with fewer valid ground-truth pixels than this are skipped by
        ``DepthMetricAccumulator`` (and reported in ``n_skipped``).
    """

    min_depth: float = 1e-3
    max_depth: float = 10.0
    clamp_pred: bool = True
    crop: str = "none"
    eps: float = 1e-6
    min_valid_pixels: int = 1

    def __post_init__(self) -> None:
        if self.crop not in _CROPS:
            raise ValueError(f"unknown crop {self.crop!r}; expected one of {sorted(_CROPS)}")
        if not (self.min_depth >= 0.0 and self.max_depth > self.min_depth):
            raise ValueError(
                f"need 0 <= min_depth < max_depth, got {self.min_depth} / {self.max_depth}"
            )

    def to_json(self) -> str:
        return json.dumps(asdict(self), sort_keys=True)


def _apply_crop(arr: np.ndarray, crop: str) -> np.ndarray:
    """Crop the last two dimensions of ``arr`` according to ``crop``."""
    if crop == "none":
        return arr
    if crop == "eigen":
        h, w = arr.shape[-2:]
        if (h, w) != (480, 640):
            raise ValueError(
                f"the eigen crop is defined for 480x640 frames, got {h}x{w}. "
                "Resize to 480x640 first or use crop='none'."
            )
        return arr[..., 45:471, 41:601]
    if crop == "border6":
        return arr[..., 6:-6, 6:-6]
    raise ValueError(f"unknown crop {crop!r}")


def _as_2d_float(name: str, arr) -> np.ndarray:
    a = np.asarray(arr)
    if a.ndim == 3 and a.shape[0] == 1:  # (1, H, W)
        a = a[0]
    elif a.ndim == 3 and a.shape[-1] == 1:  # (H, W, 1)
        a = a[..., 0]
    if a.ndim != 2:
        raise ValueError(f"{name} must be (H, W); got shape {np.asarray(arr).shape}")
    return a.astype(np.float64, copy=False)


def compute_depth_metrics(
    pred: np.ndarray,
    gt: np.ndarray,
    mask: np.ndarray | None = None,
    config: MetricConfig | None = None,
) -> dict[str, float]:
    """Metrics for one image pair.

    Parameters
    ----------
    pred, gt:
        Dense depth in metres, shape ``(H, W)`` (a leading or trailing singleton
        axis is accepted and squeezed).
    mask:
        Optional extra boolean validity mask, ANDed with the ground-truth range
        mask.  Use it to exclude e.g. mirror regions or unlabelled pixels.
    config:
        Evaluation protocol; defaults to ``MetricConfig()``.

    Returns
    -------
    dict
        ``METRIC_NAMES`` plus ``n_valid`` (pixels scored) and ``valid_ratio``.
        If no pixel is valid every metric is ``nan`` and ``n_valid == 0`` --
        callers must not silently average ``nan`` away; the accumulator skips
        such frames and counts them.
    """
    cfg = config or MetricConfig()

    pred = _as_2d_float("pred", pred)
    gt = _as_2d_float("gt", gt)
    if pred.shape != gt.shape:
        raise ValueError(f"pred {pred.shape} and gt {gt.shape} must have the same shape")

    if mask is None:
        valid = np.ones_like(gt, dtype=bool)
    else:
        valid = _as_2d_float("mask", mask).astype(bool)
        if valid.shape != gt.shape:
            raise ValueError(f"mask {valid.shape} does not match gt {gt.shape}")

    pred = _apply_crop(pred, cfg.crop)
    gt = _apply_crop(gt, cfg.crop)
    valid = _apply_crop(valid, cfg.crop)

    n_total = gt.size
    valid &= np.isfinite(gt) & (gt > cfg.min_depth) & (gt < cfg.max_depth)
    # A non-finite prediction is a model failure, not an excuse to drop the
    # pixel: clamping turns it into max_depth so it is penalised, not ignored.
    if cfg.clamp_pred:
        pred = np.nan_to_num(pred, nan=cfg.max_depth, posinf=cfg.max_depth, neginf=cfg.min_depth)
        pred = np.clip(pred, cfg.min_depth, cfg.max_depth)
    else:
        valid &= np.isfinite(pred)

    n_valid = int(valid.sum())
    out: dict[str, float] = {name: float("nan") for name in METRIC_NAMES}
    out["n_valid"] = n_valid
    out["valid_ratio"] = n_valid / n_total if n_total else 0.0
    if n_valid == 0:
        return out

    p = pred[valid]
    g = gt[valid]

    diff = p - g
    abs_diff = np.abs(diff)

    out["absrel"] = float(np.mean(abs_diff / np.maximum(g, cfg.eps)))
    out["sqrel"] = float(np.mean(diff**2 / np.maximum(g, cfg.eps)))
    out["rmse"] = float(np.sqrt(np.mean(diff**2)))
    out["mae"] = float(np.mean(abs_diff))

    log_p = np.log(np.maximum(p, cfg.eps))
    log_g = np.log(np.maximum(g, cfg.eps))
    log_diff = log_p - log_g
    out["rmse_log"] = float(np.sqrt(np.mean(log_diff**2)))
    # Scale-invariant log error, x100 as in Eigen et al.  The variance is
    # clamped at zero: for a perfectly scale-consistent prediction
    # E[d^2] - E[d]^2 is analytically 0 but comes out slightly negative in
    # floating point, which would turn the metric into NaN.
    log_var = max(float(np.mean(log_diff**2) - np.mean(log_diff) ** 2), 0.0)
    out["silog"] = float(np.sqrt(log_var) * 100.0)

    ratio = np.maximum(p / np.maximum(g, cfg.eps), g / np.maximum(p, cfg.eps))
    out["delta1"] = float(np.mean(ratio < 1.25))
    out["delta2"] = float(np.mean(ratio < 1.25**2))
    out["delta3"] = float(np.mean(ratio < 1.25**3))

    return out


@dataclass
class DepthMetricAccumulator:
    """Aggregate per-sample metrics over a dataset.

    ``pooling="image"`` (default) averages the per-image metrics -- the
    convention of the depth-completion / DSR literature, and what the DEPTHOR
    and DuCos papers report.  ``pooling="pixel"`` concatenates all valid pixels
    and computes the metrics once, which weights large valid regions more
    heavily; useful for ablations, never for headline numbers.
    """

    config: MetricConfig = field(default_factory=MetricConfig)
    pooling: str = "image"

    def __post_init__(self) -> None:
        if self.pooling not in ("image", "pixel"):
            raise ValueError(f"pooling must be 'image' or 'pixel', got {self.pooling!r}")
        self._rows: list[dict[str, float]] = []
        self._pixels: list[tuple[np.ndarray, np.ndarray]] = []
        self.n_skipped = 0

    def update(
        self,
        pred: np.ndarray,
        gt: np.ndarray,
        mask: np.ndarray | None = None,
        sample_id: str | None = None,
    ) -> dict[str, float]:
        """Score one sample and remember it.  Returns that sample's metrics."""
        row = compute_depth_metrics(pred, gt, mask, self.config)
        if row["n_valid"] < self.config.min_valid_pixels:
            self.n_skipped += 1
            return row
        if sample_id is not None:
            row = {"sample_id": sample_id, **row}
        self._rows.append(row)

        if self.pooling == "pixel":
            cfg = self.config
            p = _apply_crop(_as_2d_float("pred", pred), cfg.crop)
            g = _apply_crop(_as_2d_float("gt", gt), cfg.crop)
            v = (
                np.ones_like(g, dtype=bool)
                if mask is None
                else _apply_crop(_as_2d_float("mask", mask).astype(bool), cfg.crop)
            )
            v &= np.isfinite(g) & (g > cfg.min_depth) & (g < cfg.max_depth)
            if cfg.clamp_pred:
                p = np.clip(
                    np.nan_to_num(p, nan=cfg.max_depth, posinf=cfg.max_depth, neginf=cfg.min_depth),
                    cfg.min_depth,
                    cfg.max_depth,
                )
            else:
                v &= np.isfinite(p)
            self._pixels.append((p[v], g[v]))
        return row

    def __len__(self) -> int:
        return len(self._rows)

    @property
    def per_sample(self) -> list[dict[str, float]]:
        """The per-sample rows, for dumping to CSV and for error analysis."""
        return list(self._rows)

    def compute(self) -> dict[str, float]:
        """Aggregate metrics over everything seen so far."""
        if not self._rows:
            return {name: float("nan") for name in METRIC_NAMES} | {
                "n_samples": 0,
                "n_skipped": self.n_skipped,
            }

        if self.pooling == "pixel":
            p = np.concatenate([x for x, _ in self._pixels])
            g = np.concatenate([y for _, y in self._pixels])
            # Re-use the single implementation by faking a 1 x N image.
            agg = compute_depth_metrics(
                p[None, :],
                g[None, :],
                config=MetricConfig(**{**asdict(self.config), "crop": "none"}),
            )
            agg = {k: v for k, v in agg.items() if k in METRIC_NAMES}
        else:
            agg = {name: float(np.mean([r[name] for r in self._rows])) for name in METRIC_NAMES}

        agg["n_samples"] = len(self._rows)
        agg["n_skipped"] = self.n_skipped
        return agg


def format_metrics(metrics: Mapping[str, float], names: Sequence[str] | None = None) -> str:
    """One-line human-readable rendering, e.g. for stdout at the end of a run."""
    names = names or [n for n in METRIC_NAMES if n in metrics]
    parts = []
    for n in names:
        v = metrics.get(n)
        parts.append(f"{n}={v:.4f}" if isinstance(v, (int | float)) else f"{n}=?")
    return "  ".join(parts)


def metric_columns(extra: Iterable[str] = ()) -> list[str]:
    """Canonical column order for benchmark tables."""
    return [*METRIC_NAMES, *extra]
