"""The one dataset interface every loader in this repo implements.

Why one interface
-----------------
We benchmark two families of methods on ten datasets.  The methods disagree
about what the input even *is*:

* **depth completion** (DEPTHOR / dToF): RGB + a *sparse* full-resolution depth
  map, zeros where the sensor returned nothing;
* **depth super-resolution** (DuCos): RGB + a *dense low-resolution* depth map.

And the datasets disagree too -- ZJU-L5 ships real 8x8 dToF zones, RGB-D-D
ships a real low-res ToF frame, Hypersim/TartanAir ship only perfect GT.
Every sample from every loader therefore carries **both** input forms, one
native and one derived by a documented rule, so an adapter never has to know
which dataset it is looking at.

Sample contract
---------------
``__getitem__`` returns a plain ``dict``:

===============  ==================  ==========================================
key              shape / dtype       meaning
===============  ==================  ==========================================
``rgb``          (H, W, 3) float32   RGB in [0, 1], aligned to ``gt_depth``
``gt_depth``     (H, W) float32      ground-truth depth in **metres**, 0 = none
``sparse_depth`` (H, W) float32      completion-style input, metres, 0 = none
``sparse_mask``  (H, W) bool         where ``sparse_depth`` has a measurement
``lr_depth``     (h, w) float32      SR-style input, dense, metres
``mask``         (H, W) bool         GT validity, the mask metrics must use
``meta``         dict                see below
===============  ==================  ==========================================

``meta`` always has: ``dataset``, ``sample_id``, ``split``, ``input_modality``
(``"sparse"`` | ``"lr"`` | ``"synthetic"``), ``lr_scale``, ``min_depth``,
``max_depth``, ``rgb_path``, ``depth_path``.  Loaders may add ``intrinsics``
(fx, fy, cx, cy), ``baseline_m``, ``scene``.

Derivation rules (applied in ``__getitem__``, never in an adapter)
------------------------------------------------------------------
* native sparse -> ``lr_depth`` is the *hole-filled* map downsampled by
  ``lr_scale``; ``lr_scale=1`` means "the sparse map, nearest-neighbour filled".
* native lr -> ``sparse_depth`` is ``lr_depth`` nearest-upsampled to (H, W).
  It is dense, not sparse; ``meta["sparse_is_derived"]`` is True so a
  completion method's result on such a dataset is reported as such.
* GT only -> a ``degradation`` object must be supplied; it produces both, seeded
  per sample so two machines get byte-identical inputs.
"""

from __future__ import annotations

import abc
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from utils.misc import REPO_ROOT, resize_depth, resize_rgb, sparse_to_dense_nn

try:  # torch is a hard dependency of the project, but loaders stay importable without it
    from torch.utils.data import Dataset as _TorchDataset
except Exception:  # pragma: no cover

    class _TorchDataset:  # type: ignore[no-redef]
        pass


__all__ = [
    "DATASET_REGISTRY",
    "SAMPLE_KEYS",
    "BaseDepthDataset",
    "DatasetInfo",
    "build_dataset",
    "register_dataset",
    "validate_sample",
]

SAMPLE_KEYS: tuple[str, ...] = (
    "rgb",
    "gt_depth",
    "sparse_depth",
    "sparse_mask",
    "lr_depth",
    "mask",
    "meta",
)

REQUIRED_META_KEYS: tuple[str, ...] = (
    "dataset",
    "sample_id",
    "split",
    "input_modality",
    "lr_scale",
    "min_depth",
    "max_depth",
)

SPLITS_DIR = REPO_ROOT / "data" / "splits"


@dataclass(frozen=True)
class DatasetInfo:
    """Static facts about a dataset.  Mirrors one row of ``docs/datasets.md``."""

    name: str
    modality: str  # "sparse" | "lr" | "gt_only"
    min_depth: float = 1e-3
    max_depth: float = 10.0
    depth_unit_per_metre: float = 1.0  # e.g. 1000.0 for uint16 millimetre PNGs
    native_size: tuple[int, int] | None = None  # (H, W) of the depth/GT frame
    sensor: str = ""
    url: str = ""
    notes: str = ""


DATASET_REGISTRY: dict[str, type[BaseDepthDataset]] = {}


def register_dataset(key: str):
    """Class decorator so ``dataset=nyuv2`` in a config resolves to a class."""

    def deco(cls):
        if key in DATASET_REGISTRY and DATASET_REGISTRY[key] is not cls:
            raise KeyError(f"dataset key {key!r} already registered to {DATASET_REGISTRY[key]}")
        DATASET_REGISTRY[key] = cls
        cls.key = key
        return cls

    return deco


class BaseDepthDataset(_TorchDataset, abc.ABC):
    """Base class for every dataset loader.

    Subclasses implement exactly two methods:

    ``_build_index()``
        Return a list of opaque per-sample records (usually dicts of paths).
        Must be deterministic and sorted, so splits are reproducible.

    ``_load_raw(record)``
        Return a dict with at least ``rgb`` (H, W, 3) and ``gt_depth`` (H, W)
        in metres, plus ``sparse_depth`` and/or ``lr_depth`` if the dataset
        ships a real degraded input, plus optional ``mask`` and ``meta``.

    Everything else -- unit conversion checks, resizing, deriving the missing
    input form, mask construction, split filtering -- happens here, once.
    """

    info: DatasetInfo = DatasetInfo(name="base", modality="gt_only")
    key: str = "base"

    def __init__(
        self,
        root: str | Path,
        split: str = "test",
        degradation: Any = None,
        split_file: str | Path | None = None,
        target_size: Sequence[int] | None = None,
        max_samples: int | None = None,
        min_depth: float | None = None,
        max_depth: float | None = None,
        lr_scale: int = 4,
        seed: int = 0,
        strict: bool = True,
        **kwargs: Any,
    ) -> None:
        self.root = Path(root).expanduser()
        self.split = split
        self.degradation = degradation
        self.target_size = tuple(target_size) if target_size else None
        self.lr_scale = int(lr_scale)
        self.seed = int(seed)
        self.strict = strict
        self.min_depth = float(min_depth if min_depth is not None else self.info.min_depth)
        self.max_depth = float(max_depth if max_depth is not None else self.info.max_depth)
        self.extra = kwargs

        if strict and not self.root.exists():
            raise FileNotFoundError(
                f"[{self.info.name}] dataset root does not exist: {self.root}\n"
                f"Run `python scripts/download_data.py --dataset {self.key}` and see "
                f"docs/datasets.md for the manual steps."
            )

        self.records = list(self._build_index())
        if split_file is not None:
            self.records = self._filter_by_split_file(self.records, split_file)
        if max_samples is not None:
            self.records = self.records[: int(max_samples)]
        if not self.records and strict:
            raise RuntimeError(
                f"[{self.info.name}] no samples found under {self.root} for split={split!r}. "
                "Check the directory layout in docs/datasets.md."
            )

        if self.info.modality == "gt_only" and self.degradation is None and strict:
            raise ValueError(
                f"[{self.info.name}] provides ground truth only, so it needs a degradation "
                "model to produce the network input. Pass degradation=... "
                "(configs/degradation/*.yaml)."
            )

    # ------------------------------------------------------------------ API

    @abc.abstractmethod
    def _build_index(self) -> Sequence[Any]:
        """Deterministic, sorted list of per-sample records."""

    @abc.abstractmethod
    def _load_raw(self, record: Any) -> Mapping[str, Any]:
        """Load one sample from disk.  See the class docstring for the contract."""

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        raw = dict(self._load_raw(self.records[idx]))
        rgb = np.asarray(raw["rgb"], dtype=np.float32)
        gt = np.asarray(raw["gt_depth"], dtype=np.float32)

        if rgb.ndim != 3 or rgb.shape[2] != 3:
            raise ValueError(f"[{self.info.name}] rgb must be (H, W, 3), got {rgb.shape}")
        if gt.ndim != 2:
            raise ValueError(f"[{self.info.name}] gt_depth must be (H, W), got {gt.shape}")
        if rgb.max() > 1.5:  # a loader forgot to divide by 255
            raise ValueError(
                f"[{self.info.name}] rgb must be scaled to [0, 1], got max={rgb.max():.1f}"
            )

        sparse = raw.get("sparse_depth")
        lr = raw.get("lr_depth")

        if self.target_size is not None:
            rgb = resize_rgb(rgb, self.target_size)
            gt = resize_depth(gt, self.target_size, mode="nearest")
            if sparse is not None:
                sparse = resize_depth(np.asarray(sparse, np.float32), self.target_size, "nearest")
            lr = None if lr is None else lr  # regenerated below at the new scale

        if rgb.shape[:2] != gt.shape[:2]:
            raise ValueError(
                f"[{self.info.name}] rgb {rgb.shape[:2]} and gt_depth {gt.shape[:2]} must be "
                "aligned and the same size; do the registration inside the loader."
            )

        h, w = gt.shape
        modality = self.info.modality

        # ---- produce the input(s) --------------------------------------
        if sparse is None and lr is None:
            if self.degradation is None:
                raise RuntimeError(
                    f"[{self.info.name}] sample {idx} has no degraded input and no degradation "
                    "model was configured."
                )
            deg = self.degradation(gt, seed=self.seed * 1_000_003 + idx)
            sparse = deg["sparse_depth"]
            lr = deg.get("lr_depth")
            modality = "synthetic"

        if sparse is None:  # native LR -> derive the completion-style input
            lr = np.asarray(lr, dtype=np.float32)
            sparse = resize_depth(lr, (h, w), mode="nearest")
            sparse_is_derived = True
        else:
            sparse = np.asarray(sparse, dtype=np.float32)
            sparse_is_derived = False

        if lr is None:  # native sparse -> derive the SR-style input
            filled = sparse_to_dense_nn(sparse)
            lr = (
                filled
                if self.lr_scale == 1
                else resize_depth(filled, (max(1, h // self.lr_scale), max(1, w // self.lr_scale)))
            )
            lr_is_derived = True
        else:
            lr = np.asarray(lr, dtype=np.float32)
            lr_is_derived = False

        sparse = np.nan_to_num(sparse, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
        lr = np.nan_to_num(lr, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
        gt = np.nan_to_num(gt, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

        # ---- masks ------------------------------------------------------
        mask = np.isfinite(gt) & (gt > self.min_depth) & (gt < self.max_depth)
        extra_mask = raw.get("mask")
        if extra_mask is not None:
            extra_mask = np.asarray(extra_mask, dtype=bool)
            if self.target_size is not None:
                extra_mask = resize_depth(extra_mask.astype(np.float32), self.target_size) > 0.5
            mask &= extra_mask
        sparse_mask = sparse > 0

        meta: dict[str, Any] = {
            "dataset": self.info.name,
            "sample_id": str(raw.get("sample_id", idx)),
            "split": self.split,
            "input_modality": modality,
            "lr_scale": round(h / lr.shape[0]) if lr.shape[0] else 1,
            "sparse_is_derived": bool(sparse_is_derived),
            "lr_is_derived": bool(lr_is_derived),
            "sparsity": float(1.0 - sparse_mask.mean()),
            "min_depth": self.min_depth,
            "max_depth": self.max_depth,
            "rgb_path": str(raw.get("rgb_path", "")),
            "depth_path": str(raw.get("depth_path", "")),
        }
        meta.update(raw.get("meta", {}) or {})

        return {
            "rgb": rgb,
            "gt_depth": gt,
            "sparse_depth": sparse,
            "sparse_mask": sparse_mask,
            "lr_depth": lr,
            "mask": mask,
            "meta": meta,
        }

    # ------------------------------------------------------------- splits

    def _filter_by_split_file(self, records: Sequence[Any], split_file: str | Path) -> list[Any]:
        """Keep only the records whose id appears in a committed split file.

        Split files live in ``data/splits/<dataset>/<split>.json`` and are the
        single source of truth for train/val/test membership -- they are small,
        they are in git, and they are what makes two teammates' numbers
        comparable.  Format: ``{"dataset": ..., "split": ..., "ids": [...]}``.
        """
        path = Path(split_file)
        if not path.is_absolute():
            candidate = SPLITS_DIR / path
            path = candidate if candidate.exists() else path
        if not path.exists():
            raise FileNotFoundError(f"[{self.info.name}] split file not found: {path}")
        payload = json.loads(path.read_text(encoding="utf-8"))
        wanted = list(payload["ids"] if isinstance(payload, dict) else payload)
        order = {sid: i for i, sid in enumerate(wanted)}
        keep = [r for r in records if self.record_id(r) in order]
        missing = set(wanted) - {self.record_id(r) for r in keep}
        if missing and self.strict:
            raise RuntimeError(
                f"[{self.info.name}] split {path.name} lists {len(missing)} ids that are not on "
                f"disk, e.g. {sorted(missing)[:3]}. The local copy of the dataset is incomplete."
            )
        return sorted(keep, key=lambda r: order[self.record_id(r)])

    @staticmethod
    def record_id(record: Any) -> str:
        """Stable id for a record; override if a record is not a dict with ``id``."""
        if isinstance(record, Mapping) and "id" in record:
            return str(record["id"])
        return str(record)

    def write_split_file(self, path: str | Path, ids: Sequence[str] | None = None) -> Path:
        """Freeze the current index (or ``ids``) into a committed split file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "dataset": self.info.name,
            "split": self.split,
            "ids": list(ids) if ids is not None else [self.record_id(r) for r in self.records],
        }
        path.write_text(json.dumps(payload, indent=1), encoding="utf-8")
        return path


def validate_sample(sample: Mapping[str, Any], name: str = "sample") -> None:
    """Assert the sample contract.  Used by ``tests/test_loaders.py``.

    Raises ``AssertionError`` with a message that says what to fix, so a loader
    written by a teammate fails loudly at the smoke test rather than quietly
    producing wrong metrics three weeks later.
    """
    missing = [k for k in SAMPLE_KEYS if k not in sample]
    assert not missing, f"{name}: missing keys {missing}"

    rgb, gt = sample["rgb"], sample["gt_depth"]
    sparse, lr = sample["sparse_depth"], sample["lr_depth"]
    mask, smask, meta = sample["mask"], sample["sparse_mask"], sample["meta"]

    assert rgb.dtype == np.float32, f"{name}: rgb dtype {rgb.dtype}, expected float32"
    assert rgb.ndim == 3 and rgb.shape[2] == 3, f"{name}: rgb shape {rgb.shape}"
    assert float(rgb.min()) >= 0.0 and float(rgb.max()) <= 1.0 + 1e-4, f"{name}: rgb not in [0,1]"

    assert gt.dtype == np.float32, f"{name}: gt_depth dtype {gt.dtype}"
    assert gt.ndim == 2, f"{name}: gt_depth shape {gt.shape}"
    assert gt.shape == rgb.shape[:2], f"{name}: gt {gt.shape} vs rgb {rgb.shape[:2]}"
    assert np.isfinite(gt).all(), f"{name}: gt_depth has NaN/Inf; replace them with 0"

    assert sparse.shape == gt.shape, f"{name}: sparse_depth {sparse.shape} vs gt {gt.shape}"
    assert smask.shape == gt.shape and smask.dtype == bool, f"{name}: sparse_mask malformed"
    assert bool((smask == (sparse > 0)).all()), f"{name}: sparse_mask disagrees with sparse_depth"
    assert lr.ndim == 2, f"{name}: lr_depth shape {lr.shape}"
    assert mask.shape == gt.shape and mask.dtype == bool, f"{name}: mask malformed"

    for k in REQUIRED_META_KEYS:
        assert k in meta, f"{name}: meta is missing {k!r}"
    assert float(meta["min_depth"]) < float(meta["max_depth"]), f"{name}: bad depth range in meta"

    valid = gt[mask]
    if valid.size:
        assert float(valid.min()) > 0, f"{name}: masked-in gt contains non-positive depth"


def build_dataset(cfg: Mapping[str, Any], degradation: Any = None, **overrides: Any):
    """Instantiate a loader from a ``configs/dataset/*.yaml`` mapping."""
    cfg = dict(cfg)
    key = cfg.pop("loader", cfg.get("name"))
    if key not in DATASET_REGISTRY:
        # import for side effects so every @register_dataset has run
        import data.loaders  # noqa: F401

    if key not in DATASET_REGISTRY:
        raise KeyError(
            f"unknown dataset {key!r}; registered: {sorted(DATASET_REGISTRY)}. "
            "Add a loader in data/loaders/ and decorate it with @register_dataset."
        )
    cls = DATASET_REGISTRY[key]
    # documentation-only keys that live in the YAML but are not loader arguments
    for meta_key in ("name", "docs", "download", "eval", "notes", "paper", "url", "sensor"):
        cfg.pop(meta_key, None)
    cfg.update(overrides)
    return cls(degradation=degradation, **cfg)
