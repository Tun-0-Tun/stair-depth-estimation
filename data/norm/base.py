"""Normal-map construction: turn ground-truth depth into per-pixel surface normals.

Why
---
Stairs are told apart by surface *orientation*: treads face up, risers face the
viewer, and the two meet at a 90-degree crease (the nosing).  Depth says how far
a surface is; a normal says which way it faces.  A normal map built from the GT
depth is therefore a reference for geometry-aware losses and for diagnosing
where a method's error lives -- something AbsRel/RMSE hide.

Contract
--------
``normal_map(gt_depth, rgb=None, intrinsics=None) -> {"normals", "normal_mask", "meta"}``

* ``gt_depth``: (H, W) float32, **metres**, 0 = no measurement.
* ``rgb``: (H, W, 3) float32 in [0, 1], optional.  Geometry-only methods ignore
  it; the slot exists so RGB-guided methods share the interface.
* ``intrinsics``: optional ``(fx, fy, cx, cy)`` in pixels at the working
  resolution (e.g. ``sample["meta"]["intrinsics"]``).  Overrides the method's own
  fields.
* ``normals``: (H, W, 3) float32 unit vectors in the camera frame (x right,
  y down, z forward -- the same frame as ``projector_shadow``); zero where invalid.
* ``normal_mask``: (H, W) bool, where ``normals`` is trustworthy.
* ``meta``: parameters actually used, for the run record.

Methods are deterministic: there is no randomness to seed.
"""

from __future__ import annotations

import abc
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

__all__ = [
    "NORMAL_REGISTRY",
    "BaseNormalMap",
    "ConstantNormalMap",
    "build_normal_map",
    "register_normal",
]

NORMAL_REGISTRY: dict[str, type[BaseNormalMap]] = {}


def register_normal(key: str):
    def deco(cls):
        if key in NORMAL_REGISTRY and NORMAL_REGISTRY[key] is not cls:
            raise KeyError(f"normal map {key!r} already registered")
        NORMAL_REGISTRY[key] = cls
        cls.key = key
        return cls

    return deco


@dataclass
class BaseNormalMap(abc.ABC):
    """Base class.  Subclasses are dataclasses holding their parameters.

    Subclasses that need a pinhole model declare ``fx, fy, cx, cy`` fields (pixels);
    ``None`` for ``fy`` means ``fx``, ``None`` for ``cx``/``cy`` means the image centre.
    """

    key: str = "base"

    def __call__(
        self,
        gt_depth: np.ndarray,
        rgb: np.ndarray | None = None,
        intrinsics: Sequence[float] | None = None,
    ) -> dict[str, Any]:
        gt = np.asarray(gt_depth, dtype=np.float32)
        if gt.ndim != 2:
            raise ValueError(f"gt_depth must be (H, W), got {gt.shape}")
        gt = np.nan_to_num(gt, nan=0.0, posinf=0.0, neginf=0.0)
        gt[gt < 0] = 0.0
        if rgb is not None:
            rgb = np.asarray(rgb, dtype=np.float32)
            if rgb.shape != (*gt.shape, 3):
                raise ValueError(f"rgb must be {(*gt.shape, 3)} to match gt_depth, got {rgb.shape}")

        k, source = self._resolve_intrinsics(gt.shape, intrinsics)
        out = dict(self.estimate(gt, rgb, k))

        mask = np.asarray(out["normal_mask"], dtype=bool)
        normals = np.nan_to_num(np.asarray(out["normals"], dtype=np.float32))
        norm = np.linalg.norm(normals, axis=-1, keepdims=True)
        mask = mask & (norm[..., 0] > 1e-8)
        normals = np.where(mask[..., None], normals / np.maximum(norm, 1e-8), 0.0).astype(
            np.float32
        )

        meta = {"normal_map": type(self).__name__, "intrinsics_source": source}
        meta.update(self.params())
        meta.update(out.get("meta", {}))
        return {"normals": normals, "normal_mask": mask, "meta": meta}

    @abc.abstractmethod
    def estimate(
        self, gt: np.ndarray, rgb: np.ndarray | None, k: tuple[float, float, float, float]
    ) -> Mapping[str, Any]:
        """Do the work.  Return ``normals`` (H, W, 3) and ``normal_mask`` (H, W).

        ``k`` is the resolved ``(fx, fy, cx, cy)`` in pixels.  The base class
        re-normalises the vectors and zeroes them where the mask is False.
        """

    def params(self) -> dict[str, Any]:
        try:
            return {k: v for k, v in asdict(self).items() if k != "key"}
        except TypeError:  # not a dataclass
            return {}

    def _resolve_intrinsics(
        self, shape: tuple[int, int], intrinsics: Sequence[float] | None
    ) -> tuple[tuple[float, float, float, float], str]:
        h, w = shape
        if intrinsics is not None:
            fx, fy, cx, cy = (float(v) for v in intrinsics)
            return (fx, fy, cx, cy), "call"
        fx = getattr(self, "fx", None)
        if fx is None:
            return (1.0, 1.0, (w - 1) / 2.0, (h - 1) / 2.0), "none"
        fy = getattr(self, "fy", None)
        cx = getattr(self, "cx", None)
        cy = getattr(self, "cy", None)
        return (
            float(fx),
            float(fx if fy is None else fy),
            (w - 1) / 2.0 if cx is None else float(cx),
            (h - 1) / 2.0 if cy is None else float(cy),
        ), "config"


@register_normal("constant")
@dataclass
class ConstantNormalMap(BaseNormalMap):
    """Every measured pixel gets the same normal.

    The normal-map analogue of ``IdentityDegradation``: a known-answer fixture for
    tests and the "dumb" lower bound for any later normal-quality comparison.
    """

    key: str = "constant"

    normal: tuple[float, float, float] = (0.0, 0.0, -1.0)
    """Camera frame (x right, y down, z forward); default faces the camera.  Normalised on init."""

    def __post_init__(self) -> None:
        n = np.asarray(self.normal, dtype=np.float64)
        if n.shape != (3,) or not np.isfinite(n).all() or np.linalg.norm(n) < 1e-8:
            raise ValueError(f"normal must be a finite non-zero 3-vector, got {self.normal!r}")
        self.normal = tuple(float(v) for v in n / np.linalg.norm(n))

    def estimate(self, gt, rgb, k):
        normals = np.broadcast_to(np.asarray(self.normal, dtype=np.float32), (*gt.shape, 3))
        return {"normals": normals, "normal_mask": gt > 0}


def build_normal_map(cfg: Mapping[str, Any] | None) -> BaseNormalMap | None:
    """Instantiate from a ``configs/normals/*.yaml`` mapping (or ``None``)."""
    if cfg is None:
        return None
    cfg = dict(cfg)
    key = cfg.pop("type", cfg.pop("name", None))
    if key in (None, "none", "null"):
        return None
    if key not in NORMAL_REGISTRY:
        import data.norm  # noqa: F401  (populate the registry)
    if key not in NORMAL_REGISTRY:
        raise KeyError(f"unknown normal map {key!r}; registered: {sorted(NORMAL_REGISTRY)}")
    cfg.pop("name", None)
    cfg.pop("docs", None)
    return NORMAL_REGISTRY[key](**cfg)
