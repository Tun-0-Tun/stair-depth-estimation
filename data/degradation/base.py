"""Degradation models: turn clean ground-truth depth into a realistic sensor input.

Two uses:

1. **Synthetic training / evaluation.**  Hypersim, TartanAir and NYUv2 give us
   dense clean depth.  To train or evaluate a completion method we must first
   simulate the sensor.  DEPTHOR's central claim is precisely that the realism
   of this simulation is what determines real-world accuracy, so this package
   is a first-class part of the project, not a utility.
2. **Domain matching for the Astra 2.**  Our target sensor is an active-stereo
   camera whose error statistics differ from a dToF's.  Training a baseline on
   dToF-simulated data and deploying on the Astra 2 is a domain gap we have to
   measure -- ``active_stereo_astra2.py`` is the model of *our* sensor.

Contract
--------
``degradation(gt_depth, seed=None) -> {"sparse_depth", "lr_depth", "meta"}``

* ``gt_depth``: (H, W) float32, metres, 0 = invalid.
* ``sparse_depth``: (H, W) float32, metres, 0 = no measurement.
* ``lr_depth``: (h, w) float32 dense, or ``None`` if the model is not an SR-style
  degradation (the loader then derives it).
* ``meta``: parameters actually used, for the run record.

Every model is **seeded per sample**, so a given (dataset, sample index, seed)
produces identical input on every machine.  Never call ``np.random`` directly.
"""

from __future__ import annotations

import abc
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

__all__ = [
    "DEGRADATION_REGISTRY",
    "BaseDegradation",
    "IdentityDegradation",
    "build_degradation",
    "register_degradation",
]

DEGRADATION_REGISTRY: dict[str, type[BaseDegradation]] = {}


def register_degradation(key: str):
    def deco(cls):
        if key in DEGRADATION_REGISTRY and DEGRADATION_REGISTRY[key] is not cls:
            raise KeyError(f"degradation {key!r} already registered")
        DEGRADATION_REGISTRY[key] = cls
        cls.key = key
        return cls

    return deco


@dataclass
class BaseDegradation(abc.ABC):
    """Base class.  Subclasses are dataclasses holding their parameters."""

    key: str = "base"

    def __call__(self, gt_depth: np.ndarray, seed: int | None = None) -> dict[str, Any]:
        gt = np.asarray(gt_depth, dtype=np.float32)
        if gt.ndim != 2:
            raise ValueError(f"gt_depth must be (H, W), got {gt.shape}")
        rng = np.random.default_rng(seed)
        out = dict(self.apply(gt, rng))
        sparse = np.asarray(out.get("sparse_depth"), dtype=np.float32)
        sparse = np.nan_to_num(sparse, nan=0.0, posinf=0.0, neginf=0.0)
        sparse[sparse < 0] = 0.0
        out["sparse_depth"] = sparse
        lr = out.get("lr_depth")
        if lr is not None:
            out["lr_depth"] = np.nan_to_num(
                np.asarray(lr, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0
            )
        meta = {"degradation": type(self).__name__, "seed": seed}
        meta.update(self.params())
        meta.update(out.get("meta", {}))
        out["meta"] = meta
        return out

    @abc.abstractmethod
    def apply(self, gt: np.ndarray, rng: np.random.Generator) -> Mapping[str, Any]:
        """Do the work.  Use ``rng`` for *all* randomness."""

    def params(self) -> dict[str, Any]:
        try:
            return {k: v for k, v in asdict(self).items() if k != "key"}
        except TypeError:  # not a dataclass
            return {}


@register_degradation("identity")
@dataclass
class IdentityDegradation(BaseDegradation):
    """No degradation: the GT itself becomes the input.

    Only useful as a sanity check -- every metric should come out perfect, which
    is exactly what ``tests/test_metrics.py`` and the pipeline smoke test rely on.
    """

    key: str = "identity"

    def apply(self, gt: np.ndarray, rng: np.random.Generator) -> Mapping[str, Any]:
        return {"sparse_depth": gt.copy(), "lr_depth": gt.copy()}


def build_degradation(cfg: Mapping[str, Any] | None) -> BaseDegradation | None:
    """Instantiate from a ``configs/degradation/*.yaml`` mapping (or ``None``)."""
    if cfg is None:
        return None
    cfg = dict(cfg)
    key = cfg.pop("type", cfg.pop("name", None))
    if key in (None, "none", "null"):
        return None
    if key not in DEGRADATION_REGISTRY:
        import data.degradation  # noqa: F401  (populate the registry)
    if key not in DEGRADATION_REGISTRY:
        raise KeyError(f"unknown degradation {key!r}; registered: {sorted(DEGRADATION_REGISTRY)}")
    cfg.pop("name", None)
    cfg.pop("docs", None)
    return DEGRADATION_REGISTRY[key](**cfg)
