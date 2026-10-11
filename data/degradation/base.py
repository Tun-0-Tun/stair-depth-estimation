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
from typing import Any, ClassVar

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

    uses_normals: ClassVar[bool] = False
    """Subclasses that need a normal map set this; ``apply`` then gets ``normal=``."""

    def __call__(
        self,
        gt_depth: np.ndarray,
        normal: np.ndarray | None = None,
        seed: int | None = None,
        *,
        normal_estimator: Any = None,
        force_normals: bool = False,
    ) -> dict[str, Any]:
        """Degrade ``gt_depth``.

        For degradations with ``uses_normals`` the normal map is resolved here, once:
        the dataset's ``normal`` (H, W, 3) is used if given; otherwise it is computed from
        ``gt_depth`` with ``normal_estimator`` (a ``data.norm`` method); with
        ``force_normals`` the estimator is used even if ``normal`` is given.  The map
        used is returned as ``out["normals"]``.  Other degradations ignore all three.
        """
        # Set ground trueth depth
        gt = np.asarray(gt_depth, dtype=np.float32)
        if gt.ndim != 2:
            raise ValueError(f"gt_depth must be (H, W), got {gt.shape}")

        # Set randomizer
        rng = np.random.default_rng(seed)

        # Calculate result
        normal_meta: dict[str, Any] = {}
        if self.uses_normals:
            normal, normal_meta["normal_source"] = _resolve_normals(
                gt, normal, normal_estimator, force_normals
            )
            out = dict(self.apply(gt, normal, rng))
            out["normals"] = normal
        else:
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
        meta.update(normal_meta)
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


def _resolve_normals(
    gt: np.ndarray, normal: np.ndarray | None, estimator: Any, force: bool
) -> tuple[np.ndarray, str]:
    """Pick the normal map: the dataset's, else the estimator's (always, if ``force``)."""
    if normal is not None and not force:
        return np.asarray(normal), "provided"
    if estimator is None:
        raise ValueError(
            "this degradation needs a normal map: the dataset provides none"
            + (" (and --force-normals asks for estimated ones)" if force else "")
            + ", and no normal estimator was given (a configs/normals preset, e.g. --normals gt_cross)"
        )
    # BaseNormalMap.__call__ resolves intrinsics, cleans the depth and masks invalid normals
    return estimator(gt)["normals"], f"estimated:{type(estimator).__name__}"


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
