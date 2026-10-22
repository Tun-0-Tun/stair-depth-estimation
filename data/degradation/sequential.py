"""Sequential degradation: apply several sensor presets one after another.

Each name in ``steps`` is a preset in ``configs/degradation/<name>.yaml``; its ``type:``
picks the class and its other keys are the constructor arguments.  Step 1 sees the clean
GT.  Every later step sees the previous step's ``sparse_depth`` (0 = no measurement).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from data.degradation.base import (
    DEGRADATION_REGISTRY,
    BaseDegradation,
    build_degradation,
    register_degradation,
)
from utils.config import CONFIG_ROOT, load_config

__all__ = ["SequentialDegradation"]


@register_degradation("sequential")
@dataclass
class SequentialDegradation(BaseDegradation):
    """Chain of presets; ``steps`` are preset names, applied in order."""

    steps: list[str] = field(default_factory=list)
    key: str = "sequential"

    def __post_init__(self) -> None:
        if not isinstance(self.steps, list | tuple) or not self.steps:
            raise ValueError("sequential degradation needs a non-empty list of preset names")
        self.steps = list(self.steps)
        self._steps = [self._build_step(name) for name in self.steps]
        # a sequence needs a normal map only if one of its presets does
        self.uses_normals = any(step.uses_normals for step in self._steps)

    @staticmethod
    def _build_step(name: Any) -> BaseDegradation:
        if not isinstance(name, str):
            raise ValueError(f"step names must be strings, got {name!r}")
        preset_dir = CONFIG_ROOT / "degradation"
        if not (preset_dir / f"{name}.yaml").is_file():
            available = sorted(p.stem for p in preset_dir.glob("*.yaml"))
            raise KeyError(f"no degradation preset {name!r}; available: {available}")

        cfg = dict(load_config("degradation", name))
        kind = cfg.get("type")
        if kind in (None, "none", "null"):
            raise ValueError(f"preset {name!r} is a no-op; a sequence cannot contain it")
        if kind == SequentialDegradation.key:
            raise ValueError(f"preset {name!r} is itself sequential; nesting is not allowed")
        if kind not in DEGRADATION_REGISTRY:
            raise KeyError(f"preset {name!r} has unknown type {kind!r}")
        return build_degradation(cfg)

    def apply(self, gt: np.ndarray, *args: Any) -> dict[str, Any]:
        """``apply(gt, rng)``, or ``apply(gt, normal, rng)`` when ``uses_normals``."""
        rng = args[-1]
        normal = args[0] if len(args) == 2 else None

        prev = gt
        meta: dict[str, Any] = {"steps": list(self.steps)}
        calibrated = True
        for i, (name, step) in enumerate(zip(self.steps, self._steps, strict=False)):
            kwargs = {"normal": normal} if step.uses_normals else {}
            # one seed per step, drawn from the sample rng: reproducible per sample
            seed = int(rng.integers(2**31 - 1))
            out = step(prev, seed=seed, **kwargs)
            prev = out["lr_depth"]  # out["sparse_depth"]
            meta[f"step{i}_{name}"] = out["meta"]
            calibrated = calibrated and out["meta"].get("calibrated") is not False
        meta["calibrated"] = calibrated

        return {"sparse_depth": prev, "lr_depth": out.get("lr_depth"), "meta": meta}
