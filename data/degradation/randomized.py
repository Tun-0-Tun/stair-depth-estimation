"""Training-time randomisation of a sensor chain: pick a strength level, jitter its parameters, per sample.

``levels`` are sequential presets (``configs/degradation/<name>.yaml``): one per strength level, e.g. the main chain now and
the two further strength levels later -- add their names to the list, no code change.  ``jitter`` multiplies numeric
parameters of a step by a log-uniform factor from [lo, hi]: ``{step_preset: {field: [lo, hi]}}`` (a list field, e.g. the
projector offset, is scaled as a whole).  With probability ``1 - jitter_prob`` the level is applied unjittered, so the exact
test chain stays in the training distribution.  The scale (x8) is never jittered: the model has a fixed scale.
"""

from __future__ import annotations

import copy
import dataclasses
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from data.degradation.base import BaseDegradation, build_degradation, register_degradation
from data.degradation.sequential import SequentialDegradation
from utils.config import load_config

__all__ = ["RandomSequentialDegradation"]


@register_degradation("random_sequential")
@dataclass
class RandomSequentialDegradation(BaseDegradation):
    levels: list[str] = field(default_factory=list)
    weights: list[float] | None = None
    jitter: dict[str, dict[str, list[float]]] = field(default_factory=dict)
    jitter_prob: float = 0.7
    key: str = "random_sequential"

    def __post_init__(self) -> None:
        if not self.levels:
            raise ValueError("random_sequential needs a non-empty `levels` list of sequential presets")
        self._levels = []
        for name in self.levels:
            seq = build_degradation(dict(load_config("degradation", name)))
            if not isinstance(seq, SequentialDegradation):
                raise ValueError(f"level {name!r} must be a sequential preset")
            self._levels.append(seq)
        w = np.ones(len(self.levels)) if self.weights is None else np.asarray(self.weights, float)
        if len(w) != len(self.levels) or (w < 0).any() or w.sum() <= 0:
            raise ValueError("weights must match levels and be non-negative")
        self._p = w / w.sum()
        for spec in self.jitter.values():
            for lo, hi in spec.values():
                if not 0 < lo <= hi:
                    raise ValueError("jitter ranges are multiplicative factors 0 < lo <= hi")
        self.uses_normals = any(s.uses_normals for s in self._levels)

    def apply(self, gt: np.ndarray, *args: Any) -> dict[str, Any]:
        rng = args[-1]
        i = int(rng.choice(len(self._levels), p=self._p))
        seq = copy.copy(self._levels[i])
        factors: dict[str, float] = {}
        if rng.random() < self.jitter_prob:
            steps = []
            for name, step in zip(seq.steps, seq._steps, strict=True):
                upd = {}
                for f, (lo, hi) in self.jitter.get(name, {}).items():
                    k = float(np.exp(rng.uniform(np.log(lo), np.log(hi))))
                    v = getattr(step, f)
                    upd[f] = [x * k for x in v] if isinstance(v, list | tuple) else v * k
                    factors[f"{name}.{f}"] = k
                steps.append(dataclasses.replace(step, **upd) if upd else step)
            seq._steps = steps
        out = dict(seq.apply(gt, *args))
        out["meta"] = {**out["meta"], "level": self.levels[i], "jitter": factors}
        return out
