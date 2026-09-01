"""The single interface every baseline is wrapped in.

Design rule
-----------
``third_party/*`` is vendored **verbatim as a git submodule** and never edited.
Every difference between a paper's code and our pipeline -- input resolution,
tensor layout, unit convention, per-sample normalisation, hard-coded checkpoint
paths -- is absorbed by an adapter in this package.  Experiment scripts see one
interface and one unit system (metres), so ``scripts/run_benchmark.py`` contains
no ``if model == ...`` branches.

That is not tidiness for its own sake.  Concretely, the two baselines disagree
about almost everything:

============  =====================================  ==============================
              DEPTHOR                                DuCos
============  =====================================  ==============================
input         RGB + sparse dToF points, full res     RGB + dense LR depth
units         metres, absolute                       per-sample min-max [0, 1]
resolution    fixed 480x640                          arbitrary, multiple of 14 inside
call          ``model({'image', 'sparse_depth'})``   ``model({'rgb', 'dep'})``
output        ``(coarse, final)`` tuple              ``{'pred', ...}`` dict
extra deps    ``BpOps`` CUDA extension               none beyond torch
============  =====================================  ==============================

Availability
------------
Both adapters need weights that we cannot redistribute, and DEPTHOR needs a
CUDA extension.  Rather than failing deep inside a forward pass, each adapter
implements :meth:`BaselineModel.availability`, which returns a structured report
that ``scripts/run_baseline.py`` prints as actionable instructions.
"""

from __future__ import annotations

import abc
import sys
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from utils.misc import REPO_ROOT, resolve_device

__all__ = [
    "BASELINE_REGISTRY",
    "Availability",
    "BaselineModel",
    "add_third_party_to_path",
    "build_baseline",
    "register_baseline",
]

THIRD_PARTY = REPO_ROOT / "third_party"

BASELINE_REGISTRY: dict[str, type[BaselineModel]] = {}


def register_baseline(key: str):
    def deco(cls):
        if key in BASELINE_REGISTRY and BASELINE_REGISTRY[key] is not cls:
            raise KeyError(f"baseline {key!r} already registered")
        BASELINE_REGISTRY[key] = cls
        cls.key = key
        return cls

    return deco


@dataclass
class Availability:
    """Can this baseline actually run here, and if not, what does the user do?"""

    ok: bool
    reasons: list[str] = field(default_factory=list)
    instructions: list[str] = field(default_factory=list)

    def report(self, name: str) -> str:
        if self.ok:
            return f"[{name}] ready"
        lines = [f"[{name}] cannot run:"]
        lines += [f"  - {r}" for r in self.reasons]
        if self.instructions:
            lines.append("  fix:")
            lines += [f"    * {i}" for i in self.instructions]
        return "\n".join(lines)


class BaselineModel(abc.ABC):
    """Wrap an upstream method behind ``load()`` + ``predict()``.

    Subclasses set the class attributes below and implement ``_load`` and
    ``_predict``.  ``predict`` itself does the shared bookkeeping (shape checks,
    unit checks, output validation) so an adapter bug surfaces immediately.
    """

    key: str = "base"
    #: ``"sparse"`` (completion: full-resolution map with zeros) or ``"lr"``
    #: (super-resolution: dense low-resolution map).  Decides which field of a
    #: loader sample ``predict_sample`` feeds in.
    input_modality: str = "sparse"
    #: Native input size the upstream network was trained at, or ``None``.
    native_size: tuple[int, int] | None = None
    #: Human-readable citation, printed in the results table.
    paper: str = ""
    submodule: str | None = None

    def __init__(
        self,
        device: str = "auto",
        min_depth: float = 1e-3,
        max_depth: float = 10.0,
        **kwargs: Any,
    ) -> None:
        self.device = resolve_device(device)
        self.min_depth = float(min_depth)
        self.max_depth = float(max_depth)
        self.options = kwargs
        self.model: Any = None
        self.checkpoint_path: Path | None = None

    # ------------------------------------------------------------- hooks

    @classmethod
    def availability(cls, checkpoint_path: str | Path | None = None, **kwargs: Any) -> Availability:
        """Cheap check that does not import torch or touch the GPU."""
        return Availability(ok=True)

    @abc.abstractmethod
    def _load(self, checkpoint_path: Path | None) -> Any:
        """Build the upstream model and load weights.  Return the model object."""

    @abc.abstractmethod
    def _predict(self, rgb: np.ndarray, depth_in: np.ndarray, **kwargs: Any) -> np.ndarray:
        """Run one forward pass.  ``depth_in`` is already in metres."""

    # -------------------------------------------------------------- API

    def load(self, checkpoint_path: str | Path | None = None) -> BaselineModel:
        avail = self.availability(checkpoint_path, **self.options)
        if not avail.ok:
            raise RuntimeError(avail.report(self.key))
        self.checkpoint_path = Path(checkpoint_path) if checkpoint_path else None
        self.model = self._load(self.checkpoint_path)
        return self

    def predict(self, rgb: np.ndarray, lr_depth: np.ndarray, **kwargs: Any) -> np.ndarray:
        """Dense depth in metres, ``(H, W)`` float32, same size as ``rgb``.

        ``lr_depth`` is the degraded depth input: a sparse full-resolution map
        for a completion method, a dense low-resolution map for an SR method --
        whichever this adapter declares in ``input_modality``.
        """
        if self.model is None:
            raise RuntimeError(f"[{self.key}] call load() before predict()")

        rgb = np.asarray(rgb, dtype=np.float32)
        depth_in = np.asarray(lr_depth, dtype=np.float32)
        if rgb.ndim != 3 or rgb.shape[2] != 3:
            raise ValueError(f"[{self.key}] rgb must be (H, W, 3), got {rgb.shape}")
        if rgb.max() > 1.5:
            raise ValueError(f"[{self.key}] rgb must be in [0, 1], got max={rgb.max():.2f}")
        if depth_in.ndim != 2:
            raise ValueError(f"[{self.key}] depth input must be (H, W), got {depth_in.shape}")

        out = self._predict(rgb, depth_in, **kwargs)
        out = np.asarray(out, dtype=np.float32)
        if out.ndim == 3:
            out = out.squeeze()
        if out.shape != rgb.shape[:2]:
            raise ValueError(
                f"[{self.key}] prediction {out.shape} does not match the RGB frame "
                f"{rgb.shape[:2]}; the adapter must resize back to the input resolution."
            )
        return out

    def predict_sample(self, sample: Mapping[str, Any], **kwargs: Any) -> np.ndarray:
        """Run on a loader sample, feeding whichever input this method wants."""
        depth_in = sample["lr_depth"] if self.input_modality == "lr" else sample["sparse_depth"]
        return self.predict(sample["rgb"], depth_in, sample=sample, **kwargs)

    def describe(self) -> dict[str, Any]:
        return {
            "method": self.key,
            "paper": self.paper,
            "input_modality": self.input_modality,
            "device": self.device,
            "checkpoint": str(self.checkpoint_path) if self.checkpoint_path else "",
        }


def add_third_party_to_path(name: str) -> Path:
    """Put ``third_party/<name>`` on ``sys.path`` and check the submodule is cloned."""
    path = THIRD_PARTY / name
    if not path.exists() or not any(path.iterdir()):
        raise RuntimeError(
            f"third_party/{name} is empty. Run:\n" "    git submodule update --init --recursive"
        )
    p = str(path)
    if p not in sys.path:
        sys.path.insert(0, p)
    return path


def build_baseline(cfg: Mapping[str, Any], **overrides: Any) -> BaselineModel:
    """Instantiate from a ``configs/model/*.yaml`` mapping (without loading weights)."""
    cfg = dict(cfg)
    key = cfg.pop("adapter", cfg.get("name"))
    if key not in BASELINE_REGISTRY:
        # populate the registry: adapters over third_party/, plus our own method
        import baselines  # noqa: F401
        import models  # noqa: F401
    if key not in BASELINE_REGISTRY:
        raise KeyError(
            f"unknown baseline {key!r}; registered: {sorted(BASELINE_REGISTRY)}. "
            "Add an adapter in baselines/ and decorate it with @register_baseline."
        )
    for drop in ("name", "docs", "checkpoint", "available", "unavailable_reason", "weights"):
        cfg.pop(drop, None)
    cfg.update(overrides)
    return BASELINE_REGISTRY[key](**cfg)
