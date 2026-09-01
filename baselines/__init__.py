"""Thin adapters over the vendored official implementations in ``third_party/``.

Never edit ``third_party/*``.  Every input-format, unit or path difference is
handled in an adapter here.  See ``baselines/base.py`` for the interface.
"""

from baselines.base import (
    BASELINE_REGISTRY,
    Availability,
    BaselineModel,
    build_baseline,
    register_baseline,
)
from baselines.depthor_adapter import DepthorAdapter, DepthorPlusPlusAdapter
from baselines.ducos_adapter import DuCosAdapter
from baselines.reference import BicubicBaseline, NearestFillBaseline

__all__ = [
    "BASELINE_REGISTRY",
    "Availability",
    "BaselineModel",
    "BicubicBaseline",
    "DepthorAdapter",
    "DepthorPlusPlusAdapter",
    "DuCosAdapter",
    "NearestFillBaseline",
    "build_baseline",
    "register_baseline",
]
