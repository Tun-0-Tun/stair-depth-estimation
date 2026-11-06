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
from baselines.dornet_adapter import DORNetAdapter
from baselines.ducos_adapter import DuCosAdapter
from baselines.local_bilateral import LocalBilateralBaseline
from baselines.omni_dc_adapter import OmniDCAdapter
from baselines.prior_da_adapter import PriorDAAdapter
from baselines.promptda_adapter import PromptDAAdapter
from baselines.reference import BicubicBaseline, NearestFillBaseline
from baselines.wave_adapter import WaveAdapter

__all__ = [
    "BASELINE_REGISTRY",
    "Availability",
    "BaselineModel",
    "BicubicBaseline",
    "DORNetAdapter",
    "DepthorAdapter",
    "DepthorPlusPlusAdapter",
    "DuCosAdapter",
    "LocalBilateralBaseline",
    "NearestFillBaseline",
    "OmniDCAdapter",
    "PriorDAAdapter",
    "PromptDAAdapter",
    "WaveAdapter",
    "build_baseline",
    "register_baseline",
]
