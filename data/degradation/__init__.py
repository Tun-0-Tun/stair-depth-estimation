"""Sensor degradation models (GT depth -> realistic sensor input)."""

from data.degradation.active_stereo_astra2 import ActiveStereoAstra2Degradation
from data.degradation.base import (
    DEGRADATION_REGISTRY,
    BaseDegradation,
    IdentityDegradation,
    build_degradation,
    register_degradation,
)
from data.degradation.dtof_sim import DToFSimDegradation
from data.degradation.sr_downsample import BicubicSRDegradation, RandomSparseDegradation

__all__ = [
    "DEGRADATION_REGISTRY",
    "ActiveStereoAstra2Degradation",
    "BaseDegradation",
    "BicubicSRDegradation",
    "DToFSimDegradation",
    "IdentityDegradation",
    "RandomSparseDegradation",
    "build_degradation",
    "register_degradation",
]
