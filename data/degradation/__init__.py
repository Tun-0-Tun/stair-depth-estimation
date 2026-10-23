"""Sensor degradation models (GT depth -> realistic sensor input)."""

from data.degradation.active_stereo_astra2 import ActiveStereoAstra2Degradation
from data.degradation.astra_hard import AstraHardDegradation, masked_downsample
from data.degradation.axial import AxialDegradation
from data.degradation.base import (
    DEGRADATION_REGISTRY,
    BaseDegradation,
    IdentityDegradation,
    build_degradation,
    register_degradation,
)
from data.degradation.dtof_sim import DToFSimDegradation
from data.degradation.lateral import LateralDegradation
from data.degradation.projector_shadow import ProjectorShadowDegradation
from data.degradation.sequential import SequentialDegradation
from data.degradation.sr_downsample import BicubicSRDegradation, RandomSparseDegradation

__all__ = [
    "DEGRADATION_REGISTRY",
    "ActiveStereoAstra2Degradation",
    "AstraHardDegradation",
    "AxialDegradation",
    "BaseDegradation",
    "BicubicSRDegradation",
    "DToFSimDegradation",
    "IdentityDegradation",
    "LateralDegradation",
    "ProjectorShadowDegradation",
    "RandomSparseDegradation",
    "SequentialDegradation",
    "build_degradation",
    "masked_downsample",
    "register_degradation",
]
