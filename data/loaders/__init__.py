"""Dataset loaders.  Importing this package registers every dataset key.

Implemented: ``synthetic_stairs``, ``nyuv2``, ``arkitscenes``, ``zju_l5``,
``hammer``, ``rgbdd``.
Stubs (contract + TODO list, see the module docstring in each file):
``hypersim``, ``tartanair``, ``tofdc``, ``stair_dataset``, ``astra2_custom``.
"""

from data.loaders.arkitscenes import ARKitScenesDataset  # noqa: F401
from data.loaders.astra2_custom import Astra2CustomDataset  # noqa: F401
from data.loaders.base import (
    DATASET_REGISTRY,
    SAMPLE_KEYS,
    BaseDepthDataset,
    DatasetInfo,
    build_dataset,
    register_dataset,
    validate_sample,
)
from data.loaders.hammer import HammerDataset  # noqa: F401
from data.loaders.hypersim import HypersimDataset  # noqa: F401
from data.loaders.nyuv2 import NYUv2Dataset  # noqa: F401
from data.loaders.rgbdd import RGBDDDataset  # noqa: F401
from data.loaders.stair_dataset import RGBDStairDataset  # noqa: F401
from data.loaders.synthetic import SyntheticStairsDataset  # noqa: F401
from data.loaders.tartanair import TartanAirDataset  # noqa: F401
from data.loaders.tofdc import TOFDCDataset  # noqa: F401
from data.loaders.zju_l5 import ZJUL5Dataset  # noqa: F401

#: Datasets with a working loader (the rest are stubs).
IMPLEMENTED: tuple[str, ...] = (
    "synthetic_stairs",
    "nyuv2",
    "arkitscenes",
    "zju_l5",
    "hammer",
    "rgbdd",
)

__all__ = [
    "DATASET_REGISTRY",
    "IMPLEMENTED",
    "SAMPLE_KEYS",
    "BaseDepthDataset",
    "DatasetInfo",
    "build_dataset",
    "register_dataset",
    "validate_sample",
]
