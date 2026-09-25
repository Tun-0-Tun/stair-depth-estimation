"""Dataset loaders.  Importing this package registers every dataset key.

Stair-domain: ``void_stairs``, ``minjiang``.
General benchmarks: ``nyuv2``, ``zju_l5``, ``hammer``.
Depth-SR test sets: ``middlebury``, ``lu`` (both via ``depth_enhance``).
Synthetic training source: ``hypersim``.
CI only: ``synthetic_stairs`` (generated, never a reportable number).

Datasets live in one shared external folder -- see ``STAIR_DATA_ROOT`` in
``utils/misc.py`` and the layouts in ``docs/datasets.md``.
"""

from data.loaders.base import (
    DATASET_REGISTRY,
    SAMPLE_KEYS,
    BaseDepthDataset,
    DatasetInfo,
    build_dataset,
    register_dataset,
    validate_sample,
)
from data.loaders.depth_enhance import DepthEnhanceDataset  # noqa: F401
from data.loaders.hammer import HammerDataset  # noqa: F401
from data.loaders.hypersim import HyperSimDataset  # noqa: F401
from data.loaders.minjiang import MinJiangDataset  # noqa: F401
from data.loaders.nyuv2 import NYUv2Dataset  # noqa: F401
from data.loaders.synthetic import SyntheticStairsDataset  # noqa: F401
from data.loaders.void_stairs import VOIDStairsDataset  # noqa: F401
from data.loaders.zju_l5 import ZJUL5Dataset  # noqa: F401

__all__ = [
    "DATASET_REGISTRY",
    "SAMPLE_KEYS",
    "BaseDepthDataset",
    "DatasetInfo",
    "build_dataset",
    "register_dataset",
    "validate_sample",
]
