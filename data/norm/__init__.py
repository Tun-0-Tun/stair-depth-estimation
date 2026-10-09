"""Normal-map construction (GT depth -> per-pixel surface normals)."""

from data.norm.base import (
    NORMAL_REGISTRY,
    BaseNormalMap,
    ConstantNormalMap,
    build_normal_map,
    register_normal,
)
from data.norm.cross_product import CrossProductNormals

__all__ = [
    "NORMAL_REGISTRY",
    "BaseNormalMap",
    "ConstantNormalMap",
    "CrossProductNormals",
    "build_normal_map",
    "register_normal",
]
