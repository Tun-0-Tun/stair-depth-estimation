"""Normal-map construction (GT depth -> per-pixel surface normals)."""

from data.norm.base import (
    NORMAL_REGISTRY,
    BaseNormalMap,
    ConstantNormalMap,
    build_normal_map,
    register_normal,
)
from data.norm.cross_product import CrossProductNormals
from data.norm.theta import get_theta_map, resize_normals, resize_theta

__all__ = [
    "NORMAL_REGISTRY",
    "BaseNormalMap",
    "ConstantNormalMap",
    "CrossProductNormals",
    "build_normal_map",
    "get_theta_map",
    "register_normal",
    "resize_normals",
    "resize_theta",
]
