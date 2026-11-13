"""Normal-map construction (GT depth -> per-pixel surface normals)."""

from data.norm.base import (
    NORMAL_REGISTRY,
    BaseNormalMap,
    ConstantNormalMap,
    build_normal_map,
    register_normal,
)
from data.norm.cross_product import CrossProductNormals
from data.norm.edge_mask import (
    close_and_filter_edges,
    distance_to_edge,
    edge_normal_direction,
    normal_jump_mask,
    plane_edge_mask,
    unit_normals,
    z_jump_mask,
)
from data.norm.theta import get_theta_map, resize_normals, resize_theta

__all__ = [
    "NORMAL_REGISTRY",
    "BaseNormalMap",
    "ConstantNormalMap",
    "CrossProductNormals",
    "build_normal_map",
    "close_and_filter_edges",
    "distance_to_edge",
    "edge_normal_direction",
    "get_theta_map",
    "normal_jump_mask",
    "plane_edge_mask",
    "register_normal",
    "resize_normals",
    "resize_theta",
    "unit_normals",
    "z_jump_mask",
]
