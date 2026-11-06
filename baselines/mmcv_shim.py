"""Portable stand-in for ``mmcv.ops.modulated_deform_conv2d`` (DORNet's only mmcv use).

DORNet's ``net/deform_conv.py`` does ``from mmcv.ops import modulated_deform_conv2d``.
mmcv is a compiled OpenMMLab package that has to be built against the exact
torch/CUDA in use, and has no wheels for torch 2.6 / cu124.  The operation itself
is DCNv2 (modulated deformable convolution), which torchvision ships as
``torchvision.ops.deform_conv2d(..., mask=...)`` with the same offset layout
(``[dy0, dx0, dy1, dx1, ...]`` per kernel position and offset group), the same
sigmoid-mask-times-sample modulation, and zero for samples outside the image.

Only the signature differs: mmcv passes ``groups`` / ``deform_groups`` explicitly,
torchvision derives them from the tensor shapes.  The shim checks they agree
instead of silently computing something else.  ``tests/test_mmcv_shim.py`` checks it
against a literal per-pixel transcription of the DCNv2 formula.

A number produced with the shim came from torchvision's kernel, not mmcv's; the
adapter records ``deform="shim"`` in the run.  Install mmcv and this file goes unused.
"""

from __future__ import annotations

import sys
import types

import torch
from torchvision.ops import deform_conv2d

__all__ = ["install", "modulated_deform_conv2d"]


def modulated_deform_conv2d(
    x: torch.Tensor,
    offset: torch.Tensor,
    mask: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    stride,
    padding,
    dilation,
    groups: int,
    deform_groups: int,
) -> torch.Tensor:
    kh, kw = weight.shape[2:]
    if x.shape[1] // weight.shape[1] != groups:
        raise ValueError(
            f"groups={groups} does not match x {tuple(x.shape)} / weight {tuple(weight.shape)}"
        )
    if offset.shape[1] // (2 * kh * kw) != deform_groups or offset.shape[0] != x.shape[0]:
        raise ValueError(
            f"deform_groups={deform_groups} does not match offset {tuple(offset.shape)} for x "
            f"{tuple(x.shape)}; DORNet's call is only well-formed at batch size 1"
        )
    return deform_conv2d(x, offset, weight, bias, stride, padding, dilation, mask)


def install() -> bool:
    """Register this module as ``mmcv.ops`` if mmcv is not importable.

    Returns True if the shim was installed, False if the real mmcv is available.
    """
    if (
        getattr(sys.modules.get("mmcv.ops"), "modulated_deform_conv2d", None)
        is modulated_deform_conv2d
    ):
        return True  # already installed by an earlier load in this process
    try:
        from mmcv.ops import modulated_deform_conv2d as _  # noqa: F401

        return False
    except ImportError:
        ops = types.ModuleType("mmcv.ops")
        ops.modulated_deform_conv2d = modulated_deform_conv2d
        mmcv = types.ModuleType("mmcv")
        mmcv.ops = ops
        sys.modules["mmcv"], sys.modules["mmcv.ops"] = mmcv, ops
        return True
