"""Cross-modal fusion and refinement blocks -- the parts we expect to iterate on.

Each block here is one *hypothesis* about what a stair scene needs, stated so it
can be ablated independently:

``ConfidenceGatedFusion``
    The sparse input is not uniformly trustworthy: an active-stereo camera is
    confident on a flat tread and wrong on a nosing.  Instead of concatenating
    RGB and depth features, predict a per-pixel gate from both and let the
    network decide where to trust the measurement.

``EdgeAwareRefine``
    A stair is a stack of depth discontinuities.  A refinement that averages
    across them destroys exactly the structure we are being judged on, so the
    propagation weights are modulated by an RGB-derived edge map.

Ablation protocol: turn a block off in ``configs/model/ours.yaml`` and rerun
``scripts/run_benchmark.py`` -- the table diff is the block's contribution.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.modules.blocks import conv_bn_act

__all__ = ["ConfidenceGatedFusion", "EdgeAwareRefine", "sobel_edges"]


def sobel_edges(x: torch.Tensor) -> torch.Tensor:
    """Gradient magnitude of a (B, C, H, W) tensor, averaged over channels."""
    kx = torch.tensor([[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]], device=x.device)
    ky = kx.t()
    k = torch.stack([kx, ky])[:, None]  # (2, 1, 3, 3)
    c = x.shape[1]
    k = k.repeat(c, 1, 1, 1)  # depthwise
    g = F.conv2d(x, k, padding=1, groups=c)
    gx, gy = g[:, 0::2], g[:, 1::2]
    return torch.sqrt(gx**2 + gy**2 + 1e-12).mean(dim=1, keepdim=True)


class ConfidenceGatedFusion(nn.Module):
    """Fuse RGB and depth features through a learned per-pixel confidence gate."""

    def __init__(self, rgb_ch: int, depth_ch: int, out_ch: int) -> None:
        super().__init__()
        self.gate = nn.Sequential(
            conv_bn_act(rgb_ch + depth_ch, out_ch), nn.Conv2d(out_ch, 1, 1), nn.Sigmoid()
        )
        self.proj_rgb = conv_bn_act(rgb_ch, out_ch, kernel=1)
        self.proj_depth = conv_bn_act(depth_ch, out_ch, kernel=1)

    def forward(
        self, f_rgb: torch.Tensor, f_depth: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        gate = self.gate(torch.cat([f_rgb, f_depth], dim=1))
        fused = gate * self.proj_depth(f_depth) + (1.0 - gate) * self.proj_rgb(f_rgb)
        return fused, gate


class EdgeAwareRefine(nn.Module):
    """Iterative local propagation whose weights are suppressed across RGB edges.

    A cheap, dependency-free stand-in for CSPN++ (which DEPTHOR gets from the
    ``BpOps`` CUDA extension and which therefore does not run on CPU/MPS).  Same
    idea -- learned local affinities, several iterations, the measured pixels
    re-injected each step -- implemented with plain convolutions so it is
    portable and differentiable everywhere.
    """

    def __init__(self, feat_ch: int, kernel: int = 3, iters: int = 6) -> None:
        super().__init__()
        if kernel % 2 == 0:
            raise ValueError("kernel must be odd")
        self.kernel = kernel
        self.iters = int(iters)
        self.affinity = nn.Conv2d(feat_ch + 1, kernel * kernel, 3, padding=1)
        self.unfold = nn.Unfold(kernel_size=kernel, padding=kernel // 2)

    def forward(
        self,
        depth: torch.Tensor,
        feat: torch.Tensor,
        sparse: torch.Tensor,
        rgb: torch.Tensor | None = None,
    ) -> torch.Tensor:
        b, _, h, w = depth.shape
        logits = self.affinity(torch.cat([feat, depth], dim=1))
        if rgb is not None:
            # suppress propagation across image edges
            edge = sobel_edges(rgb)
            logits = logits - 4.0 * edge
        weights = torch.softmax(logits.view(b, self.kernel**2, -1), dim=1).view_as(logits)

        known = (sparse > 0).float()
        out = depth
        for _ in range(self.iters):
            patches = self.unfold(out).view(b, self.kernel**2, h * w)
            out = (patches * weights.view(b, self.kernel**2, h * w)).sum(1).view(b, 1, h, w)
            out = known * sparse + (1.0 - known) * out  # measurements are hard constraints
        return out
