"""Our method: a stair-specialised depth completion / super-resolution network.

STATUS: working skeleton.  It trains, it runs, and it plugs into the same
benchmark as the baselines -- but it is a starting point, not the SOW
deliverable.  The research work is in ``models/modules/``: swap a block, rerun
``scripts/run_benchmark.py``, read the table diff.

Architecture (v0)
-----------------
Two encoders (RGB, sparse depth + validity mask) -> per-scale confidence-gated
fusion -> U-Net decoder -> coarse depth -> edge-aware refinement with the
measured pixels re-injected as hard constraints.

Deliberate choices, all of them contestable and therefore all of them ablatable:

* **The validity mask is an input channel.**  A zero in the depth map means "no
  measurement", not "zero metres"; without the mask the first convolution
  cannot tell those apart.  This is the single most common bug in completion
  code.
* **Depth is normalised by the dataset range, not per-sample.**  Per-sample
  min-max (what DuCos does) leaks the ground-truth scale into the input at test
  time and makes metric depth impossible to recover if the input is empty.
* **Refinement re-injects the measurements every iteration.**  Where the sensor
  measured, we should not invent.
* **No monocular foundation backbone yet.**  Both baselines lean on Depth
  Anything V2; adding it is the obvious next experiment, and the reason
  ``FusionConfig.mono_features`` exists as a hook rather than being wired in.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn

from baselines.base import BaselineModel, register_baseline
from models.modules.blocks import ConvBlock, DownBlock, UpBlock, conv_bn_act
from models.modules.fusion import ConfidenceGatedFusion, EdgeAwareRefine

__all__ = ["OursAdapter", "StairDepthNet", "StairDepthNetConfig"]


@dataclass
class StairDepthNetConfig:
    base_channels: int = 32
    depth_levels: int = 4
    refine_iters: int = 6
    refine_kernel: int = 3
    use_confidence_gate: bool = True
    use_edge_aware_refine: bool = True
    min_depth: float = 0.3
    max_depth: float = 8.0
    mono_features: bool = False  # hook for a Depth-Anything-style backbone


class StairDepthNet(nn.Module):
    """RGB-guided depth completion, ``(rgb, sparse) -> dense depth in metres``."""

    def __init__(self, cfg: StairDepthNetConfig | None = None) -> None:
        super().__init__()
        self.cfg = cfg or StairDepthNetConfig()
        c = self.cfg.base_channels
        levels = self.cfg.depth_levels

        self.rgb_stem = ConvBlock(3, c)
        self.depth_stem = ConvBlock(2, c)  # depth + validity mask
        self.fuse = (
            ConfidenceGatedFusion(c, c, c)
            if self.cfg.use_confidence_gate
            else nn.Sequential(conv_bn_act(2 * c, c))
        )

        chans = [c * (2**i) for i in range(levels + 1)]
        self.downs = nn.ModuleList([DownBlock(chans[i], chans[i + 1]) for i in range(levels)])
        self.ups = nn.ModuleList(
            [UpBlock(chans[i + 1], chans[i], chans[i]) for i in reversed(range(levels))]
        )
        self.head = nn.Conv2d(c, 1, 3, padding=1)
        self.refine = (
            EdgeAwareRefine(c, kernel=self.cfg.refine_kernel, iters=self.cfg.refine_iters)
            if self.cfg.use_edge_aware_refine
            else None
        )

    def forward(self, rgb: torch.Tensor, sparse: torch.Tensor) -> dict[str, torch.Tensor]:
        """``rgb``: (B, 3, H, W) in [0, 1]; ``sparse``: (B, 1, H, W) metres, 0 = none."""
        lo, hi = self.cfg.min_depth, self.cfg.max_depth
        mask = (sparse > 0).float()
        sparse_n = torch.where(mask > 0, (sparse - lo) / (hi - lo), torch.zeros_like(sparse))

        f_rgb = self.rgb_stem(rgb)
        f_dep = self.depth_stem(torch.cat([sparse_n, mask], dim=1))
        if isinstance(self.fuse, ConfidenceGatedFusion):
            x, gate = self.fuse(f_rgb, f_dep)
        else:
            x, gate = self.fuse(torch.cat([f_rgb, f_dep], dim=1)), None

        skips = [x]
        for down in self.downs:
            x = down(x)
            skips.append(x)
        x = skips.pop()
        for up in self.ups:
            x = up(x, skips.pop())

        coarse_n = torch.sigmoid(self.head(x))
        coarse = lo + coarse_n * (hi - lo)

        out = {"coarse": coarse}
        if gate is not None:
            out["confidence"] = gate
        if self.refine is not None:
            out["pred"] = self.refine(coarse, x, sparse, rgb)
        else:
            out["pred"] = coarse
        return out


@register_baseline("ours")
class OursAdapter(BaselineModel):
    """Our model behind the same interface as the baselines, so it lands in the same table."""

    input_modality = "sparse"
    paper = "ours (WIP)"

    def __init__(self, net: dict[str, Any] | None = None, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        net = dict(net or {})
        net.setdefault("min_depth", self.min_depth)
        net.setdefault("max_depth", self.max_depth)
        self.net_config = StairDepthNetConfig(**net)

    def _load(self, checkpoint_path: Path | None) -> Any:
        model = StairDepthNet(self.net_config)
        if checkpoint_path is not None:
            state = torch.load(checkpoint_path, map_location="cpu")
            model.load_state_dict(state.get("model", state))
        elif not self.options.get("allow_random_init", False):
            raise RuntimeError(
                "[ours] no checkpoint given. Train one with `python scripts/train.py "
                "experiment=<name>`, or pass model.allow_random_init=true for a plumbing check "
                "(the numbers are then meaningless and must not be reported)."
            )
        return model.to(self.device).eval()

    def _predict(self, rgb: np.ndarray, depth_in: np.ndarray, **kwargs: Any) -> np.ndarray:
        rgb_t = torch.from_numpy(rgb.transpose(2, 0, 1))[None].to(self.device).float()
        dep_t = torch.from_numpy(depth_in)[None, None].to(self.device).float()
        with torch.no_grad():
            out = self.model(rgb_t, dep_t)
        pred = out["pred"].squeeze().detach().float().cpu().numpy()
        return np.clip(pred, self.min_depth, self.max_depth).astype(np.float32)
