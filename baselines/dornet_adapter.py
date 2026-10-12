"""Adapter for DORNet (CVPR 2025 oral) -- ``third_party/dornet``.

Upstream: https://github.com/yanzq95/DORNet
Paper:    Wang et al., "DORNet: A Degradation Oriented and Regularized Network for
          Blind Depth Super-Resolution", CVPR 2025 (arXiv:2410.11666)
Weights:  ``third_party/dornet/checkpoints/`` (in the repo itself): ``NYU_X4/X8/X16``
          (synthetic bicubic), ``RGBDD``, ``TOFDSR`` and their ``_Noisy`` / ``_T`` (tiny) variants.

Same group, same data pack and same protocol as DuCos, so this adapter *is* the
DuCos adapter with a different network (see ``baselines/ducos_adapter.py`` for
the per-sample min-max normalisation, the bicubic LR -> HR step and the
zero-range guard).  Differences absorbed here:

1. ``Net(tiny_model=...)`` instead of an args namespace; no Depth Anything backbone.
2. Call ``net(x_query=lr, rgb=guidance)`` and get a bare tensor.
3. ``net/deform_conv.py`` imports ``mmcv.ops.modulated_deform_conv2d`` -- a CUDA
   op, so DORNet needs mmcv built against the installed torch/CUDA.
4. ``third_party/dornet`` has a top-level ``utils.py`` and ``data/``: it goes on
   ``sys.path`` *after* the repo root so it cannot shadow our packages.

Unverified: the RGB is min-max normalised as in the DuCos adapter, because
DORNet's own ``nyu_dataloader.py`` only calls ``ToTensor`` on the pack's images
and we have not checked what range ``test_images_v2.npy`` holds.  The NYU x4
experiment is the check; if its RMSE is off, the RGB normalisation is the first suspect.
"""

from __future__ import annotations

import importlib
import importlib.util
from pathlib import Path
from typing import Any

from baselines.base import Availability, add_third_party_to_path, register_baseline
from baselines.ducos_adapter import DuCosAdapter

__all__ = ["DORNetAdapter"]


@register_baseline("dornet")
class DORNetAdapter(DuCosAdapter):
    """DORNet: RGB + dense low-resolution depth -> dense high-resolution depth."""

    paper = "Wang et al., DORNet, CVPR 2025 (arXiv:2410.11666)"
    submodule = "dornet"

    def __init__(
        self, scale: int = 4, tiny_model: bool = False, upsample: str = "pil", **kwargs: Any
    ) -> None:
        super().__init__(scale=scale, depth_anything_ckpt=None, upsample=upsample, **kwargs)
        self.tiny_model = bool(tiny_model)

    @classmethod
    def availability(cls, checkpoint_path: str | Path | None = None, **kwargs: Any) -> Availability:
        reasons: list[str] = []
        instructions: list[str] = []

        root = Path(__file__).resolve().parent.parent / "third_party" / "dornet"
        if not (root / "net" / "dornet.py").exists():
            reasons.append("third_party/dornet is not checked out")
            instructions.append("git submodule update --init --recursive")

        for mod, hint in (
            ("torch", "pip install torch"),
            (
                "mmcv",
                "install mmcv built for the installed torch/CUDA (upstream pins mmcv-full==1.7.2)",
            ),
        ):
            if importlib.util.find_spec(mod) is None:
                reasons.append(f"missing python package: {mod}")
                instructions.append(hint)

        ckpt = kwargs.get("checkpoint") or checkpoint_path
        if not ckpt or not Path(ckpt).exists():
            reasons.append(f"DORNet checkpoint not found: {ckpt or '<unset>'}")
            instructions.append(
                "set model.checkpoint=third_party/dornet/checkpoints/NYU_X4.pth "
                "(weights are per scale and per tiny/full)"
            )
        return Availability(ok=not reasons, reasons=reasons, instructions=instructions)

    def _load(self, checkpoint_path: Path | None) -> Any:
        import torch

        add_third_party_to_path("dornet", prepend=False)
        Net = importlib.import_module("net.dornet").Net

        model = Net(tiny_model=self.tiny_model)
        state = torch.load(checkpoint_path, map_location="cpu")
        state = {k.replace("module.", "", 1): v for k, v in state.items()}
        model.load_state_dict(state)  # strict: a tiny/full mix-up must fail here
        return model.to(self.device).eval()

    def _forward(self, rgb_t, dep_t):
        return self.model(x_query=dep_t, rgb=rgb_t)

    def describe(self) -> dict[str, Any]:
        return {**super().describe(), "tiny_model": self.tiny_model}
