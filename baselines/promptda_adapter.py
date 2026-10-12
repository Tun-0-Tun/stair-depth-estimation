"""Adapter for Prompt Depth Anything (CVPR 2025) -- ``third_party/promptda``.

Upstream: https://github.com/DepthAnything/PromptDA
Paper:    Lin et al., "Prompting Depth Anything for 4K Resolution Accurate Metric
          Depth Estimation", CVPR 2025 (arXiv:2412.14015)
Weights:  https://huggingface.co/depth-anything/prompt-depth-anything-vitl (``model.ckpt``);
          also ``-vits`` and ``-vits-transparent``.

What this adapter absorbs
-------------------------
1. **Metres in, metres out.**  Upstream min-max normalises the prompt inside
   ``forward`` and denormalises the output, so nothing to do -- but the min/max
   are taken over the *whole* prompt, zeros included.  A prompt with holes would
   put 0 at the bottom of the range, so holes are nearest-filled first.
2. **Any depth map in.**  The prompt may have any resolution (the network
   resamples it); the RGB must be a multiple of 14 (DINOv2 patch).  The RGB is
   resized to the nearest such grid, capped at ``max_size`` on the long side as
   upstream's ``load_image`` does, and the prediction is resized back.
3. **Load.**  ``PromptDA.load_checkpoint`` only *warns* when the file is missing
   and then runs on random weights; ``availability`` rules that out.  We load
   the state dict ourselves (``weights_only=False``, a Lightning checkpoint), so
   the "Checkpoint  not found" warning printed on construction is expected.

Trained on iPhone LiDAR (192x256) prompts against ~1440x1920 images, i.e. an
LR/HR ratio of ~7.5, so ``bicubic_x8`` is the closest synthetic protocol.
"""

from __future__ import annotations

import importlib
import importlib.util
from pathlib import Path
from typing import Any

import numpy as np

from baselines.base import Availability, BaselineModel, add_third_party_to_path, register_baseline
from utils.misc import resize_depth, resize_rgb, sparse_to_dense_nn

__all__ = ["PromptDAAdapter"]

_PATCH = 14


@register_baseline("promptda")
class PromptDAAdapter(BaselineModel):
    """PromptDA: RGB + low-resolution metric depth prompt -> dense high-resolution depth."""

    input_modality = "lr"
    native_size = None  # any size; RGB snapped to a multiple of 14
    paper = "Lin et al., PromptDA, CVPR 2025 (arXiv:2412.14015)"
    submodule = "promptda"

    def __init__(self, encoder: str = "vitl", max_size: int = 1008, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.encoder = encoder
        self.max_size = int(max_size)
        self.n_degenerate = 0

    @classmethod
    def availability(cls, checkpoint_path: str | Path | None = None, **kwargs: Any) -> Availability:
        reasons: list[str] = []
        instructions: list[str] = []

        root = Path(__file__).resolve().parent.parent / "third_party" / "promptda"
        if not (root / "promptda" / "promptda.py").exists():
            reasons.append("third_party/promptda is not checked out")
            instructions.append("git submodule update --init --recursive")

        for mod, hint in (
            ("torch", "pip install torch"),
            ("huggingface_hub", "pip install huggingface-hub"),  # imported by promptda.py
        ):
            if importlib.util.find_spec(mod) is None:
                reasons.append(f"missing python package: {mod}")
                instructions.append(hint)

        ckpt = kwargs.get("checkpoint") or checkpoint_path
        if not ckpt or not Path(ckpt).exists():
            reasons.append(f"PromptDA checkpoint not found: {ckpt or '<unset>'}")
            instructions.append(
                "wget https://huggingface.co/depth-anything/prompt-depth-anything-vitl/resolve/"
                "main/model.ckpt -O checkpoints/promptda/promptda_vitl.ckpt"
            )
        return Availability(ok=not reasons, reasons=reasons, instructions=instructions)

    def _load(self, checkpoint_path: Path | None) -> Any:
        import torch

        add_third_party_to_path("promptda")
        PromptDA = importlib.import_module("promptda.promptda").PromptDA

        model = PromptDA(encoder=self.encoder, ckpt_path="")
        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        model.load_state_dict({k[9:]: v for k, v in ckpt["state_dict"].items()})
        return model.to(self.device).eval()

    def _grid(self, h: int, w: int) -> tuple[int, int]:
        s = min(1.0, self.max_size / max(h, w))
        return (
            max(_PATCH, round(h * s / _PATCH) * _PATCH),
            max(_PATCH, round(w * s / _PATCH) * _PATCH),
        )

    def _predict(self, rgb: np.ndarray, depth_in: np.ndarray, **kwargs: Any) -> np.ndarray:
        import torch

        h, w = rgb.shape[:2]
        depth = np.nan_to_num(depth_in, nan=0.0, posinf=0.0, neginf=0.0)
        if not (depth > 0).any():
            self.n_degenerate += 1
            return np.full((h, w), self.min_depth, dtype=np.float32)
        prompt = depth if (depth > 0).all() else sparse_to_dense_nn(depth)

        rgb_g = resize_rgb(rgb, self._grid(h, w), "bilinear")
        image = torch.from_numpy(rgb_g.transpose(2, 0, 1))[None].to(self.device).float()
        prompt_t = torch.from_numpy(prompt)[None, None].to(self.device).float()
        with torch.no_grad():
            out = self.model.predict(image, prompt_t)

        pred = out.squeeze().detach().float().cpu().numpy()
        pred = resize_depth(pred, (h, w), mode="bilinear")
        return np.clip(pred, self.min_depth, self.max_depth).astype(np.float32)

    def describe(self) -> dict[str, Any]:
        return {**super().describe(), "encoder": self.encoder, "n_degenerate": self.n_degenerate}
