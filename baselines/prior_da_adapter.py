"""Adapter for Prior Depth Anything (ICLR 2026) -- ``third_party/prior_depth_anything``.

Upstream: https://github.com/SpatialVision/Prior-Depth-Anything
Paper:    Wang et al., "Depth Anything with Any Prior", ICLR 2026 (arXiv:2505.10565)
Weights:  https://huggingface.co/Rain729/Prior-Depth-Anything -- two files, both needed:
          ``depth_anything_v2_vitb.pth`` (frozen coarse stage) and
          ``prior_depth_anything_vitb_1_1.pth`` (fine stage, v1.1; v1.0 is ``..._vitb.pth``).

One set of weights takes sparse points *or* a low-resolution map, so this
adapter serves both of our input modalities: ``input="sparse"`` (default,
completion; full-resolution map with zeros) or ``input="lr"`` (SR; dense LR map).

What this adapter absorbs
-------------------------
1. **Two input types.**  ``infer_one_sample`` decides by shape: a prior whose
   size differs from the RGB is treated as a low-resolution map and placed on
   the HR grid (``pattern`` must stay ``None``); otherwise its non-zeros are the
   sparse points.  We pass ``pattern=None`` and let it.
2. **uint8 RGB.**  Upstream wants ``[0, 255]``; ours is ``[0, 1]``.
3. **Hard-coded weight lookup.**  Upstream builds file names from the version
   and model size and downloads from the Hub when a directory is not given.  We
   always pass directories (the server may be offline) and check the names.
4. **Too few points.**  Upstream raises below ``K`` (=5) known points, which
   would abort a whole benchmark; such a sample is counted in ``n_degenerate``
   and gets a constant (the mean of what is known).
5. ``torch_cluster`` (KNN) is a compiled extension and must match torch/CUDA.
"""

from __future__ import annotations

import importlib
import importlib.util
from pathlib import Path
from typing import Any

import numpy as np

from baselines.base import Availability, BaselineModel, add_third_party_to_path, register_baseline

__all__ = ["PriorDAAdapter"]

_K = 5  # upstream Arguments.K: minimum known points
_FINE = {"1.1": "prior_depth_anything_vitb_1_1.pth", "1.0": "prior_depth_anything_vitb.pth"}
_COARSE = "depth_anything_v2_vitb.pth"


@register_baseline("prior_da")
class PriorDAAdapter(BaselineModel):
    """Prior Depth Anything: RGB + sparse points or LR depth -> dense metric depth."""

    native_size = None  # fine stage runs at 518 internally and returns the input size
    paper = "Wang et al., Prior Depth Anything, ICLR 2026 (arXiv:2505.10565)"
    submodule = "prior_depth_anything"

    def __init__(self, **kwargs: Any) -> None:
        # input / version stay in kwargs: they reach ``self.options`` and so ``availability``
        super().__init__(**kwargs)
        self.input_modality = str(kwargs.get("input", "sparse"))
        self.version = str(kwargs.get("version", "1.1"))
        if self.input_modality not in ("sparse", "lr"):
            raise ValueError(f"input must be 'sparse' or 'lr', got {self.input_modality!r}")
        if self.version not in _FINE:
            raise ValueError(f"version must be one of {sorted(_FINE)}, got {self.version!r}")
        self.n_degenerate = 0

    @classmethod
    def availability(cls, checkpoint_path: str | Path | None = None, **kwargs: Any) -> Availability:
        reasons: list[str] = []
        instructions: list[str] = []

        root = Path(__file__).resolve().parent.parent / "third_party" / "prior_depth_anything"
        if not (root / "prior_depth_anything" / "__init__.py").exists():
            reasons.append("third_party/prior_depth_anything is not checked out")
            instructions.append("git submodule update --init --recursive")

        for mod, hint in (
            ("torch", "pip install torch"),
            (
                "torch_cluster",
                "pip install torch_cluster -f https://data.pyg.org/whl/torch-<torch>+<cuXXX>.html",
            ),
            ("huggingface_hub", "pip install huggingface-hub"),
            ("cv2", "pip install opencv-python-headless"),
            ("einops", "pip install einops"),
            ("matplotlib", "pip install matplotlib"),  # upstream utils imports it at module level
            ("safetensors", "pip install safetensors"),
        ):
            if importlib.util.find_spec(mod) is None:
                reasons.append(f"missing python package: {mod}")
                instructions.append(hint)

        fine = _FINE.get(str(kwargs.get("version", "1.1")), _FINE["1.1"])
        for label, path, name in (
            ("fine-stage", kwargs.get("checkpoint") or checkpoint_path, fine),
            ("coarse-stage Depth Anything V2", kwargs.get("depth_anything_ckpt"), _COARSE),
        ):
            if not path or Path(path).name != name or not Path(path).exists():
                reasons.append(
                    f"Prior Depth Anything {label} weights not found as {name}: {path or '<unset>'}"
                )
                instructions.append(
                    f"huggingface-cli download Rain729/Prior-Depth-Anything {name} "
                    "--local-dir checkpoints/prior_depth_anything (the file name is fixed by upstream)"
                )
        return Availability(ok=not reasons, reasons=reasons, instructions=instructions)

    def _load(self, checkpoint_path: Path | None) -> Any:
        add_third_party_to_path("prior_depth_anything")
        PriorDepthAnything = importlib.import_module("prior_depth_anything").PriorDepthAnything

        return PriorDepthAnything(
            device=self.device,
            version=self.version,
            mde_dir=str(Path(self.options["depth_anything_ckpt"]).parent),
            ckpt_dir=str(Path(checkpoint_path).parent),
        )

    def _predict(self, rgb: np.ndarray, depth_in: np.ndarray, **kwargs: Any) -> np.ndarray:
        import torch

        h, w = rgb.shape[:2]
        # C-contiguous: loaders may hand over transposed views, and upstream calls .view() on them
        depth = np.ascontiguousarray(
            np.nan_to_num(depth_in, nan=0.0, posinf=0.0, neginf=0.0), dtype=np.float32
        )
        depth[depth < 0] = 0.0
        if (depth > 0).sum() < _K:
            self.n_degenerate += 1
            known = depth[depth > 0]
            fill = float(known.mean()) if known.size else self.min_depth
            return np.clip(np.full((h, w), fill, dtype=np.float32), self.min_depth, self.max_depth)

        image = np.ascontiguousarray(np.clip(np.rint(rgb * 255.0), 0, 255), dtype=np.uint8)
        with torch.no_grad():
            out = self.model.infer_one_sample(image=image, prior=depth)
        pred = out.squeeze().detach().float().cpu().numpy()
        return np.clip(pred, self.min_depth, self.max_depth).astype(np.float32)

    def describe(self) -> dict[str, Any]:
        return {**super().describe(), "version": self.version, "n_degenerate": self.n_degenerate}
