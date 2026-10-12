"""Adapter for OMNI-DC v1.1 (ICCV 2025) -- ``third_party/omni_dc``.

Upstream: https://github.com/princeton-vl/OMNI-DC
Paper:    Zuo et al., "OMNI-DC: Highly Robust Depth Completion with Multiresolution
          Depth Integration", ICCV 2025 (arXiv:2411.19278)
Weights:  ``modelv1.1_best_72epochs.pt`` (Google Drive, see upstream README) or the
          Hub mirror https://huggingface.co/zuoym15/OMNI-DC (a directory with
          ``config.json`` + ``model.safetensors``); plus Depth Anything V2 **vitl**
          (``depth_anything_v2_vitl.pth``), which v1.1 concatenates to the sparse map.

Completion only: RGB + sparse metric depth (zeros = no measurement).

What this adapter absorbs
-------------------------
1. **A model configured by ``argparse`` at import time.**  ``src/config.py`` parses
   ``sys.argv`` into a module-level ``args``.  We import it under a private name
   with ``sys.argv`` scoped to the v1.1 flags from ``testing_scripts/demo.sh``
   (``max_depth=300`` is the *model's* setting, not our evaluation window).
2. **A relative weight path.**  ``model/ognidc.py`` loads the DAv2 backbone from
   ``./depth_models/.../depth_anything_v2_vitl.pth``, relative to the cwd; that
   one ``torch.load`` is redirected (as for DuCos).
3. **Module-name collisions.**  ``src/`` has ``data``, ``utils`` and ``config``
   packages of its own.  It is appended to ``sys.path``, after our repo root.
4. **Input geometry.**  ImageNet-normalised RGB, H and W zero-padded at the
   bottom/right to a multiple of ``4 * 2**(num_resolution - 1)`` (16), the output
   cropped back.  The model whitens the sparse depth by its median internally,
   so metres go in and metres come out.

Environment: the model imports ``mmcv.runner`` and ``mmseg`` (``model/pvt.py``),
so it needs mmcv 1.x (upstream pins mmcv==1.4.4) -- which does not coexist with
the mmcv that DORNet's deformable conv uses.  Plan on a separate environment.
"""

from __future__ import annotations

import contextlib
import importlib
import importlib.util
import sys
from pathlib import Path
from typing import Any

import numpy as np

from baselines.base import Availability, BaselineModel, add_third_party_to_path, register_baseline
from baselines.ducos_adapter import _redirect_torch_load

__all__ = ["OmniDCAdapter"]

_DA2_FILENAME = "depth_anything_v2_vitl.pth"
_IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
#: testing_scripts/demo.sh, v1.1
_V11_FLAGS = [
    "--max_depth",
    "300.0",
    "--data_normalize_median",
    "1",
    "--num_resolution",
    "3",
    "--multi_resolution_learnable_gradients_weights",
    "uniform",
    "--load_dav2",
    "1",
    "--gpus",
    "0",
    "--GRU_iters",
    "1",
    "--optim_layer_input_clamp",
    "1.0",
    "--depth_activation_format",
    "exp",
    "--whiten_sparse_depths",
    "1",
    "--gru_internal_whiten_method",
    "median",
    "--backbone_mode",
    "rgbd",
    "--pred_confidence_input",
    "1",
]


@contextlib.contextmanager
def _argv(argv: list[str]):
    saved = sys.argv
    sys.argv = argv
    try:
        yield
    finally:
        sys.argv = saved


def _omni_args(src: Path) -> Any:
    """``src/config.py`` evaluated under a private module name with the v1.1 flags."""
    spec = importlib.util.spec_from_file_location("omni_dc_config", src / "config.py")
    mod = importlib.util.module_from_spec(spec)
    with _argv(["omni_dc", *_V11_FLAGS]):
        spec.loader.exec_module(mod)
    return mod.args


@register_baseline("omni_dc")
class OmniDCAdapter(BaselineModel):
    """OMNI-DC v1.1: RGB + sparse metric depth -> dense metric depth."""

    input_modality = "sparse"
    native_size = None  # padded to a multiple of 16
    paper = "Zuo et al., OMNI-DC, ICCV 2025 (arXiv:2411.19278)"
    submodule = "omni_dc"

    def __init__(
        self,
        depth_anything_ckpt: str | None = "checkpoints/depth_anything_v2_vitl.pth",
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.depth_anything_ckpt = depth_anything_ckpt
        self.n_degenerate = 0
        self._divisor = 16

    @classmethod
    def availability(cls, checkpoint_path: str | Path | None = None, **kwargs: Any) -> Availability:
        reasons: list[str] = []
        instructions: list[str] = []

        root = Path(__file__).resolve().parent.parent / "third_party" / "omni_dc"
        if not (root / "src" / "model" / "ognidc.py").exists():
            reasons.append("third_party/omni_dc is not checked out")
            instructions.append("git submodule update --init --recursive")

        for mod, hint in (
            ("torch", "pip install torch"),
            ("timm", "pip install timm"),
            ("einops", "pip install einops"),
            ("huggingface_hub", "pip install huggingface-hub"),
            ("mmcv", "pip install mmcv==1.4.4 (upstream; needs `mmcv.runner`, i.e. mmcv 1.x)"),
            ("mmseg", "pip install mmsegmentation==0.22.1"),
        ):
            if importlib.util.find_spec(mod) is None:
                reasons.append(f"missing python package: {mod}")
                instructions.append(hint)

        ckpt = kwargs.get("checkpoint") or checkpoint_path
        if not ckpt or not Path(ckpt).exists():
            reasons.append(f"OMNI-DC v1.1 checkpoint not found: {ckpt or '<unset>'}")
            instructions.append(
                "download modelv1.1_best_72epochs.pt (Google Drive link in third_party/omni_dc/"
                "README.md) to checkpoints/omni_dc/, or "
                "huggingface-cli download zuoym15/OMNI-DC --local-dir checkpoints/omni_dc/hf "
                "and point model.checkpoint at that directory"
            )

        da = kwargs.get("depth_anything_ckpt", "checkpoints/depth_anything_v2_vitl.pth")
        if da and not Path(da).exists():
            reasons.append(f"Depth Anything V2 vitl weights not found: {da}")
            instructions.append(
                "wget https://huggingface.co/depth-anything/Depth-Anything-V2-Large/resolve/main/"
                f"{_DA2_FILENAME} -P checkpoints/"
            )
        return Availability(ok=not reasons, reasons=reasons, instructions=instructions)

    def _load(self, checkpoint_path: Path | None) -> Any:
        import torch

        src = add_third_party_to_path("omni_dc", "src", prepend=False)
        args = _omni_args(src)
        self._divisor = int(4 * 2 ** (args.num_resolution - 1))

        with _redirect_torch_load(_DA2_FILENAME, self.depth_anything_ckpt):
            OGNIDC = importlib.import_module("model.ognidc").OGNIDC
            if Path(checkpoint_path).is_dir():
                net = OGNIDC.from_pretrained(str(checkpoint_path), args=args)
            else:
                net = OGNIDC(args)
                ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
                state = {k.replace("module.", "", 1): v for k, v in ckpt["net"].items()}
                missing, unexpected = net.load_state_dict(state, strict=False)
                if missing:
                    raise RuntimeError(
                        f"[omni_dc] checkpoint {checkpoint_path} is missing {len(missing)} tensors, "
                        f"e.g. {missing[:5]}. Is it v1.1 (load_dav2=1)?"
                    )
                if unexpected:
                    print(
                        f"[omni_dc] ignoring {len(unexpected)} unexpected tensors in the checkpoint"
                    )
        return net.to(self.device).eval()

    def _predict(self, rgb: np.ndarray, depth_in: np.ndarray, **kwargs: Any) -> np.ndarray:
        import torch

        h, w = rgb.shape[:2]
        depth = np.nan_to_num(depth_in, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
        depth[depth < 0] = 0.0
        if not (depth > 0).any():
            self.n_degenerate += 1
            return np.full((h, w), self.min_depth, dtype=np.float32)

        d = self._divisor
        h_pad, w_pad = -(-h // d) * d, -(-w // d) * d
        rgb_p = np.zeros((h_pad, w_pad, 3), dtype=np.float32)
        rgb_p[:h, :w] = (rgb - _IMAGENET_MEAN) / _IMAGENET_STD
        dep_p = np.zeros((h_pad, w_pad), dtype=np.float32)
        dep_p[:h, :w] = depth

        sample = {
            "rgb": torch.from_numpy(rgb_p.transpose(2, 0, 1))[None].to(self.device),
            "dep": torch.from_numpy(dep_p)[None, None].to(self.device),
            "K": torch.eye(3)[None].to(self.device),  # unused, but forward() reads it
            "pattern": 0,  # unused, ditto
        }
        with torch.no_grad():
            out = self.model(sample)

        pred = out["pred"].squeeze().detach().float().cpu().numpy()[:h, :w]
        return np.clip(pred, self.min_depth, self.max_depth).astype(np.float32)

    def describe(self) -> dict[str, Any]:
        return {**super().describe(), "n_degenerate": self.n_degenerate}
