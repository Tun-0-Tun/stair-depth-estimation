"""Adapter for DuCos (ICCV 2025) -- ``third_party/ducos``.

Upstream: https://github.com/yanzq95/DuCos
Paper:    Yan et al., "DuCos: Duality Constrained Depth Super-Resolution via
          Foundation Model", ICCV 2025 (arXiv:2503.04171)
Weights:  https://huggingface.co/RaynWu2002/DuCos/tree/main

What this adapter absorbs
-------------------------
1. **Per-sample min-max normalisation.**  This is the big one.  DuCos does not
   work in metres: ``data/rgbdd_dataloader.py`` normalises the low-resolution
   depth to ``[0, 1]`` with *that sample's own* min and max, and
   ``utils/metric_func.py`` denormalises the prediction with the same constants
   before scoring.  Feeding metric depth straight in produces plausible-looking
   garbage.  We normalise on the way in and invert on the way out, so the rest
   of the repo only sees metres.  The RGB is min-max normalised too, exactly as
   upstream does.
2. **Dense HR input.**  DuCos expects the LR depth already bicubically
   upsampled to the GT resolution (that is what their test scripts feed it), not
   the LR grid.  We do that here.
3. **A hard-coded backbone path.**  ``model/ours/ducos.py`` loads Depth
   Anything V2 from ``/opt/data/private/DSR/...`` with a bare ``torch.load``.
   There is no seam to patch, so we scope-patch ``torch.load`` for the duration
   of model construction and redirect that one filename.  Ugly, but it keeps
   the submodule pristine -- editing vendored code is how a repo stops being
   reproducible.
4. **The args namespace.**  ``model_list.import_model`` wants an argparse
   ``Namespace``; we build the minimal one and let it fill in its own defaults.
5. **Output unpacking.**  Returns ``{'pred', 'pred_init', 'out_rgb', 'out_dep'}``;
   we take ``pred``.

Zero-range guard: if a sample's LR depth is constant (all holes, or a wall at a
single distance), min == max and the normalisation is undefined.  We detect it
and return the input unchanged rather than dividing by zero -- and count it in
``meta`` so it shows up rather than silently skewing the average.
"""

from __future__ import annotations

import contextlib
import importlib
import importlib.util
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np

from baselines.base import Availability, BaselineModel, add_third_party_to_path, register_baseline
from utils.misc import resize_depth, resize_rgb

__all__ = ["DuCosAdapter"]

_DA2_FILENAME = "depth_anything_v2_vits.pth"


@contextlib.contextmanager
def _redirect_torch_load(filename: str, replacement: str | None):
    """Temporarily make ``torch.load(<...>/filename)`` read ``replacement``."""
    import torch

    original = torch.load

    def patched(f, *args, **kwargs):
        if replacement and isinstance(f, (str | Path)) and Path(str(f)).name == filename:
            f = replacement
        kwargs.setdefault("map_location", "cpu")
        return original(f, *args, **kwargs)

    torch.load = patched
    try:
        yield
    finally:
        torch.load = original


@register_baseline("ducos")
class DuCosAdapter(BaselineModel):
    """DuCos: RGB + dense low-resolution depth -> dense high-resolution depth."""

    input_modality = "lr"
    native_size = None  # accepts arbitrary sizes; DAv2 branch pads to a multiple of 14
    paper = "Yan et al., DuCos, ICCV 2025 (arXiv:2503.04171)"
    submodule = "ducos"

    def __init__(
        self,
        scale: int = 4,
        model_name: str = "DuCos",
        depth_anything_ckpt: str | None = "checkpoints/depth_anything_v2_vits.pth",
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.scale = int(scale)
        self.model_name = model_name
        self.depth_anything_ckpt = depth_anything_ckpt
        self.n_degenerate = 0

    # ------------------------------------------------------------ checks

    @classmethod
    def availability(cls, checkpoint_path: str | Path | None = None, **kwargs: Any) -> Availability:
        reasons: list[str] = []
        instructions: list[str] = []

        root = Path(__file__).resolve().parent.parent / "third_party" / "ducos"
        if not (root / "model" / "ours" / "ducos.py").exists():
            reasons.append("third_party/ducos is not checked out")
            instructions.append("git submodule update --init --recursive")

        for mod, hint in (
            ("torch", "pip install torch"),
            ("torchvision", "pip install torchvision"),
            # Depth Anything V2's dpt.py imports cv2 at module level
            ("cv2", "pip install opencv-python-headless"),
        ):
            if importlib.util.find_spec(mod) is None:
                reasons.append(f"missing python package: {mod}")
                instructions.append(hint)

        ckpt = kwargs.get("checkpoint") or checkpoint_path
        if not ckpt or not Path(ckpt).exists():
            reasons.append(f"DuCos checkpoint not found: {ckpt or '<unset>'}")
            instructions.append(
                "huggingface-cli download RaynWu2002/DuCos --local-dir checkpoints/ducos "
                "(the files live under DuCos/ckpts/ inside the repo), then set model.checkpoint"
            )

        da = kwargs.get("depth_anything_ckpt", "checkpoints/depth_anything_v2_vits.pth")
        if da and not Path(da).exists():
            reasons.append(f"Depth Anything V2 backbone weights not found: {da}")
            instructions.append(
                "wget https://huggingface.co/depth-anything/Depth-Anything-V2-Small/resolve/main/"
                f"{_DA2_FILENAME} -P checkpoints/"
            )

        return Availability(ok=not reasons, reasons=reasons, instructions=instructions)

    # -------------------------------------------------------------- load

    def _load(self, checkpoint_path: Path | None) -> Any:
        import torch

        add_third_party_to_path("ducos")
        import_model = importlib.import_module("model_list").import_model

        args = SimpleNamespace(
            model_name=self.model_name,
            scale=self.scale,
            max_depth=self.max_depth,
            min_depth=self.min_depth,
        )
        with _redirect_torch_load(_DA2_FILENAME, self.depth_anything_ckpt):
            model = import_model(args)  # sets prop_kernel / prop_time / conf_prop / loss on args

        if checkpoint_path is not None:
            ckpt = torch.load(checkpoint_path, map_location="cpu")
            state = ckpt.get("state_dict", ckpt) if isinstance(ckpt, dict) else ckpt
            state = {k.replace("module.", "", 1): v for k, v in state.items()}
            missing, unexpected = model.load_state_dict(state, strict=False)
            if missing:
                raise RuntimeError(
                    f"[ducos] checkpoint {checkpoint_path} is missing {len(missing)} tensors, "
                    f"e.g. {missing[:5]}. Wrong checkpoint for this scale/variant?"
                )
            if unexpected:
                print(f"[ducos] ignoring {len(unexpected)} unexpected tensors in the checkpoint")

        return model.to(self.device).eval()

    # ----------------------------------------------------------- predict

    def _predict(self, rgb: np.ndarray, depth_in: np.ndarray, **kwargs: Any) -> np.ndarray:
        import torch

        h, w = rgb.shape[:2]

        # 2. LR depth -> dense HR grid, exactly as the upstream test scripts do
        dep = depth_in if depth_in.shape == (h, w) else resize_depth(depth_in, (h, w), "bicubic")

        # 1. per-sample min-max normalisation (theirs, reproduced verbatim)
        d_min, d_max = float(np.min(dep)), float(np.max(dep))
        if d_max - d_min < 1e-6:
            self.n_degenerate += 1
            return np.full((h, w), d_min, dtype=np.float32)
        dep_n = (dep - d_min) / (d_max - d_min)

        r_min, r_max = float(np.min(rgb)), float(np.max(rgb))
        rgb_n = rgb if r_max - r_min < 1e-6 else (rgb - r_min) / (r_max - r_min)
        rgb_n = resize_rgb(rgb_n.astype(np.float32), (h, w))

        sample = {
            "rgb": torch.from_numpy(rgb_n.transpose(2, 0, 1))[None].to(self.device).float(),
            "dep": torch.from_numpy(dep_n.astype(np.float32))[None, None].to(self.device).float(),
        }
        with torch.no_grad():
            out = self.model(sample)

        pred_n = out["pred"].squeeze().detach().float().cpu().numpy()
        pred = pred_n * (d_max - d_min) + d_min  # back to metres
        pred = np.clip(pred, self.min_depth, self.max_depth)
        if pred.shape != (h, w):
            pred = resize_depth(pred, (h, w), mode="bilinear")
        return pred.astype(np.float32)

    def describe(self) -> dict[str, Any]:
        return {**super().describe(), "scale": self.scale, "n_degenerate": self.n_degenerate}
