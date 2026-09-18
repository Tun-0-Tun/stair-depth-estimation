"""Adapter for WAVE (2026) -- ``third_party/wave``.

Upstream: https://github.com/tayyabnasir22/WAVE
Paper:    Nasir et al., "WAVE: Reversing the Guidance Hierarchy for Coarse-to-Fine
          Guided Depth Super-Resolution", 2026 (arXiv:2601.17723)

.. important::
   Only x8, x16 and x32 are supported, one checkpoint each.  Scale has no
   weights (just a parameter-free ``nn.Upsample``), so a wrong-scale checkpoint
   loads without error -- ``_load`` checks the scale against the path.

What this adapter absorbs
-------------------------
1. **Min-max normalisation.**  Upstream takes min/max from GT; we take them from
   the input depth's valid (``> 0``) pixels and denormalise with the same pair.
2. **Any depth map in.**  Dense or sparse, low or full resolution: holes are
   nearest-filled, then the map is bicubic-resized to the ``H / scale`` grid.
3. **Image size.**  H and W are padded with zeros up to a multiple of
   ``max(16, scale)`` and the output is cropped back to the original size.
4. **Expired DINOv3 URL.**  Always ``use_pretrained=False`` as the checkpoint
   already holds the DINO weights.
5. **RGB standardisation.**  ImageNet mean/std by default, ``rgb_norm="unit"``
   for plain ``[0, 1]``. 
"""

from __future__ import annotations

import importlib
import importlib.util
import re
from pathlib import Path
from typing import Any

import numpy as np

from baselines.base import Availability, BaselineModel, add_third_party_to_path, register_baseline
from utils.misc import resize_depth, sparse_to_dense_nn

__all__ = ["WaveAdapter"]

_SUPPORTED_SCALES = (8, 16, 32)
_DINO_PATCH = 16
_IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
# "x16" (our layout) or "Scale_16" (upstream's model_states_* folder names)
_SCALE_IN_PATH = re.compile(r"(?:scale_|(?<![a-z0-9])x)(\d+)(?!\d)", re.IGNORECASE)


def _pad_multiple(scale: int) -> int:
    return max(_DINO_PATCH, int(scale))


def _padded_hw(hw: tuple[int, int], scale: int) -> tuple[int, int]:
    m = _pad_multiple(scale)
    return (-(-hw[0] // m) * m, -(-hw[1] // m) * m)


def _prepare_lr(
    depth: np.ndarray, hw_pad: tuple[int, int], scale: int
) -> tuple[np.ndarray | None, float, float]:
    """Any depth map -> (normalised dense LR on the WAVE grid, d_min, d_max).

    Returns ``(None, d_min, d_min)`` when the normalisation is undefined.
    """
    depth = np.nan_to_num(np.asarray(depth, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    valid = depth > 0
    if not valid.any():
        return None, 0.0, 0.0

    # 1. min/max from the input's own measurements, before anything is invented
    d_min, d_max = float(depth[valid].min()), float(depth[valid].max())
    if d_max - d_min < 1e-6:
        return None, d_min, d_min

    # 2. fill holes first, then onto WAVE's LR grid
    dense = depth if valid.all() else sparse_to_dense_nn(depth)
    lr = resize_depth(dense, (hw_pad[0] // scale, hw_pad[1] // scale), mode="bicubic")

    # bicubic can overshoot the input range slightly
    lr_n = np.clip((lr - d_min) / (d_max - d_min), 0.0, 1.0)
    return lr_n.astype(np.float32), d_min, d_max


def _check_checkpoint_scale(checkpoint_path: Path | str | None, scale: int) -> str | None:
    """Raise if the path names another scale; return a warning if it names none."""
    if checkpoint_path is None:
        return None
    found = {int(s) for s in _SCALE_IN_PATH.findall(str(checkpoint_path))}
    if not found:
        return (
            f"[wave] cannot tell the scale of {checkpoint_path} from its path; "
            f"make sure it is the x{scale} checkpoint (the weights cannot tell)"
        )
    if scale not in found:
        raise RuntimeError(
            f"[wave] model.scale={scale} but the checkpoint path {checkpoint_path} is for "
            f"x{'/x'.join(str(s) for s in sorted(found))}. WAVE loads any scale's weights "
            "without error and then predicts garbage, so this is refused. Set model.scale "
            "and model.checkpoint together."
        )
    return None


@register_baseline("wave")
class WaveAdapter(BaselineModel):
    """WAVE: RGB + depth (dense or sparse, any resolution) -> dense high-resolution depth."""

    input_modality = "lr"
    native_size = None  # any size; padded to a multiple of max(16, scale) internally
    paper = "Nasir et al., WAVE, 2026 (arXiv:2601.17723)"
    submodule = "wave"

    def __init__(
        self,
        scale: int = 8,
        num_feats: int = 48,
        patch_size: int = _DINO_PATCH,
        rgb_norm: str = "imagenet",
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        if int(scale) not in _SUPPORTED_SCALES:
            raise ValueError(f"scale must be one of {_SUPPORTED_SCALES}, got {scale!r}")
        if rgb_norm not in ("imagenet", "unit"):
            raise ValueError(f"rgb_norm must be 'imagenet' or 'unit', got {rgb_norm!r}")
        self.scale = int(scale)
        self.num_feats = int(num_feats)
        self.patch_size = int(patch_size)
        #: (H, W) actually fed to the network; set per call from the input image in _predict
        self.img_size: tuple[int, int] | None = None
        self.rgb_norm = rgb_norm
        self.n_degenerate = 0

    def describe(self) -> dict[str, Any]:
        return {
            **super().describe(),
            "scale": self.scale,
            "img_size": list(self.img_size) if self.img_size else None,
            "rgb_norm": self.rgb_norm,
            "n_degenerate": self.n_degenerate,
        }

    # ------------------------------------------------------------ checks

    @classmethod
    def availability(cls, checkpoint_path: str | Path | None = None, **kwargs: Any) -> Availability:
        reasons: list[str] = []
        instructions: list[str] = []

        root = Path(__file__).resolve().parent.parent / "third_party" / "wave"
        if not (root / "Components" / "WAVE.py").exists():
            reasons.append("third_party/wave is not checked out")
            instructions.append("git submodule update --init --recursive")

        for mod, hint in (
            ("torch", "pip install torch"),
            ("torchvision", "pip install torchvision"),
            ("pywt", "pip install PyWavelets"),
            ("pytorch_wavelets", "pip install pytorch-wavelets"),
        ):
            if importlib.util.find_spec(mod) is None:
                reasons.append(f"missing python package: {mod}")
                instructions.append(hint)

        scale = kwargs.get("scale", 8)
        ckpt = kwargs.get("checkpoint") or checkpoint_path
        if not ckpt or not Path(ckpt).exists():
            reasons.append(f"WAVE x{scale} checkpoint not found: {ckpt or '<unset>'}")
            instructions.append(
                "gdown --folder https://drive.google.com/drive/folders/"
                "1uY5uzU8AAKafeoN_hbuC3bxVN3WJXhin -O checkpoints/wave_upstream, then copy "
                f"model_states_WAVE_*_Scale_{scale}/last.pth to checkpoints/wave/x{scale}/last.pth "
                "and set model.checkpoint"
            )

        return Availability(ok=not reasons, reasons=reasons, instructions=instructions)

    # -------------------------------------------------------------- load

    def _load(self, checkpoint_path: Path | None) -> Any:
        import torch

        warning = _check_checkpoint_scale(checkpoint_path, self.scale)
        if warning:
            print(warning)

        add_third_party_to_path("wave")
        WAVE = importlib.import_module("Components.WAVE").WAVE

        # 4. never True: the DINOv3 URL behind it has expired.
        # img_size is left at WAVE's default: it creates no weights, and the real
        # size is only known once _predict sees an image (see _set_img_size).
        model = WAVE(
            use_pretrained=False,
            num_feats=self.num_feats,
            patch_size=self.patch_size,
            scale=self.scale,
        )

        if checkpoint_path is not None:
            ckpt = torch.load(checkpoint_path, map_location="cpu")
            state = ckpt.get("model", ckpt) if isinstance(ckpt, dict) else ckpt
            state = {k.replace("module.", "", 1): v for k, v in state.items()}
            missing, unexpected = model.load_state_dict(state, strict=False)
            if missing:
                raise RuntimeError(
                    f"[wave] checkpoint {checkpoint_path} is missing {len(missing)} tensors, "
                    f"e.g. {missing[:5]}. Wrong num_feats? (num_feats={self.num_feats})"
                )
            if unexpected:
                print(f"[wave] ignoring {len(unexpected)} unexpected tensors in the checkpoint")

        return model.to(self.device).eval()

    # ----------------------------------------------------------- predict

    def _set_img_size(self, hw: tuple[int, int]) -> None:
        """Record the input size on DINOv3's PatchEmbed, the only place WAVE keeps it.

        Metadata only (patch grid / FLOP count): RoPE makes the encoder
        size-agnostic, so no weights depend on it.
        """
        if self.img_size == hw:
            return
        self.img_size = hw
        pe = self.model.semantics_encoder.patch_embed
        pe.img_size = hw
        pe.patches_resolution = (hw[0] // self.patch_size, hw[1] // self.patch_size)
        pe.num_patches = pe.patches_resolution[0] * pe.patches_resolution[1]

    def _predict(self, rgb: np.ndarray, depth_in: np.ndarray, **kwargs: Any) -> np.ndarray:
        import torch

        h, w = rgb.shape[:2]
        h_pad, w_pad = _padded_hw((h, w), self.scale)
        self._set_img_size((h_pad, w_pad))

        # 1. + 2. any depth -> normalised dense LR, min/max from the input only
        lr_n, d_min, d_max = _prepare_lr(depth_in, (h_pad, w_pad), self.scale)
        if lr_n is None:
            self.n_degenerate += 1
            return np.clip(np.full((h, w), d_min, dtype=np.float32), self.min_depth, self.max_depth)

        # 5. RGB standardisation (unverified against upstream, see module docstring)
        rgb_n = (rgb - _IMAGENET_MEAN) / _IMAGENET_STD if self.rgb_norm == "imagenet" else rgb
        # 3. zero-pad bottom/right, as upstream's GetInference16
        rgb_p = np.zeros((h_pad, w_pad, 3), dtype=np.float32)
        rgb_p[:h, :w] = rgb_n

        image = torch.from_numpy(rgb_p.transpose(2, 0, 1))[None].to(self.device).float()
        lr = torch.from_numpy(lr_n)[None, None].to(self.device).float()
        with torch.no_grad():
            out = self.model(image, lr)

        pred_n = out.squeeze().detach().float().cpu().numpy()[:h, :w]
        pred = pred_n * (d_max - d_min) + d_min  # back to metres
        pred = np.clip(pred, self.min_depth, self.max_depth)
        return pred.astype(np.float32)
