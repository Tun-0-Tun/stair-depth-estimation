"""Adapter for DEPTHOR (ICCV 2025) -- ``third_party/depthor``.

Upstream: https://github.com/ShadowBbBb/Depthor
Paper:    Xiang et al., "DEPTHOR: Depth Enhancement from a Practical
          Light-Weight dToF Sensor and RGB Image", ICCV 2025 (arXiv:2504.01596)

.. important::
   **DEPTHOR++ (arXiv:2509.26498) has no public code or weights** as of
   2026-09-01.  This adapter runs DEPTHOR v1, the same authors' released ICCV
   2025 model.  ``configs/model/depthor_plus_plus.yaml`` exists but is marked
   unavailable and the ``depthor_plus_plus`` key below fails loudly rather than
   silently substituting v1 -- a v1 number must never be reported in a
   DEPTHOR++ row.  See ``docs/metrics_protocol.md`` for the status and the
   options we have.

What this adapter absorbs
-------------------------
1. **Import isolation.**  The upstream package assumes it is the process's root
   and parses ``argparse`` at import time (``src/config.py``).  We import only
   ``src.models.depthor`` and never ``src.config``.
2. **A hard-coded checkpoint path.**  ``src/utils/set_mde.py`` loads Depth
   Anything V2 from ``/home/xjj/code/Depth-Anything-V2/checkpoints/...``.  We
   monkey-patch ``set_depthanything`` before constructing the model so the
   weights come from our ``depth_anything_ckpt`` option instead.  We patch the
   function rather than the file so the submodule stays pristine.
3. **Fixed geometry.**  ``forward`` hard-codes the ``zju_l`` transform config,
   which assumes a 480x640 input.  We resize in and out.
4. **Input construction.**  The network wants ``{'image': (B,3,H,W) in [0,1],
   'sparse_depth': (B,1,H,W) metres, 0 = no measurement}`` and returns
   ``(coarse, final)``; we take ``final`` and clamp to the dataset range, which
   is what ``evaluate.py::predict_tta`` does.
5. **The CUDA dependency.**  The CSPN++ refinement imports ``BpOps``, a CUDA
   extension from BP-Net that only builds against CUDA 12.1.  There is no CPU
   or MPS path, so this baseline cannot run on a Mac.  ``availability()`` says
   so up front instead of dying inside ``import``.
"""

from __future__ import annotations

import importlib
from pathlib import Path
from typing import Any

import numpy as np

from baselines.base import Availability, BaselineModel, add_third_party_to_path, register_baseline
from utils.misc import resize_depth, resize_rgb

__all__ = ["DepthorAdapter", "DepthorPlusPlusAdapter"]

_DEPTHOR_INPUT_HW = (480, 640)


@register_baseline("depthor")
class DepthorAdapter(BaselineModel):
    """DEPTHOR v1: RGB + sparse dToF -> dense metric depth."""

    input_modality = "sparse"
    native_size = _DEPTHOR_INPUT_HW
    paper = "Xiang et al., DEPTHOR, ICCV 2025 (arXiv:2504.01596)"
    submodule = "depthor"

    def __init__(
        self,
        variant: str = "large",
        n_bins: int = 256,
        depth_anything_ckpt: str | None = "checkpoints/depth_anything_v2_vits.pth",
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        if variant not in ("large", "small"):
            raise ValueError(f"variant must be 'large' or 'small', got {variant!r}")
        self.variant = variant
        self.n_bins = int(n_bins)
        self.depth_anything_ckpt = depth_anything_ckpt

    # ------------------------------------------------------------ checks

    @classmethod
    def availability(cls, checkpoint_path: str | Path | None = None, **kwargs: Any) -> Availability:
        reasons: list[str] = []
        instructions: list[str] = []

        root = Path(__file__).resolve().parent.parent / "third_party" / "depthor"
        if not (root / "src" / "models" / "depthor.py").exists():
            reasons.append("third_party/depthor is not checked out")
            instructions.append("git submodule update --init --recursive")

        for mod, hint in (
            ("torch", "pip install torch"),
            ("torchvision", "pip install torchvision"),
            ("timm", "pip install timm"),
            ("cv2", "pip install opencv-python-headless"),
        ):
            if importlib.util.find_spec(mod) is None:
                reasons.append(f"missing python package: {mod}")
                instructions.append(hint)

        if importlib.util.find_spec("BpOps") is None:
            reasons.append(
                "missing CUDA extension 'BpOps' (CSPN++ refinement, from BP-Net; CUDA 12.1 only)"
            )
            instructions.append(
                "on a CUDA 12.1 machine: git clone https://github.com/kakaxi314/BP-Net && "
                "cd BP-Net && python setup.py install   (there is no CPU/MPS build)"
            )

        ckpt = kwargs.get("checkpoint") or checkpoint_path
        if not ckpt or not Path(ckpt).exists():
            reasons.append(f"DEPTHOR checkpoint not found: {ckpt or '<unset>'}")
            instructions.append(
                "download Depthor-ZJU-Large/Small from the links in the upstream README "
                "(third_party/depthor/README.md) into checkpoints/ and set "
                "model.checkpoint in the config -- see docs/datasets.md"
            )

        da = kwargs.get("depth_anything_ckpt", "checkpoints/depth_anything_v2_vits.pth")
        if da and not Path(da).exists():
            reasons.append(f"Depth Anything V2 backbone weights not found: {da}")
            instructions.append(
                "wget https://huggingface.co/depth-anything/Depth-Anything-V2-Small/resolve/main/"
                "depth_anything_v2_vits.pth -P checkpoints/"
            )

        return Availability(ok=not reasons, reasons=reasons, instructions=instructions)

    # -------------------------------------------------------------- load

    def _patch_depth_anything(self) -> None:
        """Redirect the hard-coded Depth Anything V2 checkpoint path."""
        import torch

        set_mde = importlib.import_module("src.utils.set_mde")
        ckpt = self.depth_anything_ckpt
        configs = {
            "vits": {"encoder": "vits", "features": 64, "out_channels": [48, 96, 192, 384]},
            "vitb": {"encoder": "vitb", "features": 128, "out_channels": [96, 192, 384, 768]},
            "vitl": {"encoder": "vitl", "features": 256, "out_channels": [256, 512, 1024, 1024]},
        }

        def set_depthanything(encoder: str = "vits"):
            from src.models.depth_anything_v2.dpt import DepthAnythingV2

            model = DepthAnythingV2(**configs[encoder])
            if ckpt:
                model.load_state_dict(torch.load(ckpt, map_location="cpu"))
            return model

        set_mde.set_depthanything = set_depthanything

    def _load(self, checkpoint_path: Path | None) -> Any:
        import torch

        add_third_party_to_path("depthor")
        self._patch_depth_anything()

        module = "src.models.depthor" if self.variant == "large" else "src.models.depthor_s"
        Depthor = importlib.import_module(module).Depthor

        model = Depthor(
            n_bins=self.n_bins,
            min_val=self.min_depth,
            max_val=self.max_depth,
            norm="linear",
        )
        if checkpoint_path is not None:
            state = torch.load(checkpoint_path, map_location="cpu")
            state = state.get("model", state) if isinstance(state, dict) else state
            state = {k.replace("module.", "", 1): v for k, v in state.items()}
            missing, unexpected = model.load_state_dict(state, strict=False)
            if missing:
                raise RuntimeError(
                    f"[depthor] checkpoint {checkpoint_path} is missing {len(missing)} tensors, "
                    f"e.g. {missing[:5]}. Wrong variant? (variant={self.variant!r})"
                )
            if unexpected:
                print(f"[depthor] ignoring {len(unexpected)} unexpected tensors in the checkpoint")

        model = model.to(self.device)
        model.set_extra_param(device=self.device)
        return model.eval()

    # ----------------------------------------------------------- predict

    def _predict(self, rgb: np.ndarray, depth_in: np.ndarray, **kwargs: Any) -> np.ndarray:
        import torch

        h, w = rgb.shape[:2]
        th, tw = self.native_size

        rgb_r = resize_rgb(rgb, (th, tw))
        # nearest, never bilinear: interpolating a sparse map smears zeros into
        # the measurements and invents depth between them.
        sparse_r = resize_depth(depth_in, (th, tw), mode="nearest")

        image = torch.from_numpy(rgb_r.transpose(2, 0, 1))[None].to(self.device).float()
        sparse = torch.from_numpy(sparse_r)[None, None].to(self.device).float()

        with torch.no_grad():
            _, final = self.model({"image": image, "sparse_depth": sparse})

        pred = final.squeeze().detach().float().cpu().numpy()
        pred = np.clip(pred, self.min_depth, self.max_depth)
        return resize_depth(pred, (h, w), mode="bilinear")


@register_baseline("depthor_plus_plus")
class DepthorPlusPlusAdapter(DepthorAdapter):
    """DEPTHOR++ -- **not runnable**: the authors have not released code or weights.

    Kept as a registered key so the experiment configs, the benchmark matrix and
    the docs can name the method the SOW requires, while any attempt to actually
    run it fails with an explanation instead of quietly producing v1 numbers
    under a v1.5 label.

    When upstream publishes: point the ``depthor`` submodule at the release (or
    add a second submodule), then this class only needs its ``_load`` to select
    the new architecture -- everything else in ``DepthorAdapter`` still applies.
    """

    paper = "Xiang et al., DEPTHOR++, 2025 (arXiv:2509.26498) -- code unreleased"

    _MESSAGE = (
        "DEPTHOR++ (arXiv:2509.26498) has no public implementation or weights.\n"
        "The upstream repo https://github.com/ShadowBbBb/Depthor contains DEPTHOR v1 "
        "(ICCV 2025) only, and the DEPTHOR++ paper links no code.\n"
        "Options, in the order we recommend them:\n"
        "  1. run `model=depthor` (v1) and label the row DEPTHOR-v1 -- honest, and v1 is\n"
        "     itself SOTA on ZJU-L5; the ++ delta is +22% RMSE per their abstract.\n"
        "  2. email the authors (see the paper) asking for weights or an early release.\n"
        "  3. re-implement ++ from the paper -- only with an explicit decision from the\n"
        "     team, and it must live in models/, never in baselines/.\n"
        "See docs/metrics_protocol.md :: 'Baseline availability'."
    )

    @classmethod
    def availability(cls, checkpoint_path: str | Path | None = None, **kwargs: Any) -> Availability:
        return Availability(
            ok=False,
            reasons=["DEPTHOR++ code and weights are not publicly released"],
            instructions=cls._MESSAGE.splitlines(),
        )

    def _load(self, checkpoint_path: Path | None) -> Any:
        raise RuntimeError(self._MESSAGE)
