"""Run the official StairNet-DepthOut (RGB-only, StairNetV3) on one RGB frame.

The authors' repo (third_party/StairNet-DepthOut, CC BY-NC 3.0) is imported, not copied.  It is
a script collection with top-level modules (``nets``, ``config``, ``utils``) that collide with
ours, so the import is isolated and the entries removed from ``sys.modules`` afterwards.

Their input convention (detect.py): RGB, ``(x/255 - 0.5)/0.5``, zero-padded to a square
(rows added top/bottom for landscape frames), bilinear resize to 512x512.  We invert exactly
that: probabilities are resized back to the padded square, the padding is cropped away, so the
returned maps are in the original frame.  Pixels that came from padding are never returned.
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

from utils.misc import REPO_ROOT, data_root

_REPO = REPO_ROOT / "third_party" / "StairNet-DepthOut"
STRIDE = (8, 16)  # (h, w) of the 64x32 line-cell grid on the 512 canvas


def import_stairnet_net():
    """``StairNet_DepthOut`` class from the third-party repo without polluting ``sys.modules``."""
    if not (_REPO / "nets" / "NetV3.py").exists():
        raise FileNotFoundError(f"{_REPO} missing; clone MrChenWang/StairNet-DepthOut there")
    clash = {k: sys.modules.pop(k) for k in list(sys.modules) if k == "nets" or k.startswith("nets.")}
    sys.path.insert(0, str(_REPO))
    try:
        mod = importlib.import_module("nets.NetV3")
    finally:
        sys.path.remove(str(_REPO))
        for k in [k for k in sys.modules if k == "nets" or k.startswith("nets.")]:
            del sys.modules[k]
        sys.modules.update(clash)
    return mod.StairNet_DepthOut


def pad_square(img: np.ndarray) -> tuple[np.ndarray, tuple[int, int]]:
    """Zero-pad to a square; returns ``(padded, (pad_top, pad_left))``."""
    h, w = img.shape[:2]
    top = (w - h) // 2 if w > h else 0
    left = (h - w) // 2 if h > w else 0
    side = max(h, w)
    out = np.zeros((side, side) + img.shape[2:], img.dtype)
    out[top : top + h, left : left + w] = img
    return out, (top, left)


class StairNetSeg:
    """``predict(rgb01) -> {cls, conf, lines}`` in the frame of ``rgb01``."""

    def __init__(
        self,
        weights: str | Path = "weights/stairnet/depth_out/best.pth",
        width_factor: float = 1.0,
        device: str = "cuda",
        tread_class: int = 1,
        riser_class: int = 2,
        line_conf: float = 0.5,
    ) -> None:
        w = Path(weights)
        w = w if w.is_absolute() else data_root() / w
        if not w.exists():
            raise FileNotFoundError(
                f"StairNet weights not found: {w}\n"
                "Put the DepthOut weights (Baidu link in third_party/StairNet-DepthOut/README.md) "
                "into $STAIR_DATA_ROOT/weights/stairnet/depth_out/."
            )
        net = import_stairnet_net()(width=width_factor)
        sd = torch.load(w, map_location="cpu", weights_only=True)
        sd = sd.get("state_dict", sd.get("model", sd))
        net.load_state_dict({k.replace("module.", "", 1): v for k, v in sd.items()})
        self.net, self.device = net.to(device).eval(), device
        # UNVERIFIED default: which of the net's channels 1/2 is tread/riser must be checked
        # on the visualisations (scripts/stairnet_viz.py) before the masks are used.
        self.class_map = {0: 0, tread_class: 1, riser_class: 2}
        self.line_conf = line_conf

    @torch.no_grad()
    def predict(self, rgb01: np.ndarray, size: int = 512) -> dict[str, np.ndarray]:
        h, w = rgb01.shape[:2]
        sq, (top, left) = pad_square(np.ascontiguousarray(rgb01, np.float32))
        side = sq.shape[0]
        x = cv2.resize(sq, (size, size), interpolation=cv2.INTER_LINEAR)
        x = torch.from_numpy((x - 0.5) / 0.5).permute(2, 0, 1)[None].to(self.device).float()
        fb1, fb2, fr1, fr2, fm, _ = self.net(x)
        prob = torch.softmax(fm, dim=1)[0].permute(1, 2, 0).cpu().numpy()
        prob = cv2.resize(prob, (side, side), interpolation=cv2.INTER_LINEAR)
        prob = prob[top : top + h, left : left + w]  # drop the padding
        raw = prob.argmax(-1)
        cls = np.zeros_like(raw, np.uint8)
        for src, dst in self.class_map.items():
            cls[raw == src] = dst
        lines = []
        for k, (c, loc) in enumerate(((fb1, fb2), (fr1, fr2))):
            c, loc = c[0, 0].cpu().numpy(), loc[0].cpu().numpy()
            for i, j in zip(*np.nonzero(c >= self.line_conf)):
                x1, y1 = (j + loc[0, i, j]) * STRIDE[1], (i + loc[1, i, j]) * STRIDE[0]
                x2, y2 = (j + loc[2, i, j]) * STRIDE[1], (i + loc[3, i, j]) * STRIDE[0]
                f = side / size  # 512 canvas -> padded square -> frame
                lines.append([k, x1 * f - left, y1 * f - top, x2 * f - left, y2 * f - top, c[i, j]])
        return {
            "cls": cls,
            "conf": prob.max(-1).astype(np.float32),
            "lines": np.asarray(lines, np.float32).reshape(-1, 6),
        }
