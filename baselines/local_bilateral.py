"""Local affine calibration of a monocular prior.

Not a published method: our own, ported from the earlier calibration work. It has
no trained parameters of its own, but it is **not** a no-learning baseline the way
``bicubic`` and ``nn_fill`` are -- it leans on Depth Anything V2, a large
pretrained backbone. What it has is no *task-specific* training: nothing here was
fitted to depth completion, to stairs, or to any of our datasets.

That is what makes it worth having. DuCos builds on exactly the same DAv2 vits
backbone and checkpoint, and DEPTHOR on the same family, so this is their natural
ablation: it measures how far the backbone plus a handful of sparse metric
anchors gets you, and therefore how much of their published margin comes from
their own trained machinery rather than from the foundation model underneath.

It also fills a second gap. ``bicubic`` and ``nn_fill`` ignore RGB entirely, so
part of any RGB-guided network's margin over them is simply "it looks at the
image at all" -- with no RGB-aware comparison that part cannot be separated out.

Division of labour inside the method: the dense structure comes from the
monocular network, the absolute scale from the sparse sensor. Neither source
solves the task alone -- drop DAv2 and there is nothing left to calibrate, which
puts you back at ``nn_fill``.

The method, per frame:

1. Depth Anything V2 gives a *relative* (inverse-depth-like) map ``d_rel``.
2. SLIC cuts the RGB into ``n_segments`` superpixels.
3. In each superpixel, a closed-form least-squares fit of its own ``s_k, t_k``
   mapping ``d_rel`` to metres, using only the sparse input points that fall
   inside it. Superpixels with fewer than ``min_pixels`` points fall back to a
   single global fit.
4. The piecewise-constant ``s(x, y)``, ``t(x, y)`` fields are smoothed:
   ``none``, ``gaussian``, or ``bilateral``.
5. ``D = s(x, y) * d_rel + t(x, y)``.

In the ``bilateral`` mode the range term is computed on ``(s, t)`` themselves,
not on RGB -- it is edge-preserving smoothing of the calibration field, not
cross-filtering by the image. The image enters earlier, through SLIC.

.. important::
   The fit uses ``depth_in`` -- the sparse sensor input -- and never
   ``gt_depth``. Fitting to the ground truth would be fitting to the answer, and
   every metric would become meaningless while still looking plausible.

Two deliberate departures from the original implementation, both because the
original was too slow and too memory-hungry to put in a results table:

* the per-superpixel fit was a Python loop that masked the whole frame once per
  superpixel (~200 passes over 307k pixels); here it is a single pass with
  ``np.bincount``, which is exact, not an approximation;
* the bilateral step built an ``(H, W, 2r+1, 2r+1)`` array -- 542 MB per
  temporary at 480x640 with ``r=10``, several of them at once. Here the window
  is accumulated offset by offset, so memory is O(H*W) instead.

Depth Anything V2 is borrowed from ``third_party/ducos``, which vendors it, so
this baseline needs no submodule of its own; the checkpoint is the one DuCos and
DEPTHOR already use.
"""

from __future__ import annotations

import importlib
import importlib.util
from pathlib import Path
from typing import Any

import numpy as np

from baselines.base import Availability, BaselineModel, add_third_party_to_path, register_baseline

__all__ = ["LocalBilateralBaseline"]

_DA2_MODULE = "model.ours.Depth_Anything_V2.depth_anything_v2.dpt"
_DA2_ARGS = {"encoder": "vits", "features": 64, "out_channels": [48, 96, 192, 384]}
_SMOOTH_MODES = ("none", "gaussian", "bilateral")


def _affine_ls(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """Least-squares ``s, t`` for ``y ~ s*x + t``; ``(0, mean y)`` if degenerate."""
    if x.size == 0:
        return 0.0, 0.0
    mx, my = float(x.mean()), float(y.mean())
    xc = x - mx
    denom = float((xc * xc).sum())
    if denom <= 1e-8:
        return 0.0, my
    s = float((xc * (y - my)).sum() / denom)
    return s, my - s * mx


def _affine_per_label(
    x: np.ndarray,
    y: np.ndarray,
    labels: np.ndarray,
    valid: np.ndarray,
    n_labels: int,
    min_pixels: int,
    fallback: tuple[float, float],
) -> tuple[np.ndarray, np.ndarray]:
    """Per-label least-squares fit in one pass. Exact, not an approximation."""
    lab = labels[valid]
    xs = x[valid].astype(np.float64)
    ys = y[valid].astype(np.float64)

    n = np.bincount(lab, minlength=n_labels).astype(np.float64)
    sx = np.bincount(lab, weights=xs, minlength=n_labels)
    sy = np.bincount(lab, weights=ys, minlength=n_labels)
    sxx = np.bincount(lab, weights=xs * xs, minlength=n_labels)
    sxy = np.bincount(lab, weights=xs * ys, minlength=n_labels)

    with np.errstate(invalid="ignore", divide="ignore"):
        safe = np.maximum(n, 1.0)
        mx, my = sx / safe, sy / safe
        denom = sxx - n * mx * mx
        s = (sxy - n * mx * my) / denom
        t = my - s * mx

    ok = (n >= min_pixels) & (np.abs(denom) > 1e-8) & np.isfinite(s) & np.isfinite(t)
    return np.where(ok, s, fallback[0]), np.where(ok, t, fallback[1])


def _smooth_bilateral(
    s: np.ndarray,
    t: np.ndarray,
    sigma_spatial: float,
    range_scale: float,
    max_radius: int,
    device: str = "cpu",
) -> tuple[np.ndarray, np.ndarray]:
    """Edge-preserving smoothing of the (s, t) fields, accumulated per offset.

    The range term is on (s, t), so superpixels whose calibration agrees blend
    together while a sharp change in the fit stays sharp. Offset-by-offset
    accumulation keeps this O(H*W) in memory; the original materialised the
    whole (H, W, 2r+1, 2r+1) window.

    Runs in torch on ``device``: with the default radius that is 441 offsets of
    full-frame ops, which in float64 numpy cost 0.7 s at 640x480 and 6 s at
    1920x1440 -- the bulk of this method's runtime. float32 agrees with the
    float64 version to ~1e-6 (``tests/test_baseline_adapters.py``).
    """
    import torch
    import torch.nn.functional as F

    r = max(1, min(round(2.5 * sigma_spatial), int(max_radius)))
    sr_s = max(float(np.std(s)), 1e-9) * range_scale
    sr_t = max(float(np.std(t)), 1e-9) * range_scale

    h, w = s.shape
    s_t = torch.as_tensor(np.ascontiguousarray(s), dtype=torch.float32, device=device)
    t_t = torch.as_tensor(np.ascontiguousarray(t), dtype=torch.float32, device=device)
    pad = (r, r, r, r)
    sp = F.pad(s_t[None, None], pad, mode="replicate")[0, 0]
    tp = F.pad(t_t[None, None], pad, mode="replicate")[0, 0]

    acc_s = torch.zeros_like(s_t)
    acc_t = torch.zeros_like(t_t)
    wsum = torch.zeros_like(s_t)

    inv_sp2 = 1.0 / (sigma_spatial**2 + 1e-12)
    for dy in range(-r, r + 1):
        for dx in range(-r, r + 1):
            spatial = float(np.exp(-0.5 * (dy * dy + dx * dx) * inv_sp2))
            if spatial < 1e-6:
                continue
            sn = sp[r + dy : r + dy + h, r + dx : r + dx + w]
            tn = tp[r + dy : r + dy + h, r + dx : r + dx + w]
            ds = (sn - s_t) / sr_s
            dt = (tn - t_t) / sr_t
            weight = spatial * torch.exp(-0.5 * (ds * ds + dt * dt))
            acc_s += weight * sn
            acc_t += weight * tn
            wsum += weight

    wsum = wsum.clamp_min(1e-12)
    return (acc_s / wsum).cpu().numpy(), (acc_t / wsum).cpu().numpy()


@register_baseline("local_bilateral")
class LocalBilateralBaseline(BaselineModel):
    """RGB + sparse depth -> dense metric depth, with no training."""

    input_modality = "sparse"
    native_size = None
    paper = (
        "ours: local affine calibration of a Depth Anything V2 prior (no task-specific training)"
    )
    submodule = "ducos"  # borrows the Depth Anything V2 that DuCos vendors

    def __init__(
        self,
        n_segments: int = 200,
        compactness: float = 10.0,
        min_pixels: int = 10,
        smooth_mode: str = "bilateral",
        sigma: float = 15.0,
        sigma_spatial: float = 6.0,
        range_scale: float = 0.25,
        bilateral_max_radius: int = 10,
        da2_input_size: int = 518,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        if smooth_mode not in _SMOOTH_MODES:
            raise ValueError(f"smooth_mode must be one of {_SMOOTH_MODES}, got {smooth_mode!r}")
        self.n_segments = int(n_segments)
        self.compactness = float(compactness)
        self.min_pixels = int(min_pixels)
        self.smooth_mode = smooth_mode
        self.sigma = float(sigma)
        self.sigma_spatial = float(sigma_spatial)
        self.range_scale = float(range_scale)
        self.bilateral_max_radius = int(bilateral_max_radius)
        self.da2_input_size = int(da2_input_size)
        self.n_degenerate = 0

    # ------------------------------------------------------------ checks

    @classmethod
    def availability(cls, checkpoint_path: str | Path | None = None, **kwargs: Any) -> Availability:
        reasons: list[str] = []
        instructions: list[str] = []

        root = Path(__file__).resolve().parent.parent / "third_party" / "ducos"
        if not (root / "model" / "ours" / "Depth_Anything_V2").exists():
            reasons.append("third_party/ducos is not checked out (it vendors Depth Anything V2)")
            instructions.append("git submodule update --init --recursive")

        for mod, hint in (
            ("torch", "pip install torch"),
            ("cv2", "pip install opencv-python-headless"),
            ("skimage", "uv sync --group baselines  (scikit-image, for SLIC)"),
            ("scipy", "pip install scipy"),
        ):
            if importlib.util.find_spec(mod) is None:
                reasons.append(f"missing python package: {mod}")
                instructions.append(hint)

        ckpt = kwargs.get("checkpoint") or checkpoint_path
        if not ckpt or not Path(ckpt).exists():
            reasons.append(f"Depth Anything V2 checkpoint not found: {ckpt or '<unset>'}")
            instructions.append(
                "wget https://huggingface.co/depth-anything/Depth-Anything-V2-Small/resolve/main/"
                "depth_anything_v2_vits.pth -P checkpoints/"
            )

        return Availability(ok=not reasons, reasons=reasons, instructions=instructions)

    # -------------------------------------------------------------- load

    def _load(self, checkpoint_path: Path | None) -> Any:
        import torch

        if checkpoint_path is None:
            raise RuntimeError(
                "[local_bilateral] needs the Depth Anything V2 checkpoint as model.checkpoint"
            )

        add_third_party_to_path("ducos")
        DepthAnythingV2 = importlib.import_module(_DA2_MODULE).DepthAnythingV2

        model = DepthAnythingV2(**_DA2_ARGS)
        model.load_state_dict(torch.load(checkpoint_path, map_location="cpu"))
        return model.to(self.device).eval()

    # ----------------------------------------------------------- predict

    def _relative_depth(self, rgb: np.ndarray) -> np.ndarray:
        """Monocular relative depth at the frame's own resolution."""
        import torch
        import torch.nn.functional as F

        # image2tensor wants BGR uint8 and picks its own device; we move the
        # tensor to ours rather than let it decide, so --device is honoured.
        bgr = (np.clip(rgb, 0.0, 1.0)[..., ::-1] * 255.0).astype(np.uint8)
        tensor, (h, w) = self.model.image2tensor(bgr, self.da2_input_size)
        with torch.no_grad():
            out = self.model(tensor.to(self.device))
        out = F.interpolate(out[:, None], (h, w), mode="bilinear", align_corners=True)
        return out[0, 0].detach().float().cpu().numpy()

    def _predict(self, rgb: np.ndarray, depth_in: np.ndarray, **kwargs: Any) -> np.ndarray:
        from skimage.segmentation import slic

        h, w = rgb.shape[:2]
        valid = np.isfinite(depth_in) & (depth_in > 0)
        if not valid.any():
            # No anchors at all: there is nothing to calibrate against. Say so in
            # `describe()` rather than inventing a number that looks like depth.
            self.n_degenerate += 1
            return np.full((h, w), float(self.min_depth), dtype=np.float32)

        d_rel = self._relative_depth(rgb)
        labels = slic(
            (np.clip(rgb, 0.0, 1.0) * 255).astype(np.uint8),
            n_segments=self.n_segments,
            compactness=self.compactness,
            start_label=0,
            channel_axis=-1,
        ).astype(np.int64)

        fallback = _affine_ls(d_rel[valid], depth_in[valid])
        s_vals, t_vals = _affine_per_label(
            d_rel,
            depth_in,
            labels,
            valid,
            int(labels.max()) + 1,
            self.min_pixels,
            fallback,
        )
        s_map, t_map = s_vals[labels], t_vals[labels]

        if self.smooth_mode == "gaussian":
            from scipy.ndimage import gaussian_filter

            s_map = gaussian_filter(s_map, sigma=self.sigma)
            t_map = gaussian_filter(t_map, sigma=self.sigma)
        elif self.smooth_mode == "bilateral":
            s_map, t_map = _smooth_bilateral(
                s_map,
                t_map,
                self.sigma_spatial,
                self.range_scale,
                self.bilateral_max_radius,
                device=self.device,
            )

        pred = s_map * d_rel.astype(np.float64) + t_map
        return np.clip(pred, self.min_depth, self.max_depth).astype(np.float32)

    def describe(self) -> dict[str, Any]:
        return {
            **super().describe(),
            "n_segments": self.n_segments,
            "smooth_mode": self.smooth_mode,
            "n_degenerate": self.n_degenerate,
        }
