"""A' training loss:  L = mean_valid(W * L_rec) + beta * L_planar,  W = 1 + alpha * E.

``L_rec`` is WAVE's own (``Components/SimpleGradientL1Loss.py``, used as
``SimpleGradientL1Loss(lambda_grad=0.1)`` in ``WAVE_TrainingHelper``): L1 + 0.1 * (L1 of the
Sobel gradients), on min-max normalised depth.  WAVE averages it over *all* pixels and ignores
holes; here it is evaluated per pixel and averaged over valid GT pixels only (Sobel terms only
where the whole 3x3 window is valid).
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import torch
import torch.nn.functional as F

from models.planar_loss import planar_loss

_SX = torch.tensor([[-1.0, 0, 1], [-2, 0, 2], [-1, 0, 1]]).view(1, 1, 3, 3)


def sobel(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    kx = _SX.to(x)
    return F.conv2d(x, kx, padding=1), F.conv2d(x, kx.transpose(2, 3), padding=1)


def rec_loss_map(pred, gt, valid, lambda_grad: float = 0.1) -> torch.Tensor:
    """Per-pixel WAVE L_rec [B,1,H,W], 0 on invalid pixels."""
    v = valid.float()
    gv = (F.avg_pool2d(v, 3, 1, 1, count_include_pad=False) > 0.999).float()
    gx_p, gy_p = sobel(pred)
    gx_t, gy_t = sobel(gt * v)
    return (pred - gt).abs() * v + lambda_grad * ((gx_p - gx_t).abs() + (gy_p - gy_t).abs()) * gv


@dataclass
class LossConfig:
    alpha: float = 0.0  # edge weight; 0 = off
    beta: float = 0.0  # planar weight; 0 = off
    planar_warmup_steps: int = 1000
    huber_delta: float = 0.1
    normal_source: str = "gt"
    min_tread_px: int = 500
    lambda_grad: float = 0.1
    gate_cos_tread: float = 0.85
    gate_cos_riser: float = 0.5
    min_inlier_frac: float = 0.5


class AprimeLoss:
    def __init__(self, cfg: LossConfig, K=None) -> None:
        self.cfg, self.K = cfg, K
        if cfg.beta > 0 and K is None:
            warnings.warn("[loss] use_planar_loss requested but intrinsics K are not set: planar loss DISABLED")
            self.cfg = LossConfig(**{**cfg.__dict__, "beta": 0.0})

    def beta(self, step: int) -> float:
        c = self.cfg
        return c.beta * min(1.0, step / c.planar_warmup_steps) if c.planar_warmup_steps > 0 else c.beta

    def __call__(self, pred_n: torch.Tensor, b: dict, step: int = 0) -> dict[str, torch.Tensor]:
        """``b``: gt_n, valid, E, dmin, dmax, tread, riser, band, ignore (all [B,1,H,W] / [B,1,1,1])."""
        c = self.cfg
        valid = b["valid"]
        W = 1.0 + c.alpha * b["E"] * (~b["ignore"]).float() if c.alpha > 0 else 1.0
        rec = rec_loss_map(pred_n, b["gt_n"], valid, c.lambda_grad)
        l_rec = (W * rec).sum() / valid.float().sum().clamp_min(1.0)
        out = {"rec": l_rec.detach(), "total": l_rec}
        beta = self.beta(step)
        out["beta"] = torch.tensor(beta)
        if c.beta > 0:
            sc = b["dmax"] - b["dmin"]
            pred_m, gt_m = pred_n * sc + b["dmin"], b["gt_n"] * sc + b["dmin"]
            l_pl = planar_loss(
                pred_m, gt_m, valid, b["tread"], b["riser"], b["band"] | b["ignore"], self.K,
                c.normal_source, c.huber_delta, c.min_tread_px,
                gate_cos_tread=c.gate_cos_tread, gate_cos_riser=c.gate_cos_riser, min_inlier_frac=c.min_inlier_frac,
            )
            out["planar"] = l_pl.detach()
            out["total"] = l_rec + beta * l_pl
        return out
