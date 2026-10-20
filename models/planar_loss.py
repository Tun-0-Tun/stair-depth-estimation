"""Planar stair loss: treads are parallel planes (normal = "up"), risers are perpendicular to it.

Idea: "stairs as parallel planes, riser/tread told apart by the normal's angle to the up
vector" -- Perez-Yus et al., ECCV-W 2014 and CVIU 2017; the transfer into a training loss is ours.

Per frame: depth (m) -> point cloud with intrinsics ``K = (fx, fy, cx, cy)`` (or an undistorted ray map ``[2,H,W]``) -> normals
(cross product of central differences, camera-facing, masked where any neighbour is invalid
or across a depth jump) -> up vector ``u`` = robust mean of the tread normals ->
tread: ``1 - |n.u|``, riser: ``|n.u|`` through Huber.  Frames with fewer than
``min_tread_px`` usable tread pixels contribute 0.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def box_smooth(z: torch.Tensor, valid: torch.Tensor, k: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Normalised box filter that only averages valid pixels; result valid where the window is full."""
    if k <= 1:
        return z, valid
    v = valid.float()
    num = F.avg_pool2d(z * v, k, 1, k // 2, count_include_pad=False)
    den = F.avg_pool2d(v, k, 1, k // 2, count_include_pad=False)
    return num / den.clamp_min(1e-6), den > 0.999


def normals_from_depth(
    z: torch.Tensor, K, valid: torch.Tensor, max_rel_jump: float = 0.15
) -> tuple[torch.Tensor, torch.Tensor]:
    """``z`` [B,1,H,W] metres -> unit normals [B,3,H,W] facing the camera (n.P < 0), validity [B,1,H,W]."""
    B, _, H, W = z.shape
    if hasattr(K, "shape") and len(K.shape) == 3:  # undistorted ray map [2,H,W] (utils/camera.py)
        r = torch.as_tensor(K).to(z)[None]
        P = torch.cat([r[:, 0:1] * z, r[:, 1:2] * z, z], dim=1)
    else:  # pinhole (fx, fy, cx, cy)
        fx, fy, cx, cy = (float(k) for k in K)
        v = torch.arange(H, device=z.device, dtype=z.dtype).view(1, 1, H, 1)
        u = torch.arange(W, device=z.device, dtype=z.dtype).view(1, 1, 1, W)
        P = torch.cat([(u - cx) / fx * z, (v - cy) / fy * z, z], dim=1)
    dPu = P[:, :, 1:-1, 2:] - P[:, :, 1:-1, :-2]
    dPv = P[:, :, 2:, 1:-1] - P[:, :, :-2, 1:-1]
    n = torch.cross(dPu, dPv, dim=1)
    n = n / n.norm(dim=1, keepdim=True).clamp_min(1e-12)
    n = torch.where((n * P[:, :, 1:-1, 1:-1]).sum(1, keepdim=True) > 0, -n, n)  # face the camera (n.P < 0)
    c = z[:, :, 1:-1, 1:-1]
    ok = valid[:, :, 1:-1, 1:-1] & valid[:, :, 1:-1, 2:] & valid[:, :, 1:-1, :-2]
    ok = ok & valid[:, :, 2:, 1:-1] & valid[:, :, :-2, 1:-1]
    for nb in (z[:, :, 1:-1, 2:], z[:, :, 1:-1, :-2], z[:, :, 2:, 1:-1], z[:, :, :-2, 1:-1]):
        ok = ok & ((nb - c).abs() < max_rel_jump * c.clamp_min(1e-3))
    n, ok = F.pad(n, (1, 1, 1, 1)), F.pad(ok, (1, 1, 1, 1))
    return n, ok


def robust_up(n: torch.Tensor, sel: torch.Tensor, min_px: int, cos_inlier: float = 0.9, iters: int = 3):
    """Per-frame up vector from tread normals. ``n`` [B,3,H,W], ``sel`` [B,1,H,W] bool -> ([B,3], [B] ok)."""
    B = n.shape[0]
    up = n.new_zeros(B, 3)
    ok = torch.zeros(B, dtype=torch.bool, device=n.device)
    for b in range(B):
        x = n[b][:, sel[b, 0]].T  # [N,3]
        if x.shape[0] < min_px:
            continue
        u = F.normalize(x.median(dim=0).values, dim=0)
        for _ in range(iters):
            inl = x[(x @ u) > cos_inlier]
            if inl.shape[0] == 0:
                break
            u = F.normalize(inl.mean(0), dim=0)
        up[b], ok[b] = u, True
    return up, ok


def planar_loss(
    pred_m: torch.Tensor,
    gt_m: torch.Tensor,
    valid: torch.Tensor,
    tread: torch.Tensor,
    riser: torch.Tensor,
    exclude: torch.Tensor,
    K,
    source: str = "gt",
    delta: float = 0.1,
    min_tread_px: int = 500,
    gt_smooth_k: int = 5,
) -> torch.Tensor:
    """Scalar loss. All masks [B,1,H,W] bool; depths in metres. ``exclude`` = edge band | low-confidence."""
    if source == "gt":
        zs, vs = box_smooth(gt_m, valid, gt_smooth_k)
        n_src, ok_src = normals_from_depth(zs, K, vs)
    elif source == "pred":
        n_src, ok_src = normals_from_depth(pred_m.detach(), K, valid)
    else:
        raise ValueError(f"normal source must be 'gt' or 'pred', got {source!r}")
    n_src = n_src.detach()
    good = valid & ~exclude
    up, has_up = robust_up(n_src, tread & ok_src & good, min_tread_px)

    n, ok = normals_from_depth(pred_m, K, valid)
    cos = (n * up[:, :, None, None]).sum(1, keepdim=True).abs()
    r = torch.where(tread, 1.0 - cos, cos)
    use = (tread | riser) & ok & good & has_up[:, None, None, None]
    h = F.huber_loss(r, torch.zeros_like(r), delta=delta, reduction="none")
    per = (h * use).flatten(1).sum(1) / use.flatten(1).sum(1).clamp_min(1)
    return (per * has_up).mean()
