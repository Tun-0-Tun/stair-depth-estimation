import numpy as np
import pytest
import torch

from data.stairnet.edges import build_edge_maps, rasterize
from models.aprime_losses import AprimeLoss, LossConfig
from models.planar_loss import normals_from_depth, planar_loss

K = (300.0, 300.0, 160.0, 120.0)
H, W = 240, 320


def staircase(n_steps=6, rise=0.17, run=0.28, z0=1.2, cam_h=0.9):
    """Analytic ray-cast of an ascending staircase (camera y points down). -> depth, tread, riser."""
    fx, fy, cx, cy = K
    u, v = np.meshgrid(np.arange(W), np.arange(H))
    dx, dy = (u - cx) / fx, (v - cy) / fy
    best = np.full((H, W), np.inf)
    kind = np.zeros((H, W), np.int8)  # 1 tread, 2 riser
    # floor (k=-1) and treads: plane y = cam_h - (k+1)*rise, z in [z_k, z_k+run]
    for k in range(-1, n_steps):
        y_pl = cam_h - (k + 1) * rise
        zk = z0 + k * run
        with np.errstate(divide="ignore", invalid="ignore"):
            t = np.where(dy > 1e-9, y_pl / dy, np.inf)
        ok = (t >= (zk if k >= 0 else 0.3)) & (t <= (zk + run if k >= 0 else z0)) & (t < best)
        best, kind = np.where(ok, t, best), np.where(ok, 1, kind)
        if k >= 0:  # riser at z = z_k, y between this tread and the one below
            y = zk * dy
            ok = (y >= y_pl) & (y <= y_pl + rise) & (zk < best)
            best, kind = np.where(ok, zk, best), np.where(ok, 2, kind)
    valid = np.isfinite(best)
    return np.where(valid, best, 0).astype(np.float32), kind == 1, kind == 2


def tens(a, dt=torch.float32):
    return torch.from_numpy(np.asarray(a))[None, None].to(dt)


@pytest.fixture(scope="module")
def stairs():
    z, tr, ri = staircase()
    valid = z > 0
    # exclude a band around tread|riser boundaries, as the loss does
    from scipy import ndimage

    band = ndimage.binary_dilation(
        ndimage.binary_dilation(tr, iterations=1) & ndimage.binary_dilation(ri, iterations=1), iterations=2
    ) | ndimage.binary_dilation(~valid, iterations=3)
    return z, tr, ri, valid, band


def run(pred, s, **kw):
    z, tr, ri, valid, band = s
    return planar_loss(tens(pred), tens(z), tens(valid, torch.bool), tens(tr, torch.bool),
                       tens(ri, torch.bool), tens(band, torch.bool), K, min_tread_px=200, **kw).item()


def test_normals_of_a_plane_match_analytic():
    n0 = np.array([0.3, -0.35, -0.85]); n0 /= np.linalg.norm(n0)
    u, v = np.meshgrid(np.arange(W), np.arange(H))
    ray = np.stack([(u - K[2]) / K[0], (v - K[3]) / K[1], np.ones_like(u, float)], -1)
    z = -2.0 / (ray @ n0)  # n0.P = -2  ->  P = z*ray
    assert (z > 0).all()
    n0f = n0  # n0.P = -2 < 0: already faces the camera
    n, ok = normals_from_depth(tens(z), K, tens(np.ones_like(z, bool), torch.bool))
    err = (n[0].permute(1, 2, 0).numpy() - n0f)[5:-5, 5:-5]
    assert np.abs(err).max() < 1e-4 and ok[0, 0, 5:-5, 5:-5].all()


def test_planar_zero_on_ideal_stairs_and_grows_with_wave_and_tilt(stairs):
    z, tr = stairs[0], stairs[1]
    assert run(z, stairs) < 1e-3
    for src in ("gt", "pred"):
        assert run(z, stairs, source=src) < 1e-3
    u, v = np.meshgrid(np.arange(W), np.arange(H))
    prev = 0.0
    for a in (0.02, 0.05, 0.12):  # wave on the treads
        zw = np.where(tr, z + a * np.sin(u / 3.0), z).astype(np.float32)
        l = run(zw, stairs, delta=0.01)
        assert l > 1.5 * prev + 1e-7, (a, l, prev)
        prev = l
    prev = 0.0
    for c in (0.001, 0.003, 0.008):  # tilt of the treads
        zt = np.where(tr, z + c * (u - K[2]), z).astype(np.float32)
        l = run(zt, stairs, delta=0.01)
        assert l > 1.5 * prev + 1e-8, (c, l, prev)
        prev = l


def test_no_tread_gives_zero(stairs):
    z, tr, ri, valid, band = stairs
    l = planar_loss(tens(z), tens(z), tens(valid, torch.bool), tens(np.zeros_like(tr), torch.bool),
                    tens(ri, torch.bool), tens(band, torch.bool), K)
    assert l.item() == 0.0


def test_gaussian_edge_map_one_at_ends_06_in_middle():
    seg = np.array([[20.0, 50.0, 120.0, 50.0]], np.float32)
    E, band = rasterize(seg, (100, 160), 6, "gaussian")
    assert E[50, 20] == pytest.approx(1.0, abs=1e-3) and E[50, 120] == pytest.approx(1.0, abs=1e-3)
    assert E[50, 70] == pytest.approx(np.exp(-0.5), abs=1e-3)  # ~0.607
    assert band[50 + 2, 70] and not band[50 + 5, 70]
    Eb, _ = rasterize(seg, (100, 160), 6, "binary")
    assert set(np.unique(Eb)) == {0.0, 1.0}


def test_ignore_pixels_get_weight_one_and_warmup():
    cfg = LossConfig(alpha=2.0, beta=1.0, planar_warmup_steps=100)
    loss = AprimeLoss(cfg, K=None)  # no K -> planar disabled with a warning
    assert loss.cfg.beta == 0.0
    loss = AprimeLoss(cfg, K=K)
    assert loss.beta(0) == 0 and loss.beta(50) == pytest.approx(0.5) and loss.beta(500) == 1.0


def test_smoke_total_loss_finite_and_decreases(stairs):
    z, tr, ri, valid, band = stairs
    dmin, dmax = float(z[valid].min()), float(z[valid].max())
    gt_n = tens((z - dmin) / (dmax - dmin))
    b = dict(gt_n=gt_n, valid=tens(valid, torch.bool), E=tens(band.astype(np.float32)),
             dmin=torch.tensor(dmin).view(1, 1, 1, 1), dmax=torch.tensor(dmax).view(1, 1, 1, 1),
             tread=tens(tr, torch.bool), riser=tens(ri, torch.bool), band=tens(band, torch.bool),
             ignore=torch.zeros(1, 1, H, W, dtype=torch.bool))
    pred = (gt_n + 0.1 * torch.randn_like(gt_n)).requires_grad_(True)
    opt = torch.optim.Adam([pred], lr=0.02)
    loss = AprimeLoss(LossConfig(alpha=2.0, beta=1.0, planar_warmup_steps=5, min_tread_px=200), K=K)
    first = None
    for step in range(40):
        out = loss(pred, b, step)
        assert torch.isfinite(out["total"])
        first = first if first is not None else out["total"].item()
        opt.zero_grad(); out["total"].backward(); opt.step()
    assert out["total"].item() < first


def test_step_geometry_recovers_rise_and_run_and_error_scales(stairs):
    from metrics.stair_geometry import step_geometry, step_geometry_error, up_vector

    z, tr, ri = staircase(z0=2.6)  # far enough that the image does not clip the near treads
    up = up_vector(z, tr, K)
    g = step_geometry(z, tr, K, up)
    assert np.allclose(g["rise"], 0.17, atol=0.012), g  # finite-difference normals bias ~4% at grazing view
    assert np.allclose(g["run"][1:], 0.28, atol=0.1), g  # k=0 is the floor; at grazing view one pixel row spans ~10 cm of tread, so run is coarse
    e0 = step_geometry_error(z, z, tr, K)
    assert np.nanmax(e0["rise_abs"]) < 1e-6
    e1 = step_geometry_error(np.where(tr, z * 1.03, z).astype(np.float32), z, tr, K)
    assert np.nanmax(e1["rise_abs"]) > 1e-3
