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


def test_frame_rays_invert_brown_conrady_and_match_pinhole():
    cv2 = pytest.importorskip("cv2")
    from utils.camera import convert_K, frame_rays

    Kf, D, size = (1253.8, 1254.4, 963.2, 531.8), (0.1125, -0.3288, 0.2582, 0, 0, 0, -0.0005, -0.00012), (1920, 1080)
    crop, tgt = (12, 1068, 256, 1632), (528, 688)
    rays = frame_rays(tgt, Kf, D, crop, tgt)
    # forward-project a known ray with cv2 (distorted), map to the cropped/resized frame, compare
    ray = np.array([[0.3, -0.2, 1.0]])
    Km = np.array([[Kf[0], 0, Kf[2]], [0, Kf[1], Kf[3]], [0, 0, 1.0]])
    px = cv2.projectPoints(ray, np.zeros(3), np.zeros(3), Km, np.array([D[0], D[1], D[6], D[7], D[2], D[3], D[4], D[5]]))[0][0, 0]
    u, v = (px[0] - crop[2] + 0.5) * 0.5 - 0.5, (px[1] - crop[0] + 0.5) * 0.5 - 0.5
    got = rays[:, int(round(v)), int(round(u))]
    assert np.abs(got - ray[0, :2]).max() < 6e-3  # one frame pixel ~ 1/627 rad
    # without distortion == pinhole of the converted K
    r0 = frame_rays(tgt, Kf, None, crop, tgt)
    fx, fy, cx, cy = convert_K(Kf, size, crop, tgt)
    assert np.allclose(r0[0, 100, 200], (200 - cx) / fx, atol=1e-5) and np.allclose(r0[1, 100, 200], (100 - cy) / fy, atol=1e-5)
    # ray maps work in the loss/geometry code and equal the pinhole path
    zz = tens(np.full((tgt[0], tgt[1]), 2.0, np.float32))
    n1, _ = normals_from_depth(zz, (fx, fy, cx, cy), tens(np.ones(tgt, bool), torch.bool))
    n2, _ = normals_from_depth(zz, r0, tens(np.ones(tgt, bool), torch.bool))
    assert (n1 - n2).abs().max() < 1e-4


def test_clean_masks_drops_noise_and_non_stair_frames():
    from data.stairnet.edges import clean_masks

    cls = np.zeros((240, 320), np.uint8)
    cls[60:140, 20:300] = 1  # a big tread
    cls[140:170, 20:300] = 2  # a big riser
    cls[10:20, 10:20] = 2  # speckle
    cls[200:206, 100:140] = 1  # thin sliver
    out = clean_masks(cls)
    assert (out[10:20, 10:20] == 0).all() and (out[200:206, 100:140] == 0).all()
    assert (out[60:140, 20:300] == 1).all() and (out[140:170, 20:300] == 2).all()
    sparse = np.zeros_like(cls)
    sparse[:30, :60] = 1  # < 5 % of the frame: not a stair scene
    assert clean_masks(sparse).sum() == 0
    assert clean_masks(cls, False) is cls


def test_planar_gating_ignores_inconsistent_labels(stairs):
    z, tr, ri, valid, band = stairs
    rng = np.random.default_rng(0)
    wrong = tr.copy()
    wrong[ri] = True  # label every riser pixel as tread as well: geometrically inconsistent
    base = run(z, stairs)
    noisy = planar_loss(tens(z), tens(z), tens(valid, torch.bool), tens(wrong, torch.bool), tens(ri, torch.bool),
                        tens(band, torch.bool), K, min_tread_px=200).item()
    assert noisy < 1e-3, noisy  # gating removed the bogus tread pixels instead of penalising a perfect staircase
    junk = rng.random(z.shape) > 0.5  # a frame whose "treads" are random pixels: dropped as a whole
    l = planar_loss(tens(z), tens(z), tens(valid, torch.bool), tens(junk & (tr | ri), torch.bool),
                    tens(np.zeros_like(ri), torch.bool), tens(band, torch.bool), K, min_tread_px=200).item()
    assert l < 1e-3 and base < 1e-3


def test_tread_levels_tell_a_staircase_from_flat_ground_with_false_blobs():
    from metrics.stair_geometry import tread_levels, up_vector

    z, tr, ri = staircase(z0=2.6)
    up = up_vector(z, tr, K)
    assert tread_levels(z, tr, K, up) >= 3  # floor + several steps
    # flat floor seen from above, with blobs labelled "tread"
    v = np.mgrid[:H, :W][0].astype(np.float32)
    floor = (0.9 * K[1] / np.maximum(v - K[3] + 40, 1.0)).astype(np.float32)  # plane y = 0.9 m below the camera
    blobs = np.zeros((H, W), bool)
    blobs[150:200, 60:120] = blobs[170:220, 200:290] = True
    upf = np.array([0.0, -1.0, 0.0])
    assert tread_levels(floor, blobs, K, upf) == 1
