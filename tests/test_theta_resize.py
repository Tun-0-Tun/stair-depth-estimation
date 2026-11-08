"""Surface-aware ("guided") resizing of normal and theta maps."""

import numpy as np
import pytest

from data.norm import CrossProductNormals, get_theta_map, resize_normals, resize_theta


def _constant(h, w, deg):
    t = np.radians(deg)
    n = np.zeros((h, w, 3), np.float32)
    n[..., 1], n[..., 2] = np.sin(t), np.cos(t)
    return n


def _stairs(h, w):
    """Top half tread (faces up), bottom half riser (faces the camera)."""
    n = np.zeros((h, w, 3), np.float32)
    n[: h // 2, :, 1] = -1.0
    n[h // 2 :, :, 2] = -1.0
    return n


def _tilted_normals(deg, h=100, w=100, fy=570.0):
    s = np.tan(np.radians(deg))
    v = np.arange(h, dtype=np.float64)[:, None] - (h - 1) / 2
    gt = np.broadcast_to(2.0 / (1.0 - s * v / fy), (h, w)).astype(np.float32)
    return CrossProductNormals(fx=fy)(gt)["normals"]


def _deg(theta):
    return np.degrees(theta)


@pytest.mark.parametrize("out_shape", [(64, 64), (25, 25), (48, 48)])
def test_constant_plane_is_exact(out_shape):
    n = _constant(100, 100, 30)
    theta = _deg(get_theta_map(n, out_shape))
    assert theta.shape == out_shape
    assert np.abs(theta - 30).max() < 1e-4


def test_smooth_plane_matches_bilinear():
    n = _tilted_normals(20)
    g = resize_normals(n, (200, 200), mode="guided")
    b = resize_normals(n, (200, 200), mode="bilinear")
    valid = (np.linalg.norm(g, axis=-1) > 0) & (np.linalg.norm(b, axis=-1) > 0)
    assert valid.mean() > 0.9
    assert np.abs(g - b)[valid].max() < 1e-5


@pytest.mark.parametrize("src, out", [((4, 4), (8, 8)), ((16, 16), (4, 4)), ((9, 9), (6, 6))])
def test_no_phantom_normals_at_the_nosing(src, out):
    n = _stairs(*src)
    guided = _deg(get_theta_map(n, out, resize_mode="guided"))
    assert np.isfinite(guided).all()
    assert ((guided < 1e-3) | (guided > 90 - 1e-3)).all()
    if out[0] > src[0]:  # upsampling: bilinear does create the phantom
        bilinear = _deg(get_theta_map(n, out, resize_mode="bilinear"))
        assert ((bilinear > 5) & (bilinear < 85)).any()


def test_holes_do_not_bleed():
    n = _constant(32, 32, 30)
    n[10:14, 10:14] = 0.0
    g = _deg(get_theta_map(n, (64, 64), resize_mode="guided"))
    b = _deg(get_theta_map(n, (64, 64), resize_mode="bilinear"))
    assert np.isnan(g).any()
    assert np.abs(g[np.isfinite(g)] - 30).max() < 1e-4
    # hole pixels are exactly those whose nearest source pixel is in the hole
    assert np.isnan(g[20:28, 20:28]).all()
    assert np.isfinite(g[:19]).all() and np.isfinite(g[29:]).all()
    # bilinear shrinks the hole: it invents angles where the nearest source pixel
    # is missing (a zero vector only shortens the blend, the direction survives)
    assert np.isnan(b).sum() < np.isnan(g).sum()
    assert np.isfinite(b[20:28, 20:28]).any()


@pytest.mark.parametrize("angle_reference", ["optical_axis", "viewing_ray"])
@pytest.mark.parametrize("out_shape", [(200, 200), (50, 50)])
def test_resize_orders_agree(angle_reference, out_shape):
    for n in (_constant(100, 100, 30), _tilted_normals(20)):
        a = get_theta_map(n, out_shape, resize_order="normals", angle_reference=angle_reference)
        b = get_theta_map(n, out_shape, resize_order="theta", angle_reference=angle_reference)
        both = np.isfinite(a) & np.isfinite(b)
        assert both.mean() > 0.8
        # viewing_ray: theta varies across the image even on a plane, so resizing
        # the angle map interpolates it (~0.05 deg); optical_axis is constant.
        tol = 2e-3 if angle_reference == "viewing_ray" else 1e-4
        assert np.abs(a - b)[both].max() < tol


def test_theta_order_keeps_the_nosing_sharp():
    n = _stairs(8, 8)
    t = _deg(get_theta_map(n, (32, 32), resize_order="theta"))
    assert np.isfinite(t).all()
    assert ((t < 1e-3) | (t > 90 - 1e-3)).all()


def test_resize_theta_nan_matches_resize_normals_zero():
    n = _constant(16, 16, 40)
    n[3:6, 3:6] = 0.0
    theta = get_theta_map(n, (16, 16))
    assert np.isnan(theta[3:6, 3:6]).all()
    t = resize_theta(theta, (32, 32))
    v = resize_normals(n, (32, 32))
    assert (np.isnan(t) == (np.linalg.norm(v, axis=-1) == 0)).all()
    assert np.abs(_deg(t[np.isfinite(t)]) - 40).max() < 1e-4


def test_bad_arguments():
    n = _constant(8, 8, 10)
    with pytest.raises(ValueError):
        get_theta_map(n, (16, 16), resize_mode="bicubic")
    with pytest.raises(ValueError):
        get_theta_map(n, (16, 16), resize_order="angles")
    with pytest.raises(ValueError):
        resize_normals(n, (16, 32))
