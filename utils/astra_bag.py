"""Orbbec Astra 2 raw ``.bag`` (colour YUYV 1920x1080 + depth Y16 1600x1200) -> frames in the layout of ``ours/sample``.

The bags carry no aligned depth, so the depth is re-projected into the colour camera here: unproject with the depth
intrinsics, move with the depth->colour extrinsics stored in the bag (``OBStreamProfileInfo`` of the depth stream),
project with the colour intrinsics (optionally with its Brown-Conrady distortion) and z-buffer.  Holes stay 0.
The convention details (extrinsics direction, distortion, which z) are options because ``scripts/bag_to_shot.py validate``
decides them against the shots we already have; nothing here is trusted until that check passes.
"""

from __future__ import annotations

import numpy as np

from utils import rosbag_min as rb

COLOR_TOPIC, DEPTH_TOPIC = "/cam/sensor_2/frameType_2", "/cam/sensor_3/frameType_3"
PROFILE_TOPICS = {"color": "/cam/streamProfileType_2", "depth": "/cam/streamProfileType_3"}


def load_bag(path) -> tuple[dict, dict]:
    """``(profiles, frames)``: profiles ``{"color"|"depth": OB profile dict}``, frames ``{"color"|"depth": [(stamp_ns, image)]}``."""
    info = rb.read_bag(path)
    prof = {k: rb.parse_ob_profile(info["first"][t]) for k, t in PROFILE_TOPICS.items()}
    frames: dict[str, list] = {"color": [], "depth": []}
    for topic, t, data in rb.iter_messages(path, (COLOR_TOPIC, DEPTH_TOPIC)):
        frames["color" if topic == COLOR_TOPIC else "depth"].append((t, rb.parse_image(data)))
    for v in frames.values():
        v.sort(key=lambda x: x[0])
    return prof, frames


def decode_color(img: dict) -> np.ndarray:
    """uint8 (H, W, 3) RGB."""
    import cv2

    h, w, enc = img["height"], img["width"], img["encoding"].lower()
    buf = np.frombuffer(img["data"], np.uint8)
    if enc in ("rgb8",):
        return buf.reshape(h, w, 3).copy()
    if enc in ("bgr8",):
        return buf.reshape(h, w, 3)[..., ::-1].copy()
    if "yuv" in enc or "yuy" in enc or enc == "uyvy":
        return cv2.cvtColor(buf.reshape(h, w, 2), cv2.COLOR_YUV2RGB_YUY2)
    raise ValueError(f"unsupported colour encoding {img['encoding']!r} (report it, one branch to add)")


def decode_depth(img: dict) -> np.ndarray:
    """uint16 (H, W), values as stored (millimetres for this camera: ``unit`` = 1.0)."""
    dt = ">u2" if img["big_endian"] else "<u2"
    return np.frombuffer(img["data"], dt).reshape(img["height"], img["width"]).astype(np.uint16)


def nearest(frames: list, stamp: int) -> int:
    """Index of the frame in ``frames`` closest in time to ``stamp``."""
    return int(np.argmin([abs(t - stamp) for t, _ in frames]))


def _distort(x, y, d):
    """Brown-Conrady, Orbbec coefficient order [k1, k2, k3, k4, k5, k6, p1, p2] (rational radial)."""
    k1, k2, k3, k4, k5, k6, p1, p2 = d
    r2 = x * x + y * y
    rad = (1 + r2 * (k1 + r2 * (k2 + r2 * k3))) / (1 + r2 * (k4 + r2 * (k5 + r2 * k6)))
    return x * rad + 2 * p1 * x * y + p2 * (r2 + 2 * x * x), y * rad + p1 * (r2 + 2 * y * y) + 2 * p2 * x * y


def align_depth(depth, pd, pc, direction="d2c", use_dist=False, z_mode="color") -> np.ndarray:
    """Depth (uint16 mm, depth-camera frame) -> uint16 mm in the colour frame (0 = no data).

    ``pd`` / ``pc``: OB profile dicts of the depth / colour stream.  ``direction``: "d2c" -> P_c = R P_d + t with the
    depth profile's extrinsics, "c2d" -> the inverse.  ``use_dist``: apply the colour distortion when projecting.
    ``z_mode``: "color" stores Z in the colour frame, "depth" keeps the depth camera's own Z.
    """
    fx, fy, cx, cy = pd["K4"]
    v, u = np.nonzero(depth)
    z = depth[v, u].astype(np.float64)
    P = np.stack([(u - cx) / fx * z, (v - cy) / fy * z, z], 1)
    R, t = np.asarray(pd["rotation"], np.float64).reshape(3, 3), np.asarray(pd["translation"], np.float64)
    Pc = P @ R.T + t if direction == "d2c" else (P - t) @ R
    zc = Pc[:, 2]
    front = zc > 1e-6
    x, y = Pc[:, 0] / np.where(front, zc, 1), Pc[:, 1] / np.where(front, zc, 1)
    if use_dist:
        x, y = _distort(x, y, pc["D"])
    fxc, fyc, cxc, cyc = pc["K4"]
    W, H = pc["width"], pc["height"]
    uu, vv = np.rint(fxc * x + cxc).astype(np.int64), np.rint(fyc * y + cyc).astype(np.int64)
    ok = front & (uu >= 0) & (uu < W) & (vv >= 0) & (vv < H)
    val = (zc if z_mode == "color" else z)[ok]
    order = np.argsort(-val, kind="stable")  # far first: the nearest surface is written last and wins
    out = np.zeros(H * W, np.uint16)
    out[(vv[ok] * W + uu[ok])[order]] = np.clip(np.rint(val[order]), 0, 65535).astype(np.uint16)
    return out.reshape(H, W)
