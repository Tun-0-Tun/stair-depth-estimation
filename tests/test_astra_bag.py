import struct

import numpy as np
import pytest

from utils.astra_bag import align_depth, decode_color, decode_depth
from utils.rosbag_min import MAGIC, iter_messages, parse_image


def _f(**kw):
    out = b""
    for k, v in kw.items():
        v = v if isinstance(v, bytes) else str(v).encode()
        out += struct.pack("<I", len(k) + 1 + len(v)) + k.encode() + b"=" + v
    return out


def _rec(h, d):
    return struct.pack("<I", len(h)) + h + struct.pack("<I", len(d)) + d


def image_msg(h, w, enc, data):
    s = lambda t: struct.pack("<I", len(t)) + t.encode()  # noqa: E731
    return struct.pack("<III", 0, 1, 2) + s("cam") + struct.pack("<II", h, w) + s(enc) + struct.pack("<BII", 0, len(data) // h, len(data)) + data


def test_iter_messages_and_parse_image(tmp_path):
    conn = lambda c, t: _rec(_f(op=b"\x07", conn=struct.pack("<I", c), topic=t), _f(type="sensor_msgs/Image", message_definition="x"))  # noqa: E731
    msg = lambda c, sec, d: _rec(_f(op=b"\x02", conn=struct.pack("<I", c), time=struct.pack("<II", sec, 5)), d)  # noqa: E731
    depth = (np.arange(6, dtype="<u2") * 100).reshape(2, 3)
    chunk = conn(0, "/a") + conn(1, "/b") + msg(0, 1, image_msg(2, 3, "16UC1", depth.tobytes())) + msg(1, 2, b"zz") + msg(0, 3, image_msg(2, 3, "16UC1", depth.tobytes()))
    p = tmp_path / "t.bag"
    p.write_bytes(MAGIC + _rec(_f(op=b"\x05", compression="none", size=len(chunk)), chunk))
    got = list(iter_messages(p, ["/a"]))
    assert [(t, s) for t, s, _ in got] == [("/a", 10**9 + 5), ("/a", 3 * 10**9 + 5)]
    img = parse_image(got[0][2])
    assert (img["height"], img["width"], img["encoding"]) == (2, 3, "16UC1")
    assert np.array_equal(decode_depth(img), depth)


def test_decode_color_yuyv_grey():
    img = {"height": 2, "width": 4, "encoding": "yuyv422", "data": bytes([128, 128] * 8), "big_endian": False, "step": 8}
    rgb = decode_color(img)
    assert rgb.shape == (2, 4, 3) and np.abs(rgb.astype(int) - 128).max() <= 2  # mid-grey stays mid-grey


def _profiles(R=None, t=(0, 0, 0)):
    k = [100.0, 100.0, 10.0, 8.0]
    pd = {"K4": k, "width": 20, "height": 16, "rotation": list(np.eye(3).ravel() if R is None else R), "translation": list(t), "D": [0] * 8}
    pc = {"K4": k, "width": 20, "height": 16, "rotation": [0] * 9, "translation": [0, 0, 0], "D": [0] * 8}
    return pd, pc


def test_align_identity_keeps_depth():
    pd, pc = _profiles()
    d = np.zeros((16, 20), np.uint16)
    d[4:12, 5:15] = 1000
    assert np.array_equal(align_depth(d, pd, pc), d)


def test_align_translation_shifts_pixels():
    pd, pc = _profiles(t=(10.0, 0, 0))  # 10 mm at z = 1000 mm, fx = 100 -> exactly one pixel
    d = np.zeros((16, 20), np.uint16)
    d[4:12, 5:15] = 1000
    al = align_depth(d, pd, pc, "d2c")
    assert (al[4:12, 6:16] == 1000).all() and (al[4:12, 5] == 0).all()
    back = align_depth(d, pd, pc, "c2d")
    assert (back[4:12, 4:14] == 1000).all()


def test_align_nearest_surface_wins():
    pd, pc = _profiles()
    pd = {**pd, "K4": [400.0, 400.0, 10.0, 8.0]}  # depth camera with a long lens: two depth pixels land on one colour pixel
    d = np.zeros((16, 20), np.uint16)
    d[8, 10], d[8, 11] = 3000, 1500  # u=11 at z=1500 -> colour column 10.25 -> 10, same as u=10
    assert align_depth(d, pd, pc)[8, 10] == 1500
