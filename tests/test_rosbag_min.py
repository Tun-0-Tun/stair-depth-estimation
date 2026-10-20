import bz2
import struct

import pytest

from utils.rosbag_min import MAGIC, parse_camera_info, read_bag


def _f(**kw):
    out = b""
    for k, v in kw.items():
        v = v if isinstance(v, bytes) else str(v).encode()
        out += struct.pack("<I", len(k) + 1 + len(v)) + k.encode() + b"=" + v
    return out


def _rec(h, d):
    return struct.pack("<I", len(h)) + h + struct.pack("<I", len(d)) + d


def camera_info(w, h, fx, fy, cx, cy):
    s = lambda t: struct.pack("<I", len(t)) + t.encode()  # noqa: E731
    return (struct.pack("<III", 0, 1, 2) + s("cam") + struct.pack("<II", h, w) + s("plumb_bob")
            + struct.pack("<I", 5) + struct.pack("<5d", *[0.1, 0, 0, 0, 0])
            + struct.pack("<9d", fx, 0, cx, 0, fy, cy, 0, 0, 1) + struct.pack("<9d", *[1, 0, 0, 0, 1, 0, 0, 0, 1])
            + struct.pack("<12d", *[0] * 12) + b"\0" * 20)


@pytest.mark.parametrize("comp", ["none", "bz2"])
def test_read_bag_and_camera_info(tmp_path, comp):
    conn = _rec(_f(op=b"\x07", conn=struct.pack("<I", 0), topic="/cam/color/camera_info"),
                _f(type="sensor_msgs/CameraInfo", message_definition="x"))
    msg = _rec(_f(op=b"\x02", conn=struct.pack("<I", 0), time=b"\0" * 8), camera_info(1920, 1080, 1050.5, 1049.0, 960.2, 541.1))
    chunk = conn + msg
    if comp == "bz2":
        chunk = bz2.compress(chunk)
    bag = MAGIC + _rec(_f(op=b"\x05", compression=comp, size=len(chunk)), chunk)
    p = tmp_path / "t.bag"
    p.write_bytes(bag)
    r = read_bag(p)
    assert r["connections"][0]["type"] == "sensor_msgs/CameraInfo"
    ci = parse_camera_info(r["first"]["/cam/color/camera_info"])
    assert (ci["width"], ci["height"]) == (1920, 1080)
    assert ci["fx"] == 1050.5 and ci["fy"] == 1049.0 and ci["cx"] == 960.2 and ci["cy"] == 541.1 and len(ci["D"]) == 5
