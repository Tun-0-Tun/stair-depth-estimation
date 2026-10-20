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


def test_ob_profile_and_late_latched_message(tmp_path):
    from utils.rosbag_min import parse_ob_profile

    hdr = struct.pack("<III", 0, 1, 2) + struct.pack("<I", 3) + b"cam"
    prof = (hdr + struct.pack("<BB", 2, 1) + struct.pack("<9f", *[1, 0, 0, 0, 1, 0, 0, 0, 1]) + struct.pack("<3f", 0.1, 0, 0)
            + struct.pack("<HHH", 1920, 1080, 30) + struct.pack("<4f", 1050.5, 1049.0, 960.2, 541.1)
            + struct.pack("<8f", *[0.1] * 8) + struct.pack("<B", 3))
    conn = lambda c, t, ty: _rec(_f(op=b"\x07", conn=struct.pack("<I", c), topic=t), _f(type=ty, message_definition="x"))  # noqa: E731
    msg = lambda c, d: _rec(_f(op=b"\x02", conn=struct.pack("<I", c), time=b"\0" * 8), d)  # noqa: E731
    chunks = [conn(0, "/cam/profile", "custom_msg/OBStreamProfileInfo") + conn(1, "/cam/img", "sensor_msgs/Image") + msg(1, b"x" * 8)]
    chunks += [msg(1, b"y" * 8)] * 3 + [msg(0, prof)]  # the profile only appears in the 5th chunk
    bag = MAGIC
    for c in chunks:
        bag += _rec(_f(op=b"\x05", compression="none", size=len(c)), c)
    p = tmp_path / "ob.bag"
    p.write_bytes(bag)
    r = read_bag(p, max_chunks=2)
    o = parse_ob_profile(r["first"]["/cam/profile"])
    assert o["stream_type"] == 2 and (o["width"], o["height"]) == (1920, 1080)
    assert o["K4"][0] == pytest.approx(1050.5) and o["K4"][3] == pytest.approx(541.1) and o["translation"][0] == pytest.approx(0.1)


def test_connection_defined_only_in_index_section(tmp_path):
    """Orbbec bags: the connection record of a topic comes after the chunks that hold its messages."""
    msg = _rec(_f(op=b"\x02", conn=struct.pack("<I", 0), time=b"\0" * 8), b"payload")
    conn = _rec(_f(op=b"\x07", conn=struct.pack("<I", 0), topic="/late"), _f(type="custom_msg/OBStreamProfileInfo", message_definition="x"))
    chunk = msg
    bag = MAGIC + _rec(_f(op=b"\x05", compression="none", size=len(chunk)), chunk) + conn
    p = tmp_path / "late.bag"
    p.write_bytes(bag)
    r = read_bag(p)
    assert r["first"]["/late"] == b"payload"
