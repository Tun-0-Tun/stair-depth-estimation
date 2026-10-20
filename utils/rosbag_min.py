"""Minimal ROS1 bag (v2.0) reader: topics, types, first message per topic, CameraInfo decoding.

No ROS install needed.  Chunks: ``none`` and ``bz2`` (stdlib); ``lz4`` if the ``lz4`` package exists.
Only the first ``max_chunks`` chunks are decompressed (enough for the first message of each topic);
all other records are skipped by seeking, so a multi-GB bag is scanned in a second.
"""

from __future__ import annotations

import bz2
import struct
from pathlib import Path
from typing import Any

MAGIC = b"#ROSBAG V2.0\n"


def _fields(buf: bytes) -> dict[str, bytes]:
    out, i = {}, 0
    while i + 4 <= len(buf):
        (n,) = struct.unpack_from("<I", buf, i)
        i += 4
        k, _, v = buf[i : i + n].partition(b"=")
        out[k.decode(errors="replace")] = v
        i += n
    return out


def _records(buf: bytes):
    i = 0
    while i + 4 <= len(buf):
        (hl,) = struct.unpack_from("<I", buf, i)
        h = _fields(buf[i + 4 : i + 4 + hl])
        j = i + 4 + hl
        (dl,) = struct.unpack_from("<I", buf, j)
        yield h, buf[j + 4 : j + 4 + dl]
        i = j + 4 + dl


def is_ros_bag(path) -> bool:
    with open(path, "rb") as f:
        return f.read(len(MAGIC)) == MAGIC


def read_bag(path, max_chunks: int = 20, want: tuple[str, ...] = ("CameraInfo", "OBStreamProfileInfo", "OBDisparityParam")) -> dict[str, Any]:
    """``{"connections": {conn: {topic, type, definition}}, "first": {topic: bytes}, "counts": {topic: n_seen}, ...}``

    After ``max_chunks`` chunks the scan goes on (chunk by chunk, early exit) only while a topic whose type
    ends with one of ``want`` has no message yet -- latched messages such as stream profiles may sit anywhere.
    """
    conns: dict[int, dict] = {}
    first: dict[str, bytes] = {}
    counts: dict[str, int] = {}
    n_chunks = 0
    # pass 1: Orbbec bags define some connections ONLY in the index section after the last chunk,
    # so every message seen earlier would be dropped.  Collect all connection records first (seeks over chunks).
    with open(path, "rb") as f:
        if f.read(len(MAGIC)) != MAGIC:
            raise ValueError("not a ROS1 bag (v2.0)")
        while True:
            raw = f.read(4)
            if len(raw) < 4:
                break
            (hl,) = struct.unpack("<I", raw)
            h = _fields(f.read(hl))
            (dl,) = struct.unpack("<I", f.read(4))
            if h.get("op", b"\x00")[0] == 7:
                _add_conn(conns, h, f.read(dl))
            else:
                f.seek(dl, 1)
    # pass 2: chunks
    with open(path, "rb") as f:
        f.read(len(MAGIC))
        while True:
            raw = f.read(4)
            if len(raw) < 4:
                break
            (hl,) = struct.unpack("<I", raw)
            h = _fields(f.read(hl))
            (dl,) = struct.unpack("<I", f.read(4))
            op = h.get("op", b"\x00")[0]
            if op == 5:  # chunk
                n_chunks += 1
                if n_chunks > max_chunks and all(
                    c["topic"] in first for c in conns.values() if c["type"].endswith(want)
                ):
                    f.seek(dl, 1)
                    continue
                data, comp = f.read(dl), h.get("compression", b"none").decode()
                if comp == "bz2":
                    data = bz2.decompress(data)
                elif comp == "lz4":
                    try:
                        import lz4.frame
                    except ImportError:
                        raise ImportError("this bag has lz4 chunks: run with `uv run --with lz4 ...` (or pip install lz4)") from None
                    data = lz4.frame.decompress(data)
                elif comp != "none":
                    raise ValueError(f"unsupported chunk compression {comp!r}")
                for rh, rd in _records(data):
                    rop = rh.get("op", b"\x00")[0]
                    if rop == 7:
                        _add_conn(conns, rh, rd)
                    elif rop == 2:
                        c = conns.get(struct.unpack("<I", rh["conn"])[0])
                        if c:
                            counts[c["topic"]] = counts.get(c["topic"], 0) + 1
                            first.setdefault(c["topic"], rd)
            else:
                f.seek(dl, 1)
    return {"connections": conns, "first": first, "counts_in_scanned_chunks": counts, "n_chunks": n_chunks}


def _add_conn(conns, h, data):
    d = _fields(data)
    conns[struct.unpack("<I", h["conn"])[0]] = {
        "topic": h.get("topic", b"").decode(), "type": d.get("type", b"").decode(),
        "definition": d.get("message_definition", b"").decode(errors="replace"),
    }


def parse_camera_info(msg: bytes) -> dict[str, Any]:
    """sensor_msgs/CameraInfo -> width, height, model, D, K (fx,fy,cx,cy), P."""
    i = 4 + 8  # header.seq, header.stamp
    (n,) = struct.unpack_from("<I", msg, i); i += 4 + n  # frame_id
    h, w = struct.unpack_from("<II", msg, i); i += 8
    (n,) = struct.unpack_from("<I", msg, i); model = msg[i + 4 : i + 4 + n].decode(); i += 4 + n
    (nd,) = struct.unpack_from("<I", msg, i); i += 4
    D = list(struct.unpack_from(f"<{nd}d", msg, i)); i += 8 * nd
    Km = struct.unpack_from("<9d", msg, i); i += 72
    i += 72  # R
    P = struct.unpack_from("<12d", msg, i)
    return {"width": w, "height": h, "distortion_model": model, "D": D,
            "fx": Km[0], "fy": Km[4], "cx": Km[2], "cy": Km[5], "P": list(P)}


def _hdr(msg: bytes) -> int:
    (n,) = struct.unpack_from("<I", msg, 12)
    return 16 + n


def parse_ob_profile(msg: bytes) -> dict[str, Any]:
    """custom_msg/OBStreamProfileInfo (Orbbec SDK recorder): intrinsics + extrinsics of one stream.

    Layout: Header, u8 streamType (2 = colour, 3 = depth), u8 format, f32[9] rotation, f32[3] translation,
    u16 width, u16 height, u16 fps, f32[4] cameraIntrinsic (assumed fx, fy, cx, cy -- checked by the
    caller against width/height), f32[8] cameraDistortion, u8 distortionModel.
    """
    i = _hdr(msg)
    st, fmt = struct.unpack_from("<BB", msg, i); i += 2
    R = struct.unpack_from("<9f", msg, i); i += 36
    t = struct.unpack_from("<3f", msg, i); i += 12
    w, h, fps = struct.unpack_from("<HHH", msg, i); i += 6
    k = struct.unpack_from("<4f", msg, i); i += 16
    d = struct.unpack_from("<8f", msg, i); i += 32
    (dm,) = struct.unpack_from("<B", msg, i)
    return {"stream_type": st, "format": fmt, "width": w, "height": h, "fps": fps, "K4": list(k),
            "D": list(d), "distortion_model": dm, "rotation": list(R), "translation": list(t)}


def parse_ob_disparity(msg: bytes) -> dict[str, Any]:
    """custom_msg/OBDisparityParam: zpd, zpps, baseline, fx, ... of the stereo depth engine."""
    i = _hdr(msg)
    zpd, zpps = struct.unpack_from("<dd", msg, i); i += 16
    (baseline,) = struct.unpack_from("<f", msg, i); i += 4
    (fx,) = struct.unpack_from("<d", msg, i); i += 8
    bit, = struct.unpack_from("<B", msg, i); i += 1
    unit, mind = struct.unpack_from("<ff", msg, i)
    return {"zpd": zpd, "zpps": zpps, "baseline": baseline, "fx": fx, "bitSize": bit, "unit": unit, "minDisparity": mind}
