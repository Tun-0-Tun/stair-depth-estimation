#!/usr/bin/env python3
"""
Astra 2 processing steps 11–13:
  11. Extract RGB, original RAW uint16 depth, and timestamps from an Orbbec .bag
  12. Generate a separate depth-to-color (D2C) aligned depth image
  13. Generate validity masks
  14. Save colorized depth visualization PNGs

Default behavior: scan each short .bag and save only its middle valid RGB+depth
frameset. Use --frame-selection all to save every valid frameset, or first to
save the first one.

Requirements:
    pip install numpy opencv-python
    Install a compatible Orbbec pyorbbecsdk2 build.

Example:
    python astra2_steps11_13.py --bag sample_000001.bag --out processed/sample_000001

Notes:
- This reads a recorded .bag; it does not require Orbbec Viewer.
- Raw depth is saved unchanged as uint16 .npy and 16-bit PNG.
- Alignment is a derived product and is saved separately.
- The SDK's depth scale is stored per frame. Do not assume raw uint16 values
  are millimetres without checking this scale.
- Playback uses the SDK v2 PlaybackDevice + Pipeline(playback_device) API.
  Older SDK bindings may use a different playback interface.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Optional

import cv2
import numpy as np

from pyorbbecsdk import (
    AlignFilter,
    Config,
    OBStreamType,
    Pipeline,
    PlaybackDevice,
)


# --------------------------- SDK-safe metadata ---------------------------

def json_safe(value: Any) -> Any:
    """Convert SDK/NumPy objects into JSON-compatible values."""
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    # pybind enums and SDK metadata objects are not always Python Enum types.
    try:
        return str(value)
    except Exception:
        return repr(value)


def get_frame_timestamp(frame: Any) -> dict[str, Any]:
    """Return available frame timestamp fields without assuming their units."""
    result: dict[str, Any] = {}
    for method_name in ("get_timestamp", "get_system_timestamp", "get_number"):
        method = getattr(frame, method_name, None)
        if callable(method):
            try:
                result[method_name.removeprefix("get_")] = json_safe(method())
            except Exception as exc:
                result[method_name.removeprefix("get_") + "_error"] = str(exc)
    return result


# ------------------------------- Step 11 --------------------------------

def depth_frame_to_raw_u16(depth_frame: Any) -> np.ndarray:
    """Copy a depth frame into its original 2-D uint16 array."""
    width = int(depth_frame.get_width())
    height = int(depth_frame.get_height())
    raw = np.frombuffer(depth_frame.get_data(), dtype=np.uint16)
    expected = width * height
    if raw.size != expected:
        raise ValueError(
            f"Depth buffer has {raw.size} uint16 values; expected {expected} "
            f"for {width}x{height}. Check depth format/stride."
        )
    # Copy ensures the array remains valid after the SDK reuses its frame buffer.
    return raw.reshape(height, width).copy()


def color_frame_to_bgr(color_frame: Any) -> np.ndarray:
    """
    Decode common Orbbec color formats into an 8-bit BGR image for PNG output.
    Pixel values are not enhanced; conversion only decodes the sensor format.
    """
    width = int(color_frame.get_width())
    height = int(color_frame.get_height())
    fmt = str(color_frame.get_format()).upper()
    data = bytes(color_frame.get_data())

    if "YUYV" in fmt or "YUY2" in fmt:
        expected = width * height * 2
        if len(data) < expected:
            raise ValueError(f"YUYV buffer too small: {len(data)} < {expected}")
        yuyv = np.frombuffer(data[:expected], dtype=np.uint8).reshape(height, width, 2)
        return cv2.cvtColor(yuyv, cv2.COLOR_YUV2BGR_YUY2)

    if "RGB" in fmt:
        arr = np.frombuffer(data, dtype=np.uint8)
        expected = width * height * 3
        if arr.size < expected:
            raise ValueError(f"RGB buffer too small: {arr.size} < {expected}")
        rgb = arr[:expected].reshape(height, width, 3)
        return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

    if "BGR" in fmt:
        arr = np.frombuffer(data, dtype=np.uint8)
        expected = width * height * 3
        if arr.size < expected:
            raise ValueError(f"BGR buffer too small: {arr.size} < {expected}")
        return arr[:expected].reshape(height, width, 3).copy()

    # Some SDK profiles expose MJPEG/MJPG frames as compressed JPEG bytes.
    if "MJPG" in fmt or "MJPEG" in fmt:
        encoded = np.frombuffer(data, dtype=np.uint8)
        image = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError("OpenCV could not decode the MJPEG color frame.")
        if image.shape[:2] != (height, width):
            raise ValueError(
                f"Decoded color dimensions {image.shape[1]}x{image.shape[0]} "
                f"do not match SDK metadata {width}x{height}."
            )
        return image

    raise ValueError(
        f"Unsupported color format {fmt!r}. Add a decoder for this SDK format."
    )


def save_raw_depth(raw_u16: np.ndarray, out_dir: Path, stem: str) -> None:
    """Save unchanged depth values in lossless NumPy and 16-bit PNG formats."""
    if raw_u16.dtype != np.uint16 or raw_u16.ndim != 2:
        raise TypeError("Expected a 2-D uint16 raw depth array.")
    # np.save(out_dir / f"{stem}.npy", raw_u16, allow_pickle=False)
    if not cv2.imwrite(str(out_dir / f"{stem}.png"), raw_u16):
        raise IOError(f"Could not write {out_dir / f'{stem}.png'}")


def save_colorized_depth(
    depth_u16: np.ndarray,
    depth_scale: Optional[float],
    out_dir: Path,
    stem: str,
    *,
    min_depth_mm: float = 600.0,
    max_depth_mm: float = 8000.0,
) -> None:
    """Save a viewable color map from SDK depth values; raw data is untouched.

    Uses the per-frame SDK depth scale and a fixed 200–5000 mm display range,
    so colors are comparable between bags. Invalid/out-of-range pixels are black.
    """
    if depth_scale is None or not np.isfinite(depth_scale) or depth_scale <= 0:
        raise ValueError("Cannot colorize depth: SDK depth scale is missing or invalid")
    depth_mm = depth_u16.astype(np.float32) * float(depth_scale)
    valid = (depth_u16 > 0) & (depth_mm >= min_depth_mm) & (depth_mm <= max_depth_mm)
    gray = np.zeros(depth_u16.shape, dtype=np.uint8)
    scaled = (depth_mm - min_depth_mm) / (max_depth_mm - min_depth_mm)
    gray[valid] = np.clip(scaled[valid] * 255.0, 0, 255).astype(np.uint8)
    colored = cv2.applyColorMap(gray, cv2.COLORMAP_JET)
    colored[~valid] = 0
    path = out_dir / f"{stem}_depth_colorized.png"
    if not cv2.imwrite(str(path), colored):
        raise IOError(f"Could not write colorized depth image: {path}")


def extract_rgb_depth_timestamps(
    color_frame: Any,
    depth_frame: Any,
    name: str,
    out_dir: Path,
    index: int,
) -> dict[str, Any]:
    """
    Step 11 for one frameset. Saves RGB and raw depth before any alignment.
    Returns metadata to add to timestamps.json.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{name}"

    bgr = color_frame_to_bgr(color_frame)
    if not cv2.imwrite(str(out_dir / f"{stem}_rgb.png"), bgr):
        raise IOError(f"Could not write {stem}_rgb.png")

    raw_depth = depth_frame_to_raw_u16(depth_frame)
    save_raw_depth(raw_depth, out_dir, f"{stem}_depth_raw")

    try:
        depth_scale = float(depth_frame.get_depth_scale())
    except Exception:
        depth_scale = None
    if depth_scale is not None:
        save_colorized_depth(raw_depth, depth_scale, out_dir, stem)

    return {
        "frame_index": index,
        "color": {
            "width": int(color_frame.get_width()),
            "height": int(color_frame.get_height()),
            "format": str(color_frame.get_format()),
            "timestamps": get_frame_timestamp(color_frame),
        },
        "depth": {
            "width": int(depth_frame.get_width()),
            "height": int(depth_frame.get_height()),
            "format": str(depth_frame.get_format()),
            "timestamps": get_frame_timestamp(depth_frame),
            "depth_scale": depth_scale,
            "raw_array_dtype": str(raw_depth.dtype),
            "raw_array_file": f"{stem}_depth_raw.npy",
            "raw_png_file": f"{stem}_depth_raw.png",
        },
        "rgb_file": f"{stem}_rgb.png",
        "depth_visualization_file": f"{stem}_depth_colorized.png" if depth_scale is not None else None,
        "depth_visualization_range_mm": [600.0, 8000.0],
        "note": "Raw depth PNG is 16-bit lossless. Colorized PNG is for visualization only.",
    }


# ------------------------------- Step 12 --------------------------------

def align_depth_to_color(
    frameset: Any,
    align_filter: Any,
) -> Optional[np.ndarray]:
    """
    Apply Orbbec SDK software D2C alignment to a frameset and return aligned
    uint16 depth. This does not replace the saved raw depth array.

    Alignment output dimensions/scale are recorded separately because SDK
    versions and selected profiles can affect the output representation.
    """
    aligned_frameset = align_filter.process(frameset)
    if aligned_frameset is None:
        return None
    # Some SDK versions return a FrameSet directly; others expose as_frame_set().
    as_frame_set = getattr(aligned_frameset, "as_frame_set", None)
    if callable(as_frame_set):
        try:
            converted = as_frame_set()
            if converted is not None:
                aligned_frameset = converted
        except Exception:
            pass

    aligned_depth_frame = aligned_frameset.get_depth_frame()
    if aligned_depth_frame is None:
        return None
    return depth_frame_to_raw_u16(aligned_depth_frame)


def save_aligned_depth(
    aligned_u16: np.ndarray,
    out_dir: Path,
    stem: str,
) -> None:
    """Save aligned depth separately; never overwrite raw depth."""
    save_raw_depth(aligned_u16, out_dir, stem)


# ------------------------------- Step 13 --------------------------------

def make_validity_masks(
    depth_u16: np.ndarray,
    *,
    min_raw_value: int = 1,
    max_raw_value: int = 65535,
    discontinuity_threshold_mm: Optional[float] = 100.0,
    depth_scale: Optional[float] = None,
) -> dict[str, np.ndarray]:
    """
    Make masks from a uint16 depth array.

    valid: raw values inside the configured range (zero is normally invalid).
    discontinuity: conservative mask excluding pixels near strong local depth
    edges; only computed if depth_scale is known. The scale unit follows the
    SDK convention used by this camera/profile, so verify it before relying on
    the threshold. For the SDK versions where depth_scale is mm per unit,
    threshold_mm is directly comparable after multiplication by depth_scale.
    """
    if depth_u16.dtype != np.uint16 or depth_u16.ndim != 2:
        raise TypeError("Expected a 2-D uint16 depth array.")

    valid = (
        (depth_u16 >= int(min_raw_value))
        & (depth_u16 <= int(max_raw_value))
    )
    masks = {"valid": (valid.astype(np.uint8) * 255)}

    if discontinuity_threshold_mm is not None and depth_scale is not None:
        depth_mm = depth_u16.astype(np.float32) * float(depth_scale)
        # Ignore invalid values when calculating local gradients.
        safe = depth_mm.copy()
        safe[~valid] = np.nan
        # Compare horizontal and vertical neighbors. Mark both sides of a
        # discontinuity and dilate by one pixel to make the mask conservative.
        dx = np.zeros_like(depth_mm, dtype=np.float32)
        dy = np.zeros_like(depth_mm, dtype=np.float32)
        pair_x = valid[:, 1:] & valid[:, :-1]
        pair_y = valid[1:, :] & valid[:-1, :]
        dx[:, 1:][pair_x] = np.abs(depth_mm[:, 1:][pair_x] - depth_mm[:, :-1][pair_x])
        dy[1:, :][pair_y] = np.abs(depth_mm[1:, :][pair_y] - depth_mm[:-1, :][pair_y])
        edge = (dx > float(discontinuity_threshold_mm)) | (
            dy > float(discontinuity_threshold_mm)
        )
        edge_u8 = edge.astype(np.uint8)
        edge_u8 = cv2.dilate(edge_u8, np.ones((3, 3), np.uint8), iterations=1)
        # masks["depth_discontinuity"] = edge_u8 * 255
        masks["valid"] = (
            valid & (edge_u8 == 0)
        ).astype(np.uint8) * 255

    return masks


def save_masks(
    depth_u16: np.ndarray,
    out_dir: Path,
    stem: str,
    *,
    depth_scale: Optional[float] = None,
    discontinuity_threshold_mm: Optional[float] = 100.0,
) -> dict[str, Any]:
    masks = make_validity_masks(
        depth_u16,
        depth_scale=depth_scale,
        discontinuity_threshold_mm=discontinuity_threshold_mm,
    )
    counts: dict[str, Any] = {}
    for name, mask in masks.items():
        path = out_dir / f"{stem}_{name}.png"
        if not cv2.imwrite(str(path), mask):
            raise IOError(f"Could not write mask: {path}")
        counts[name] = {
            "file": path.name,
            "valid_pixels": int(np.count_nonzero(mask)),
            "total_pixels": int(mask.size),
            "percent_nonzero": float(100.0 * np.count_nonzero(mask) / mask.size),
        }
    return counts


# ------------------------------ Bag runner -----------------------------

def process_bag(
    bag_path: Path,
    out_dir: Path,
    name: str,
    *,
    max_frames: Optional[int] = None,
    frame_selection: str = "middle",
    discontinuity_threshold_mm: Optional[float] = 100.0,
) -> None:
    """Scan a .bag and process its first, middle, or all valid framesets."""
    if not bag_path.is_file():
        raise FileNotFoundError(f"Bag file does not exist: {bag_path}")
    if frame_selection not in {"first", "middle", "all"}:
        raise ValueError("frame_selection must be 'first', 'middle', or 'all'")
    out_dir.mkdir(parents=True, exist_ok=True)

    playback_device = PlaybackDevice(str(bag_path))
    pipeline = Pipeline(playback_device)
    playback_config = Config()
    sensor_list = playback_device.get_sensor_list()
    enabled_sensor_types = []
    for sensor_index in range(len(sensor_list)):
        sensor_type = sensor_list[sensor_index].get_type()
        playback_config.enable_stream(sensor_type)
        enabled_sensor_types.append(str(sensor_type))
    if not enabled_sensor_types:
        raise RuntimeError(f"No playable sensors found in bag: {bag_path}")

    align_filter = AlignFilter(align_to_stream=OBStreamType.COLOR_STREAM)
    metadata: dict[str, Any] = {
        "source_bag": str(bag_path.resolve()),
        "processing_started_unix_s": time.time(),
        "alignment": "Orbbec SDK AlignFilter, depth-to-color (D2C)",
        "playback_enabled_sensors": enabled_sensor_types,
        "frames": [],
        "warnings": [
            "RGB PNG is decoded from the recorded color format; no enhancement applied.",
            "Raw depth NPY/PNG preserves the original uint16 samples.",
            "Colorized depth PNG is visualization only; do not use it as numeric depth data.",
            "Timestamp units are recorded as returned by the SDK; verify SDK documentation for your installed version.",
        ],
    }

    pipeline.start(playback_config)
    try:
        # wait_for_frames(timeout_ms) waits up to this many milliseconds for a
        # frameset. It is not a frame count. 1000 ms is a timeout, not a delay.
        # Keep the short bag's framesets so we can select its temporal midpoint.
        candidates = []
        consecutive_empty_reads = 0
        while max_frames is None or len(candidates) < max_frames:
            frameset = pipeline.wait_for_frames(1000)
            if frameset is None:
                # EOF signaling differs between SDK versions; use repeated empty
                # reads as a fallback. This should be validated on the user's build.
                consecutive_empty_reads += 1
                if consecutive_empty_reads >= 5:
                    break
                continue
            consecutive_empty_reads = 0
            if frameset.get_color_frame() is not None and frameset.get_depth_frame() is not None:
                candidates.append(frameset)

        if not candidates:
            raise RuntimeError(f"No framesets containing both color and depth found in {bag_path}")

        if frame_selection == "first":
            selected = [(0, candidates[0])]
        elif frame_selection == "middle":
            mid = len(candidates) // 2
            selected = [(mid, candidates[mid])]
        else:
            selected = list(enumerate(candidates))

        metadata["frames_seen_with_color_and_depth"] = len(candidates)
        metadata["frame_selection"] = frame_selection
        metadata["selected_frame_indices"] = [idx for idx, _ in selected]

        for frame_index, frameset in selected:
            color_frame = frameset.get_color_frame()
            depth_frame = frameset.get_depth_frame()
            if color_frame is None or depth_frame is None:
                continue

            frame_meta = extract_rgb_depth_timestamps(color_frame, depth_frame, name, out_dir, frame_index)
            raw_depth = depth_frame_to_raw_u16(depth_frame)
            try:
                raw_scale = float(depth_frame.get_depth_scale())
            except Exception:
                raw_scale = None

            frame_meta["raw_masks"] = save_masks(
                raw_depth,
                out_dir,
                f"{name}_mask_raw",
                depth_scale=raw_scale,
                discontinuity_threshold_mm=discontinuity_threshold_mm,
            )

            aligned = align_depth_to_color(frameset, align_filter)
            if aligned is not None:
                stem = f"{name}_depth_aligned_to_rgb"
                save_aligned_depth(aligned, out_dir, stem)
                # if raw_scale is not None:
                #     save_colorized_depth(aligned, raw_scale, out_dir, stem)
                frame_meta["aligned_depth"] = {
                    "array_file": f"{stem}.npy",
                    "png_file": f"{stem}.png",
                    "width": int(aligned.shape[1]),
                    "height": int(aligned.shape[0]),
                    "dtype": str(aligned.dtype),
                    "colorized_visualization_file": f"{stem}_depth_colorized.png" if raw_scale is not None else None,
                }
                frame_meta["aligned_masks"] = save_masks(
                    aligned,
                    out_dir,
                    f"{name}_mask_aligned",
                    depth_scale=raw_scale,
                    discontinuity_threshold_mm=discontinuity_threshold_mm,
                )
            else:
                frame_meta["aligned_depth"] = None
                frame_meta.setdefault("warnings", []).append(
                    "SDK AlignFilter did not return an aligned depth frame for this frameset."
                )
            metadata["frames"].append(frame_meta)
    finally:
        try:
            pipeline.stop()
        except Exception:
            pass

    metadata["processing_finished_unix_s"] = time.time()
    metadata["frames_processed"] = len(metadata["frames"])
    # with (out_dir / "timestamps.json").open("w", encoding="utf-8") as f:
    #     json.dump(json_safe(metadata), f, indent=2, ensure_ascii=False)

    print(f"Scanned {metadata.get('frames_seen_with_color_and_depth', 0)} valid framesets")
    print(f"Processed {len(metadata['frames'])} frame(s) using selection={frame_selection!r}")
    print(f"Output: {out_dir.resolve()}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Astra 2 Steps 11–13: extract RGB/depth/timestamps, align depth, create masks."
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--bag", help="Process one input Orbbec .bag file")
    source.add_argument("--input-dir", help="Process every .bag recursively under this directory")
    parser.add_argument("--out", required=True, help="Output directory (for --input-dir, each bag gets its own subfolder)")
    parser.add_argument("--name", required=True, help="Output files prefix")
    parser.add_argument("--max-frames", type=int, default=None, help="Optional maximum framesets to scan; mainly for testing")
    parser.add_argument(
        "--frame-selection",
        choices=("middle", "first", "all"),
        default="middle",
        help="Which valid frameset(s) to save from each bag; default: middle",
    )
    parser.add_argument(
        "--discontinuity-threshold-mm",
        type=float,
        default=100.0,
        help="Depth edge threshold for conservative mask (default 100 mm)",
    )
    parser.add_argument(
        "--no-discontinuity-mask",
        action="store_true",
        help="Only make basic validity masks; skip edge mask",
    )
    args = parser.parse_args()

    threshold = None if args.no_discontinuity_mask else args.discontinuity_threshold_mm
    out_root = Path(args.out)
    if args.bag:
        process_bag(
            Path(args.bag),
            out_root,
            args.name,
            max_frames=args.max_frames,
            frame_selection=args.frame_selection,
            discontinuity_threshold_mm=threshold,
        )
    else:
        input_root = Path(args.input_dir)
        if not input_root.is_dir():
            raise NotADirectoryError(f"Input directory does not exist: {input_root}")
        bags = sorted(input_root.rglob("*.bag"))
        if not bags:
            raise FileNotFoundError(f"No .bag files found under: {input_root}")
        print(f"Found {len(bags)} .bag file(s) under {input_root}")
        for i, bag_path in enumerate(bags, start=1):
            # Mirror relative folders to avoid collisions if nested bags share a basename.
            relative_parent = bag_path.parent.relative_to(input_root)
            bag_out = out_root / relative_parent / bag_path.stem
            print(f"\n[{i}/{len(bags)}] {bag_path}")
            try:
                process_bag(
                    bag_path,
                    bag_out,
                    args.name,
                    max_frames=args.max_frames,
                    frame_selection=args.frame_selection,
                    discontinuity_threshold_mm=threshold,
                )
            except Exception as exc:
                print(f"ERROR processing {bag_path}: {exc}")


if __name__ == "__main__":
    main()
