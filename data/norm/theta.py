"""Normal-map consumers: resizing and angle to the camera (theta).

``get_theta_map``: theta = arccos|n . a|, ``a`` the optical axis or the per-pixel
viewing ray.  NaN where the normal is missing; callers fill those with
``utils.misc.fill_nan_nearest``.

Resizing (``resize_normals`` / ``resize_theta``)
------------------------------------------------
``nearest`` / ``bilinear``: ``utils.misc.resize_depth`` per channel.
``guided`` (default): surface-aware bilinear.  Per output pixel: 4 bilinear taps
at ``align_corners=False`` pixel-centre positions; the nearest tap is the
reference; a tap is kept iff valid and within ``normal_threshold_deg`` of the reference
(normals: arccos|n_i . n_ref|; theta: |t_i - t_ref|); output = bilinear-weighted
mean of kept taps, weights renormalised (normals re-unit-normalised); missing
reference -> missing output.  Plane interior == bilinear, crease / hole ==
nearest.  Downscaling > 2x: repeated 2x steps (surface-aware box filter) before
the final step, against aliasing.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from utils.misc import resize_depth

__all__ = ["get_theta_map", "resize_normals", "resize_theta"]

_RESIZE_MODES = ("guided", "nearest", "bilinear")
_RESIZE_ORDERS = ("normals", "theta")

# ---------------------------------------------------------------------------
# surface-aware ("guided") resampling
# ---------------------------------------------------------------------------


def _tap_grid(hn: int, wn: int, h: int, w: int):
    """Bilinear taps of every output pixel of an ``(h, w)`` image sampled from ``(hn, wn)``.

    Pixel-centre convention of ``F.interpolate(align_corners=False)``: output pixel
    ``(i, j)`` sits at fractional source index ``sy = (i + 0.5) * hn / h - 0.5``,
    ``sx = (j + 0.5) * wn / w - 0.5``.  The taps are the two source pixels
    straddling that point per axis, ``y0 = floor(sy)``, ``y1 = y0 + 1`` (same for x).

    Args:
        hn: Source image height in pixels.
        wn: Source image width in pixels.
        h: Output image height in pixels.
        w: Output image width in pixels.

    Returns:
        y_taps: (4, h, w) int.  Source row index of each tap for each output pixel,
            clamped to ``[0, hn - 1]`` (a tap past the border repeats the border
            pixel, so its weight is not lost).
        x_taps: (4, h, w) int.  Source column index of each tap, clamped to ``[0, wn - 1]``.
        tap_weights: (4, h, w) float64.  Bilinear weight of each tap,
            ``(1-fy)(1-fx), (1-fy)fx, fy(1-fx), fy fx`` with ``fy = sy - y0``,
            ``fx = sx - x0``; the four sum to 1 for every output pixel.
        ref: (h, w) int in ``0..3``.  Index along the first axis of the tap
            whose centre is nearest to ``(sy, sx)``, i.e. the pixel ``nearest``
            resizing would return: ``2 * (fy >= 0.5) + (fx >= 0.5)``.

    Tap order along the first axis: ``(y0,x0), (y0,x1), (y1,x0), (y1,x1)``.
    """
    # Output pixel centres expressed as fractional source indices (1-D, per axis).
    sy = (np.arange(h, dtype=np.float64) + 0.5) * hn / h - 0.5
    sx = (np.arange(w, dtype=np.float64) + 0.5) * wn / w - 0.5

    # Lower tap = floor; fractional part = distance to it, i.e. weight of the upper tap.
    y0f, x0f = np.floor(sy), np.floor(sx)
    fy, fx = sy - y0f, sx - x0f
    y0, x0 = y0f.astype(int), x0f.astype(int)

    # Upper tap = lower + 1.
    # Clamp both: at the border the two taps coincide and
    # their weights add up, so no weight is lost.
    y0c, y1c = np.clip(y0, 0, hn - 1), np.clip(y0 + 1, 0, hn - 1)
    x0c, x1c = np.clip(x0, 0, wn - 1), np.clip(x0 + 1, 0, wn - 1)

    # Broadcast the 1-D indices to (4, h, w): rows repeat along x, columns along y.
    col = np.ones((1, w), int)
    row = np.ones((h, 1), int)
    y_taps = np.stack(
        [y0c[:, None] * col, y0c[:, None] * col, y1c[:, None] * col, y1c[:, None] * col]
    )
    x_taps = np.stack(
        [row * x0c[None, :], row * x1c[None, :], row * x0c[None, :], row * x1c[None, :]]
    )

    # Separable bilinear weights:
    # (1-f) for the lower tap, f for the upper, per axis.
    tap_weights = np.stack(
        [
            (1 - fy)[:, None] * (1 - fx)[None, :],
            (1 - fy)[:, None] * fx[None, :],
            fy[:, None] * (1 - fx)[None, :],
            fy[:, None] * fx[None, :],
        ]
    )

    # Nearest tap: upper one on an axis iff f >= 0.5; encode as 2*row_bit + col_bit.
    ref = 2 * (fy >= 0.5).astype(int)[:, None] + (fx >= 0.5).astype(int)[None, :]
    return y_taps, x_taps, tap_weights, ref


def _angle_normals(taps: np.ndarray, ref: np.ndarray) -> np.ndarray:
    """Angle between each tap normal and the reference normal"""
    cos = np.abs(np.sum(taps * ref, axis=-1))
    return np.arccos(np.clip(cos, 0.0, 1.0))


def _angle_theta(taps: np.ndarray, ref: np.ndarray) -> np.ndarray:
    """Difference between each tap angle and the reference angle"""
    return np.abs(taps[..., 0] - ref[..., 0])


def _guided_step(
    src: np.ndarray,
    valid: np.ndarray,
    out_shape: tuple[int, int],
    tol: float,
    angle_fn: Callable[[np.ndarray, np.ndarray], np.ndarray],
    renormalise: bool,
) -> tuple[np.ndarray, np.ndarray]:
    """One surface-aware bilinear resampling of ``src`` (H_n, W_n, C), finite everywhere."""

    hn, wn = src.shape[:2]  # source shape
    h, w = out_shape  # output shape

    # y_tabs and x_tabs store y and x chords of four closest pixels
    # tap_weights stores weights to each of this taps
    # ref stores the closest pixel
    y_taps, x_taps, tap_weights, ref = _tap_grid(hn, wn, h, w)

    taps = src[y_taps, x_taps]  # (4, H, W, C)
    valid_taps = valid[y_taps, x_taps]  # (4, H, W)
    ref_value = np.take_along_axis(taps, ref[None, ..., None], axis=0)  # (1, H, W, C)
    valid_ref = np.take_along_axis(valid_taps, ref[None], axis=0)[0]  # (H, W)

    keep = valid_taps & (angle_fn(taps, ref_value) <= tol)
    wk = np.where(keep, tap_weights, 0.0)
    wsum = wk.sum(axis=0)
    out = (wk[..., None] * taps).sum(axis=0) / np.maximum(wsum, 1e-12)[..., None]
    valid_out = valid_ref & (wsum > 0)
    if renormalise:
        norm = np.linalg.norm(out, axis=-1, keepdims=True)
        out = out / np.maximum(norm, 1e-12)
        valid_out &= norm[..., 0] > 1e-6
    out = np.where(valid_out[..., None], out, 0.0)
    return out, valid_out


def _resize_guided(
    src: np.ndarray,
    valid: np.ndarray,
    out_shape: tuple[int, int],
    normal_threshold_deg: float,
    angle_fn: Callable[[np.ndarray, np.ndarray], np.ndarray],
    renormalise: bool,
) -> tuple[np.ndarray, np.ndarray]:
    """Surface-aware resize of ``src`` (H_n, W_n, C) to ``out_shape``; see the module docstring.

    ``src`` must be finite (missing pixels are given by ``valid``, their values are
    ignored).  Returns ``(out, out_valid)`` with ``out`` zero where invalid.
    """
    tol = float(np.radians(normal_threshold_deg))
    src = src.astype(np.float64)
    h, w = out_shape
    hn, wn = src.shape[:2]
    # Shrink by at most 2x per step so no source pixel is skipped (anti-aliasing).
    while max(hn / h, wn / w) > 2.0:
        mid = (max(h, (hn + 1) // 2), max(w, (wn + 1) // 2))
        src, valid = _guided_step(src, valid, mid, tol, angle_fn, renormalise)
        hn, wn = mid
    return _guided_step(src, valid, (h, w), tol, angle_fn, renormalise)


def _check_resize_args(in_shape: tuple[int, int], out_shape: tuple[int, int], mode: str) -> None:
    if mode not in _RESIZE_MODES:
        raise ValueError(f"mode must be one of {_RESIZE_MODES}, got {mode!r}")


def resize_normals(
    normal: np.ndarray,
    out_shape: tuple[int, int],
    *,
    mode: str = "guided",
    normal_threshold_deg: float = 10.0,
) -> np.ndarray:
    """Resize a normal map to ``out_shape``.

    Args:
        normal: (H_n, W_n, 3), any scale, camera frame, any sign convention.  Zero
            or non-finite vectors are missing.
        out_shape: ``(H, W)``; same aspect ratio as ``normal`` within 2 %.
        mode: ``guided`` (default), ``nearest``, ``bilinear``
        normal_threshold_deg: threshold for guided resize.

    Returns:
        (H, W, 3) float32 unit vectors, zero where missing.
    """
    n = np.asarray(normal, dtype=np.float32)
    if n.ndim != 3 or n.shape[2] != 3:
        raise ValueError(f"normal must be (H, W, 3), got {n.shape}")
    _check_resize_args(n.shape[:2], tuple(out_shape), mode)
    n = np.where(np.isfinite(n), n, 0.0)
    norm = np.linalg.norm(n, axis=-1, keepdims=True)
    valid = norm[..., 0] > 1e-6
    n = np.where(valid[..., None], n / np.maximum(norm, 1e-6), 0.0)

    if n.shape[:2] == tuple(out_shape):
        return n.astype(np.float32)
    if mode == "guided":
        out, _ = _resize_guided(
            n, valid, tuple(out_shape), normal_threshold_deg, _angle_normals, True
        )
        return out.astype(np.float32)
    out = np.stack([resize_depth(n[..., c], tuple(out_shape), mode=mode) for c in range(3)], -1)
    norm = np.linalg.norm(out, axis=-1, keepdims=True)
    return np.where(norm > 1e-6, out / np.maximum(norm, 1e-6), 0.0).astype(np.float32)


def resize_theta(
    theta: np.ndarray,
    out_shape: tuple[int, int],
    *,
    mode: str = "guided",
    normal_threshold_deg: float = 10.0,
) -> np.ndarray:
    """Resize an angle map (radians, NaN = missing) to ``out_shape``.

    Same modes as :func:`resize_normals`; with ``guided`` the surface test is
    ``|theta_tap - theta_ref| <= normal_threshold_deg``.  With ``bilinear`` a NaN taints
    its neighbours (``utils.misc.fill_nan_nearest`` recovers them); ``nearest`` does not.

    Returns:
        (H, W) float32, NaN where missing.
    """
    t = np.asarray(theta, dtype=np.float32)
    if t.ndim != 2:
        raise ValueError(f"theta must be (H, W), got {t.shape}")
    _check_resize_args(t.shape, tuple(out_shape), mode)
    if t.shape == tuple(out_shape):
        return t
    if mode == "guided":
        valid = np.isfinite(t)
        src = np.where(valid, t, 0.0)[..., None]
        out, ok = _resize_guided(
            src, valid, tuple(out_shape), normal_threshold_deg, _angle_theta, False
        )
        return np.where(ok, out[..., 0], np.nan).astype(np.float32)
    return resize_depth(t, tuple(out_shape), mode=mode).astype(np.float32)


# ---------------------------------------------------------------------------
# theta
# ---------------------------------------------------------------------------


def _theta_from_unit_normals(
    n: np.ndarray,
    valid: np.ndarray,
    angle_reference: str,
    fx: float,
    fy: float | None,
    cx: float | None,
    cy: float | None,
) -> np.ndarray:
    """Angle between unit normals ``n`` (H, W, 3) and the axis / ray"""
    h, w = n.shape[:2]
    if angle_reference == "optical_axis":
        cos = np.abs(n[..., 2])
    elif angle_reference == "viewing_ray":
        fy = fx if fy is None else fy
        cx = (w - 1) / 2.0 if cx is None else cx
        cy = (h - 1) / 2.0 if cy is None else cy
        vs, us = np.mgrid[0:h, 0:w]
        ray = np.stack([(us - cx) / fx, (vs - cy) / fy, np.ones((h, w))], axis=-1)
        ray /= np.linalg.norm(ray, axis=-1, keepdims=True)
        cos = np.abs(np.sum(n * ray, axis=-1))
    else:
        raise ValueError(
            f"angle_reference must be optical_axis or viewing_ray, got {angle_reference!r}"
        )
    theta = np.arccos(np.clip(cos, 0.0, 1.0))
    return np.where(valid, theta, np.nan).astype(np.float32)


def get_theta_map(
    normal: np.ndarray,
    out_shape: tuple[int, int],
    *,
    resize_mode: str = "guided",
    resize_order: str = "normals",
    normal_threshold_deg: float = 10.0,
    angle_reference: str = "optical_axis",
    fx: float = 570.0,
    fy: float | None = None,
    cx: float | None = None,
    cy: float | None = None,
) -> np.ndarray:
    """Angle theta (radians) between surface normal and optical axis / viewing ray.

    If the normal map's size differs from ``out_shape`` it is resized with
    :func:`resize_normals` / :func:`resize_theta`.  With the default ``guided``
    mode the two ``resize_order`` choices agree inside a plane and both stay
    sharp at creases; they differ only in *what* is averaged along a plane:
    vectors (``normals``) or angles (``theta``).

    Args:
        normal: (H_n, W_n, 3) float, unit or unnormalised, in the camera frame
            (x right, y down, z forward).  Any sign convention: only ``|cos|`` is
            used.  A zero or non-finite vector marks a pixel with no normal.
        out_shape: ``(H, W)`` of the depth map the angles are for.  Must have the
            same aspect ratio as ``normal`` (within 2 %).
        resize_mode: ``guided`` (default: bilinear inside a surface, nearest across
            a crease or hole), ``nearest`` or ``bilinear`` (``utils.misc.resize_depth``;
            bilinear blends a tread with a riser into a phantom ~45 deg normal).
        resize_order: ``normals`` (default): resize the vectors, then compute theta
            at ``out_shape`` -- what native-normal datasets expect.  ``theta``:
            compute theta at the normal map's resolution (intrinsics rescaled
            accordingly), then resize the angle map.
        normal_threshold_deg: ``guided`` only; taps differing from the reference tap by
            more than this angle count as another surface and are dropped.
        angle_reference: ``optical_axis``: theta against the camera z axis
            (``arccos |n_z|``, as in Nguyen et al. 2012).  ``viewing_ray``: theta
            against the per-pixel ray through the pinhole below.
        fx: Horizontal focal length in pixels at ``out_shape`` resolution
            (570 ~ 640x480).  ``viewing_ray`` only.
        fy: Vertical focal length in pixels; ``None`` means ``fx``.  ``viewing_ray`` only.
        cx: Principal point x in pixels; ``None`` means the image centre.  ``viewing_ray`` only.
        cy: Principal point y in pixels; ``None`` means the image centre.  ``viewing_ray`` only.

    Returns:
        float32 (H, W) angles in radians, in ``[0, pi/2]``; NaN where the normal
        is unusable (zero / non-finite after resizing).

    Raises:
        ValueError: wrong ``normal`` shape, aspect-ratio mismatch, or unknown
            ``resize_mode`` / ``resize_order`` / ``angle_reference``.
    """
    if resize_order not in _RESIZE_ORDERS:
        raise ValueError(f"resize_order must be one of {_RESIZE_ORDERS}, got {resize_order!r}")
    norm = np.asarray(normal, dtype=np.float32)
    if norm.ndim != 3 or norm.shape[2] != 3:
        raise ValueError(f"normal must be (H, W, 3), got {norm.shape}")
    h, w = out_shape
    hn, wn = norm.shape[:2]

    if resize_order == "normals" or (hn, wn) == (h, w):
        norm = resize_normals(
            norm, (h, w), mode=resize_mode, normal_threshold_deg=normal_threshold_deg
        )
        valid = np.linalg.norm(norm, axis=-1) > 1e-6
        return _theta_from_unit_normals(norm, valid, angle_reference, fx, fy, cx, cy)

    # theta order: angles at the source resolution with intrinsics rescaled to it
    _check_resize_args((hn, wn), (h, w), resize_mode)
    norm = resize_normals(norm, (hn, wn), mode=resize_mode)  # no resize: normalise + mark missing
    valid = np.linalg.norm(norm, axis=-1) > 1e-6
    sy, sx = hn / h, wn / w
    fy_src = (fx if fy is None else fy) * sy
    cx_src = None if cx is None else (cx + 0.5) * sx - 0.5
    cy_src = None if cy is None else (cy + 0.5) * sy - 0.5
    theta = _theta_from_unit_normals(norm, valid, angle_reference, fx * sx, fy_src, cx_src, cy_src)
    return resize_theta(theta, (h, w), mode=resize_mode, normal_threshold_deg=normal_threshold_deg)
