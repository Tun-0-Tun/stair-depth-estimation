"""Camera geometry of the ``ours`` frames: K conversion (crop + resize) and undistorted ray maps.

The RGB stream of the Astra 2 has Brown-Conrady lens distortion (Orbbec order k1..k6, p1, p2) and the
saved ``depth_aligned_to_rgb`` lives on the RGB pixel grid, so a pinhole K alone bends planes near the
image border.  ``frame_rays`` gives, for every pixel of the cropped / resized frame, the undistorted
ray ``(x/z, y/z)``; ``X = rays * z`` replaces the pinhole unprojection everywhere K is used.
"""

from __future__ import annotations

import numpy as np


def convert_K(K, size, crop, target_size):
    """Full-frame K (fx, fy, cx, cy) -> K after ``crop=(y0,y1,x0,x1)`` and resize to ``target_size=(h,w)``.

    Pixel-centre convention (cv2/torch resize): u' = (u - x0 + 0.5) * s - 0.5.
    """
    fx, fy, cx, cy = K
    y0, y1, x0, x1 = crop if crop else (0, size[1], 0, size[0])
    sy, sx = (target_size[0] / (y1 - y0), target_size[1] / (x1 - x0)) if target_size else (1.0, 1.0)
    return (fx * sx, fy * sy, (cx - x0 + 0.5) * sx - 0.5, (cy - y0 + 0.5) * sy - 0.5)


def frame_rays(shape, K_full, D_ob, crop, target_size, size=(1920, 1080)) -> np.ndarray:
    """``[2, H, W]`` float32 undistorted rays (x/z, y/z) for a frame of ``shape`` made by crop + resize.

    ``D_ob`` = Orbbec distortion ``(k1, k2, k3, k4, k5, k6, p1, p2)`` or ``None`` (pinhole).
    """
    import cv2

    H, W = shape
    y0, y1, x0, x1 = crop if crop else (0, size[1], 0, size[0])
    sy, sx = (target_size[0] / (y1 - y0), target_size[1] / (x1 - x0)) if target_size else (1.0, 1.0)
    v, u = np.mgrid[:H, :W].astype(np.float64)
    pts = np.stack([(u + 0.5) / sx - 0.5 + x0, (v + 0.5) / sy - 0.5 + y0], -1).reshape(-1, 1, 2)
    fx, fy, cx, cy = K_full
    Km = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1.0]])
    dist = None
    if D_ob is not None:
        k1, k2, k3, k4, k5, k6, p1, p2 = D_ob
        dist = np.array([k1, k2, p1, p2, k3, k4, k5, k6], np.float64)  # OpenCV rational-model order
    xy = cv2.undistortPoints(pts, Km, dist).reshape(H, W, 2)
    return xy.transpose(2, 0, 1).astype(np.float32)
