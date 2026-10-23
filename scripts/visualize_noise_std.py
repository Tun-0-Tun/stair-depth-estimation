#!/usr/bin/env python3
"""Plot the std of axial and lateral sensor noise against depth, one line per angle.

Two panels: axial noise (std along z, metres) and lateral noise (std on the surface, metres).
Each panel has depth z on x and one line per angle theta between the surface normal and the
optical axis.

    uv run --group notebooks python scripts/visualize_noise_std.py
    uv run --group notebooks python scripts/visualize_noise_std.py \
        --axial axial_noise --lateral lateral_noise --angles 0 20 40 60 80 --z-range 0.5 4

The parameters are read from the presets in ``configs/degradation/`` (``--axial``,
``--lateral``) through ``build_degradation``, so the curves match what the degradation applies.
The std formulas are copies of ``apply_axial_noise`` and ``apply_lateral_noise``; they are
evaluated on a grid, with no sampling.  Lateral sigma is in pixels in the model and is
converted to metres on the surface with ``sigma_m = sigma_px * z / fx``.

Both presets are Kinect values from the paper, not Astra 2 calibration: the figure is marked
``uncalibrated``.  These are model curves, not measurements, and not metrics.
matplotlib lives in the ``notebooks`` dependency group.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

import _bootstrap
import numpy as np

from data.degradation import AxialDegradation, LateralDegradation, build_degradation
from utils.config import load_config

DEFAULT_OUT_DIR = _bootstrap.REPO_ROOT / "outputs" / "noise_viz"  # gitignored
DEFAULT_ANGLES = (0.0, 30.0, 45.0, 60.0, 75.0)
DEFAULT_Z_RANGE = (0.6, 8.0)  # Astra 2 datasheet operating range, used when the config has none


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--axial", default="axial_noise", help="axial preset in configs/degradation/")
    p.add_argument(
        "--lateral", default="lateral_noise", help="lateral preset in configs/degradation/"
    )
    p.add_argument(
        "--angles",
        nargs="+",
        type=float,
        default=list(DEFAULT_ANGLES),
        metavar="DEG",
        help="angles theta between normal and optical axis, degrees (default: 0 30 45 60 75)",
    )
    p.add_argument(
        "--z-range",
        nargs=2,
        type=float,
        metavar=("MIN", "MAX"),
        default=None,
        help="depth range in metres (default: the axial preset's z_min/z_max, else 0.6 8.0)",
    )
    p.add_argument(
        "--n-points", type=int, default=200, help="depth samples per line (default: 200)"
    )
    p.add_argument("-o", "--out", type=Path, default=None, help="output PNG path")
    p.add_argument("--show", action="store_true", help="open a window instead of only saving")
    return p.parse_args(argv)


def load_degradations(
    axial_name: str, lateral_name: str
) -> tuple[AxialDegradation, LateralDegradation]:
    axial = build_degradation(dict(load_config("degradation", axial_name)))
    lateral = build_degradation(dict(load_config("degradation", lateral_name)))
    if not isinstance(axial, AxialDegradation):
        raise SystemExit(f"error: {axial_name!r} is not an axial_noise preset")
    if not isinstance(lateral, LateralDegradation):
        raise SystemExit(f"error: {lateral_name!r} is not a lateral_noise preset")
    return axial, lateral


def axial_sigma_m(z: np.ndarray, theta_deg: float, ax: AxialDegradation) -> np.ndarray:
    """Axial std in metres at depth ``z`` (m); same formula as ``apply_axial_noise``."""
    phi = np.clip(np.radians(theta_deg), 0.0, np.radians(ax.theta_max_deg))
    sigma = ax.sigma_floor_m + ax.sigma_depth_gain * (z - ax.reference_depth_m) ** 2
    grazing = ax.sigma_grazing_gain / np.sqrt(z) * phi**2 / (np.pi / 2 - phi) ** 2
    if np.radians(ax.grazing_start_deg) < phi:
        sigma = sigma + grazing
    return np.minimum(sigma, ax.sigma_max_m)


def lateral_sigma_m(z: np.ndarray, theta_deg: float, lat: LateralDegradation) -> np.ndarray:
    """Lateral std on the surface, metres, at depth ``z`` (m); same formula as
    ``apply_lateral_noise`` with pixels converted by ``z / fx``."""
    phi = np.clip(np.radians(theta_deg), 0.0, np.radians(lat.theta_max_deg))
    sigma_px = lat.sigma_floor_px + lat.sigma_angle_gain * phi / (np.pi / 2 - phi)
    sigma_px = np.minimum(sigma_px, lat.sigma_max_px)
    return sigma_px * z / lat.fx


def plot(
    axial: AxialDegradation,
    lateral: LateralDegradation,
    args: argparse.Namespace,
    z_lo: float,
    z_hi: float,
):
    import matplotlib

    if not args.show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    z = np.linspace(z_lo, z_hi, args.n_points)
    cmap = matplotlib.colormaps["viridis"]
    colors = cmap(np.linspace(0.0, 0.9, len(args.angles)))

    fig, (ax_a, ax_l) = plt.subplots(1, 2, figsize=(11, 4.6), squeeze=True)
    panels = (
        (ax_a, axial_sigma_m, axial, "axial noise: std along z", "std along z, m"),
        (ax_l, lateral_sigma_m, lateral, "lateral noise: std on the surface", "std on surface, m"),
    )
    for ax, sigma_fn, deg, title, ylabel in panels:
        for theta, color in zip(args.angles, colors, strict=True):
            ax.plot(z, sigma_fn(z, theta, deg), color=color, lw=1.8, label=f"{theta:g}°")
        ax.set_xlabel("depth z, m")
        ax.set_ylabel(ylabel)
        ax.set_title(title, fontsize=10)
        ax.set_xlim(z_lo, z_hi)
        ax.set_ylim(bottom=0.0)
        ax.grid(alpha=0.3)
        ax.legend(title="angle θ", fontsize=8, title_fontsize=8)

    fig.suptitle(
        f"{args.axial} / {args.lateral}  (uncalibrated: Kinect values, not Astra 2)",
        fontsize=11,
    )
    fig.tight_layout()
    return fig


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.n_points < 2:
        raise SystemExit("--n-points must be >= 2")
    if any(not 0.0 <= t < 90.0 for t in args.angles):
        raise SystemExit("--angles must be in [0, 90) degrees")

    try:
        axial, lateral = load_degradations(args.axial, args.lateral)
    except (FileNotFoundError, RuntimeError, ValueError, KeyError) as e:
        raise SystemExit(f"error: {e}") from None

    if args.z_range is not None:
        z_lo, z_hi = args.z_range
    else:
        z_lo = axial.z_min if axial.z_min is not None else DEFAULT_Z_RANGE[0]
        z_hi = axial.z_max if axial.z_max is not None else DEFAULT_Z_RANGE[1]
    if not 0.0 < z_lo < z_hi:
        raise SystemExit("--z-range needs 0 < MIN < MAX")

    fig = plot(axial, lateral, args, z_lo, z_hi)

    out = args.out or DEFAULT_OUT_DIR / f"std_vs_depth_{args.axial}_{args.lateral}.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=110)
    print(f"saved {out}")
    if args.show:
        import matplotlib.pyplot as plt

        plt.show()
    return 0


if __name__ == "__main__":
    sys.exit(main())
