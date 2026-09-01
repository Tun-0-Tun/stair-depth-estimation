"""The single implementation of the FPS / latency protocol.

Same rule as ``depth_metrics``: no other timing code in the repository.  FPS is
an acceptance metric in the SOW, so it has to be measured identically for every
method, otherwise the comparison is meaningless.

Protocol (deliberately strict, see ``docs/metrics_protocol.md``)
---------------------------------------------------------------
* ``batch_size = 1`` -- we care about single-frame latency on the robot.
* Fixed input resolution, recorded in the result.
* ``warmup`` iterations are discarded (CUDA kernel autotuning, lazy cuDNN
  algorithm selection, MPS graph compilation).
* The device is synchronised before starting the clock and before stopping it.
* Timing covers **only** ``predict``: no disk I/O, no metric computation, no
  host<->device copies that a deployed pipeline would not do.  Pre/post
  processing that lives inside the adapter *is* included, because it is part of
  the method.
* We report median as the headline latency (robust to a stray scheduler hiccup)
  and also mean / p95 / std so a noisy measurement is visible.
"""

from __future__ import annotations

import platform
import statistics
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from typing import Any

__all__ = ["RuntimeConfig", "describe_device", "measure_runtime"]


@dataclass(frozen=True)
class RuntimeConfig:
    warmup: int = 10
    iters: int = 50
    batch_size: int = 1
    synchronize: bool = True


def _synchronizer(device: str | None) -> Callable[[], None]:
    """Return a no-arg function that blocks until ``device`` is idle."""
    if not device:
        return lambda: None
    dev = str(device).split(":")[0]
    try:
        import torch
    except ImportError:
        return lambda: None
    if dev == "cuda" and torch.cuda.is_available():
        return torch.cuda.synchronize
    if dev == "mps" and getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.mps.synchronize
    return lambda: None


def describe_device(device: str | None = None) -> dict[str, Any]:
    """Everything about the machine that a runtime number is only valid on."""
    info: dict[str, Any] = {
        "device": device or "cpu",
        "platform": platform.platform(),
        "machine": platform.machine(),
        "python": platform.python_version(),
    }
    try:
        import torch

        info["torch"] = torch.__version__
        if torch.cuda.is_available():
            info["gpu"] = torch.cuda.get_device_name(0)
            info["cuda"] = torch.version.cuda
        elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            info["gpu"] = "Apple MPS"
    except ImportError:
        info["torch"] = None
    return info


@dataclass
class RuntimeResult:
    fps: float
    latency_ms_median: float
    latency_ms_mean: float
    latency_ms_p95: float
    latency_ms_std: float
    iters: int
    warmup: int
    batch_size: int
    input_hw: tuple[int, int] | None = None
    device_info: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def measure_runtime(
    fn: Callable[[], Any],
    device: str | None = None,
    config: RuntimeConfig | None = None,
    input_hw: tuple[int, int] | None = None,
) -> RuntimeResult:
    """Time ``fn`` under the repository-wide protocol.

    ``fn`` must be a zero-argument closure that performs exactly one forward
    pass on one sample, e.g. ``lambda: model.predict(rgb, lr_depth)``.

    FPS is ``batch_size / median_latency_seconds``; with the mandated
    ``batch_size = 1`` that is simply the reciprocal of the median latency.
    """
    cfg = config or RuntimeConfig()
    sync = _synchronizer(device) if cfg.synchronize else (lambda: None)

    for _ in range(cfg.warmup):
        fn()
    sync()

    samples_ms: list[float] = []
    for _ in range(cfg.iters):
        sync()
        t0 = time.perf_counter()
        fn()
        sync()
        samples_ms.append((time.perf_counter() - t0) * 1000.0)

    median = statistics.median(samples_ms)
    ordered = sorted(samples_ms)
    p95 = ordered[min(len(ordered) - 1, round(0.95 * (len(ordered) - 1)))]
    return RuntimeResult(
        fps=cfg.batch_size / (median / 1000.0) if median > 0 else float("inf"),
        latency_ms_median=median,
        latency_ms_mean=statistics.fmean(samples_ms),
        latency_ms_p95=p95,
        latency_ms_std=statistics.pstdev(samples_ms) if len(samples_ms) > 1 else 0.0,
        iters=cfg.iters,
        warmup=cfg.warmup,
        batch_size=cfg.batch_size,
        input_hw=tuple(input_hw) if input_hw else None,
        device_info=describe_device(device),
    )
