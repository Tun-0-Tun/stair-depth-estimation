"""Portable stand-in for the ``BpOps`` CUDA extension DEPTHOR needs.

DEPTHOR's CSPN++ refinement (``third_party/depthor/src/models/refine.py``) does
``import BpOps`` and calls ``BpOps.Conv2dLocal_F``.  That extension comes from
BP-Net and builds against CUDA 12.1 only -- there is no CPU or MPS build, so
without this file DEPTHOR cannot run on a laptop at all.

This is **not** a reimplementation of DEPTHOR.  It supplies one missing
primitive, a local (per-pixel) convolution, transcribed from BP-Net's own
kernel ``conv2d_kernel_lf`` in ``exts/bp_cuda_kernel.cu``:

    z[b, c, r, k] = sum_{i,j = -(K-1)/2 .. (K-1)/2}
                        x[b, c, r+i, k+j] * y[b, c*K*K + (i+K//2)*K + (j+K//2), r, k]

with out-of-bounds neighbours skipped.  Skipping them is the same as reading
zero, so the whole thing is ``F.unfold`` with zero padding followed by a
weighted sum -- exact, not an approximation.  ``tests/test_bpops_shim.py``
checks it against a literal transcription of that CUDA loop.

Still: a number produced with the shim came from *our* kernel, not the authors'.
The adapter records ``bpops="shim"`` in the results row, and on a CUDA machine
the real extension is used instead -- install it and this file goes unused:

    git clone https://github.com/kakaxi314/BP-Net && cd BP-Net && python setup.py install
"""

from __future__ import annotations

import sys

import torch
import torch.nn.functional as F

__all__ = ["Conv2dLocal_F", "Conv2dLocal_B", "install"]


def Conv2dLocal_F(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:  # noqa: N802 - upstream name
    """Local convolution: every pixel gets its own kernel.

    ``x``: (B, Ci, H, W).  ``w``: (B, Ci*K*K, H, W), window-major within each
    channel.  Returns (B, Ci, H, W).
    """
    b, ci, h, width = x.shape
    k2 = w.shape[1] // ci
    k = int(round(k2**0.5))
    if k * k != k2 or k % 2 == 0:
        raise ValueError(
            f"weight has {w.shape[1]} channels for {ci} input channels, which is not "
            f"an odd square kernel per channel (got K={k})"
        )
    patches = F.unfold(x, k, padding=k // 2).view(b, ci, k * k, h, width)
    return (patches * w.view(b, ci, k * k, h, width)).sum(dim=2)


def Conv2dLocal_B(  # noqa: N802 - upstream name
    x: torch.Tensor, w: torch.Tensor, grad_out: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Gradients, obtained by differentiating the forward above.

    Exact by construction: autograd differentiates the same expression rather
    than a hand-written adjoint that could drift from it.  Only needed for
    training; evaluation runs under ``torch.no_grad()``.
    """
    with torch.enable_grad():
        xr = x.detach().requires_grad_(True)
        wr = w.detach().requires_grad_(True)
        out = Conv2dLocal_F(xr, wr)
        return torch.autograd.grad(out, (xr, wr), grad_out)


def install() -> bool:
    """Register this module as ``BpOps`` if the real extension is not present.

    Returns True if the shim was installed, False if the real one is available.
    """
    try:
        import BpOps  # noqa: F401

        return False
    except ImportError:
        sys.modules["BpOps"] = sys.modules[__name__]
        return True
