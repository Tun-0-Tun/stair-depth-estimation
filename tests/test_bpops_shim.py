"""Check the BpOps shim against a literal transcription of BP-Net's CUDA kernel.

The shim stands in for a CUDA extension we cannot build here, and DEPTHOR's
whole refinement stage runs through it. If it is subtly wrong, DEPTHOR produces
plausible-looking numbers that are not DEPTHOR's. So the reference below is a
line-by-line transcription of ``conv2d_kernel_lf`` from
``BP-Net/exts/bp_cuda_kernel.cu`` -- deliberately slow and loop-shaped, so it is
readable against the C++ rather than a second clever implementation that could
share a bug with the first.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from baselines.bpops_shim import Conv2dLocal_B, Conv2dLocal_F  # noqa: E402


def reference_conv2d_local_f(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Transcription of BP-Net's conv2d_kernel_lf, index for index.

    for i in -(K-1)/2 .. (K-1)/2:
      for j in -(K-1)/2 .. (K-1)/2:
        if out of bounds: continue
        result += x[b, c, r+i, k+j] * y[b, c*K*K + (i+K//2)*K + (j+K//2), r, k]
    """
    b, ci, n1, n2 = x.shape
    k = int(round((y.shape[1] // ci) ** 0.5))
    half = (k - 1) // 2
    # Built by stacking rather than by assigning into a buffer, so the reference
    # stays differentiable and the backward test can compare against it.
    rows = []
    for bb in range(b):
        chans = []
        for c in range(ci):
            grid = []
            for r in range(n1):
                line = []
                for col in range(n2):
                    acc = x.new_zeros(())
                    for i in range(-half, half + 1):
                        for j in range(-half, half + 1):
                            if r + i < 0 or r + i >= n1 or col + j < 0 or col + j >= n2:
                                continue
                            wc = c * k * k + (i + half) * k + (j + half)
                            acc = acc + x[bb, c, r + i, col + j] * y[bb, wc, r, col]
                    line.append(acc)
                grid.append(torch.stack(line))
            chans.append(torch.stack(grid))
        rows.append(torch.stack(chans))
    return torch.stack(rows)


@pytest.mark.parametrize("k", [3, 5, 7])
@pytest.mark.parametrize("ci", [1, 2])
def test_matches_the_cuda_kernel(k, ci):
    torch.manual_seed(0)
    x = torch.randn(2, ci, 6, 7, dtype=torch.float64)
    w = torch.randn(2, ci * k * k, 6, 7, dtype=torch.float64)
    assert torch.allclose(Conv2dLocal_F(x, w), reference_conv2d_local_f(x, w), atol=1e-10)


def test_border_neighbours_are_skipped_not_wrapped():
    """Out-of-frame taps must contribute nothing -- not wrap, not replicate."""
    x = torch.zeros(1, 1, 3, 3, dtype=torch.float64)
    x[0, 0, 0, 0] = 1.0
    w = torch.ones(1, 9, 3, 3, dtype=torch.float64)
    out = Conv2dLocal_F(x, w)
    # the corner value reaches only the pixels inside its 3x3 neighbourhood
    assert out[0, 0, 2, 2] == 0.0
    assert out[0, 0, 1, 1] == 1.0


def test_normalised_weights_preserve_a_constant_field():
    """CSPN normalises its weights to sum to 1, so a flat input must stay flat.

    This is the property DEPTHOR actually relies on; a transposed index or an
    off-by-one in the window would break it.
    """
    k = 5
    x = torch.full((1, 1, 8, 8), 2.5, dtype=torch.float64)
    w = torch.rand(1, k * k, 8, 8, dtype=torch.float64)
    w = w / w.sum(dim=1, keepdim=True)
    out = Conv2dLocal_F(x, w)
    inner = out[0, 0, k // 2 : -(k // 2), k // 2 : -(k // 2)]
    assert torch.allclose(inner, torch.full_like(inner, 2.5), atol=1e-12)


def test_backward_matches_autograd_through_the_reference():
    torch.manual_seed(1)
    k, ci = 3, 1
    x = torch.randn(1, ci, 5, 5, dtype=torch.float64)
    w = torch.randn(1, ci * k * k, 5, 5, dtype=torch.float64)
    g = torch.randn(1, ci, 5, 5, dtype=torch.float64)

    gx, gw = Conv2dLocal_B(x, w, g)

    xr = x.clone().requires_grad_(True)
    wr = w.clone().requires_grad_(True)
    reference_conv2d_local_f(xr, wr).backward(g)
    assert torch.allclose(gx, xr.grad, atol=1e-10)
    assert torch.allclose(gw, wr.grad, atol=1e-10)


def test_rejects_a_weight_shape_that_is_not_an_odd_square():
    x = torch.randn(1, 1, 4, 4)
    with pytest.raises(ValueError, match="odd square kernel"):
        Conv2dLocal_F(x, torch.randn(1, 8, 4, 4))
