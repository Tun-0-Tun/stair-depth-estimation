"""Check the mmcv shim against the DCNv2 formula, written out per pixel.

DORNet's deformable convolutions run through the shim, so if it is subtly wrong
DORNet produces plausible numbers that are not DORNet's.  The reference is the
modulated deformable convolution definition (Zhu et al., DCNv2) transcribed
literally -- slow and loop-shaped, so it cannot share a bug with torchvision.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torchvision")

from baselines.mmcv_shim import modulated_deform_conv2d  # noqa: E402


def bilinear(img: torch.Tensor, y: float, x: float) -> torch.Tensor:
    """img (C, H, W) sampled at (y, x); zero outside the image."""
    c, h, w = img.shape
    y0, x0 = int(torch.floor(torch.tensor(y))), int(torch.floor(torch.tensor(x)))
    out = torch.zeros(c)
    for yy, wy in ((y0, 1 - (y - y0)), (y0 + 1, y - y0)):
        for xx, wx in ((x0, 1 - (x - x0)), (x0 + 1, x - x0)):
            if 0 <= yy < h and 0 <= xx < w:
                out = out + wy * wx * img[:, yy, xx]
    return out


def reference(x, offset, mask, weight, bias, stride, padding, dilation):
    """y(p) = sum_k w_k * m_k * x(p0 + p_k + dp_k); offset[2k] = dy_k, offset[2k+1] = dx_k."""
    _, _, h, w = x.shape
    co, _, kh, kw = weight.shape
    ho = (h + 2 * padding - dilation * (kh - 1) - 1) // stride + 1
    wo = (w + 2 * padding - dilation * (kw - 1) - 1) // stride + 1
    out = torch.zeros(1, co, ho, wo)
    for r in range(ho):
        for c in range(wo):
            for i in range(kh):
                for j in range(kw):
                    k = i * kw + j
                    py = r * stride - padding + i * dilation + offset[0, 2 * k, r, c].item()
                    px = c * stride - padding + j * dilation + offset[0, 2 * k + 1, r, c].item()
                    val = bilinear(x[0], py, px) * mask[0, k, r, c]  # (Ci,)
                    out[0, :, r, c] += weight[:, :, i, j] @ val
    return out + bias.view(1, -1, 1, 1)


@pytest.mark.parametrize("stride,padding,dilation", [(1, 1, 1), (1, 0, 1), (2, 1, 1)])
def test_shim_matches_dcnv2_formula(stride, padding, dilation):
    g = torch.Generator().manual_seed(0)
    ci, co, k, h, w = 3, 4, 3, 7, 6
    x = torch.randn(1, ci, h, w, generator=g)
    weight = torch.randn(co, ci, k, k, generator=g)
    bias = torch.randn(co, generator=g)
    ho = (h + 2 * padding - dilation * (k - 1) - 1) // stride + 1
    wo = (w + 2 * padding - dilation * (k - 1) - 1) // stride + 1
    offset = 1.5 * torch.randn(1, 2 * k * k, ho, wo, generator=g)  # fractional, some out of range
    mask = torch.sigmoid(torch.randn(1, k * k, ho, wo, generator=g))

    got = modulated_deform_conv2d(x, offset, mask, weight, bias, stride, padding, dilation, 1, 1)
    want = reference(x, offset, mask, weight, bias, stride, padding, dilation)
    torch.testing.assert_close(got, want, atol=1e-4, rtol=1e-4)


def test_shim_refuses_a_mismatched_group_count():
    x, w = torch.zeros(1, 4, 5, 5), torch.zeros(2, 2, 3, 3)
    with pytest.raises(ValueError, match="groups"):
        modulated_deform_conv2d(
            x, torch.zeros(1, 18, 5, 5), torch.ones(1, 9, 5, 5), w, None, 1, 1, 1, 1, 1
        )
