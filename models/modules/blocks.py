"""Reusable convolutional blocks.

These are deliberately small and independent so that a hypothesis can be handed
to one person as "swap block X for block Y in ``models/pipeline.py`` and report
the delta", without anyone touching the training loop or the metrics.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["ConvBlock", "DownBlock", "ResidualBlock", "UpBlock", "conv_bn_act"]


def conv_bn_act(
    in_ch: int, out_ch: int, kernel: int = 3, stride: int = 1, norm: bool = True, act: bool = True
) -> nn.Sequential:
    layers: list[nn.Module] = [
        nn.Conv2d(in_ch, out_ch, kernel, stride, padding=kernel // 2, bias=not norm)
    ]
    if norm:
        layers.append(nn.BatchNorm2d(out_ch))
    if act:
        layers.append(nn.ReLU(inplace=True))
    return nn.Sequential(*layers)


class ConvBlock(nn.Module):
    """Two 3x3 convolutions, the standard U-Net stage."""

    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.body = nn.Sequential(conv_bn_act(in_ch, out_ch), conv_bn_act(out_ch, out_ch))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.body(x)


class ResidualBlock(nn.Module):
    def __init__(self, ch: int) -> None:
        super().__init__()
        self.body = nn.Sequential(conv_bn_act(ch, ch), conv_bn_act(ch, ch, act=False))
        self.act = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(x + self.body(x))


class DownBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.pool = nn.MaxPool2d(2)
        self.conv = ConvBlock(in_ch, out_ch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(self.pool(x))


class UpBlock(nn.Module):
    """Bilinear upsample + skip concat + conv.

    Bilinear rather than a transposed convolution: transposed convolutions put
    checkerboard artefacts on depth edges, which is precisely the structure a
    stair scene is made of.
    """

    def __init__(self, in_ch: int, skip_ch: int, out_ch: int) -> None:
        super().__init__()
        self.conv = ConvBlock(in_ch + skip_ch, out_ch)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        return self.conv(torch.cat([x, skip], dim=1))
