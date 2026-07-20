"""Reusable layers for the v2 model collection.

All components are initialized from scratch.  Nothing in this module downloads
weights or depends on an external model repository.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn
import torch.nn.functional as F


def conv1x1(in_channels: int, out_channels: int, stride: int = 1) -> nn.Conv2d:
    return nn.Conv2d(in_channels, out_channels, 1, stride=stride, bias=False)


def conv3x3(
    in_channels: int,
    out_channels: int,
    stride: int = 1,
    groups: int = 1,
    dilation: int = 1,
) -> nn.Conv2d:
    return nn.Conv2d(
        in_channels,
        out_channels,
        3,
        stride=stride,
        padding=dilation,
        dilation=dilation,
        groups=groups,
        bias=False,
    )


class ConvNormAct(nn.Sequential):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        groups: int = 1,
        activation: type[nn.Module] = nn.SiLU,
    ) -> None:
        padding = kernel_size // 2
        super().__init__(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size,
                stride=stride,
                padding=padding,
                groups=groups,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            activation(inplace=True),
        )


class SqueezeExcitation(nn.Module):
    def __init__(self, channels: int, reduction: int = 8) -> None:
        super().__init__()
        hidden = max(8, channels // reduction)
        self.net = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, hidden, 1),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden, channels, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: Tensor) -> Tensor:
        return x * self.net(x)


class EfficientChannelAttention(nn.Module):
    """ECA attention without a parameter-heavy bottleneck."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        kernel_size = int(abs((math.log2(channels) + 1.0) / 2.0))
        kernel_size = kernel_size if kernel_size % 2 else kernel_size + 1
        self.conv = nn.Conv1d(1, 1, kernel_size, padding=kernel_size // 2, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        weights = F.adaptive_avg_pool2d(x, 1).flatten(2).transpose(1, 2)
        weights = self.conv(weights).transpose(1, 2).unsqueeze(-1).sigmoid()
        return x * weights


class GeMPool2d(nn.Module):
    """Learnable generalized-mean pooling."""

    def __init__(self, p: float = 3.0, eps: float = 1e-6) -> None:
        super().__init__()
        self.p = nn.Parameter(torch.tensor(float(p)))
        self.eps = eps

    def forward(self, x: Tensor) -> Tensor:
        p = self.p.clamp(1.0, 8.0)
        return F.adaptive_avg_pool2d(x.clamp_min(self.eps).pow(p), 1).pow(1.0 / p)


class DropPath(nn.Module):
    def __init__(self, probability: float = 0.0) -> None:
        super().__init__()
        self.probability = probability

    def forward(self, x: Tensor) -> Tensor:
        if not self.training or self.probability == 0.0:
            return x
        keep = 1.0 - self.probability
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        mask = x.new_empty(shape).bernoulli_(keep)
        return x * mask / keep


class MBConv(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        expansion: int = 4,
        drop_path: float = 0.0,
    ) -> None:
        super().__init__()
        hidden = in_channels * expansion
        self.use_skip = stride == 1 and in_channels == out_channels
        self.expand = (
            ConvNormAct(in_channels, hidden, 1) if expansion != 1 else nn.Identity()
        )
        expanded_channels = hidden if expansion != 1 else in_channels
        self.depthwise = ConvNormAct(
            expanded_channels,
            expanded_channels,
            3,
            stride,
            groups=expanded_channels,
        )
        self.se = SqueezeExcitation(expanded_channels, reduction=8)
        self.project = nn.Sequential(
            conv1x1(expanded_channels, out_channels),
            nn.BatchNorm2d(out_channels),
        )
        self.drop_path = DropPath(drop_path)

    def forward(self, x: Tensor) -> Tensor:
        out = self.project(self.se(self.depthwise(self.expand(x))))
        return x + self.drop_path(out) if self.use_skip else out


class ResidualBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        dilation: int = 1,
    ) -> None:
        super().__init__()
        self.body = nn.Sequential(
            ConvNormAct(in_channels, out_channels, 3, stride),
            conv3x3(out_channels, out_channels, dilation=dilation),
            nn.BatchNorm2d(out_channels),
        )
        self.skip = (
            nn.Identity()
            if stride == 1 and in_channels == out_channels
            else nn.Sequential(
                conv1x1(in_channels, out_channels, stride), nn.BatchNorm2d(out_channels)
            )
        )
        self.activation = nn.ReLU(inplace=True)

    def forward(self, x: Tensor) -> Tensor:
        return self.activation(self.body(x) + self.skip(x))


def initialize_weights(module: nn.Module) -> None:
    """Consistent from-scratch initialization for every architecture."""
    if isinstance(module, nn.Conv2d):
        nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, (nn.BatchNorm2d, nn.LayerNorm, nn.GroupNorm)):
        if module.weight is not None:
            nn.init.ones_(module.weight)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Linear):
        nn.init.trunc_normal_(module.weight, std=0.02)
        if module.bias is not None:
            nn.init.zeros_(module.bias)


def count_parameters(model: nn.Module) -> int:
    return sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
