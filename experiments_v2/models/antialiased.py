"""Anti-aliased ResNet-D with selective-kernel feature selection."""

from __future__ import annotations

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from .common import ConvNormAct, GeMPool2d, conv1x1, conv3x3, initialize_weights


class BlurPool(nn.Module):
    def __init__(self, channels: int, stride: int = 2) -> None:
        super().__init__()
        kernel = torch.tensor([1.0, 2.0, 1.0])
        kernel = (kernel[:, None] * kernel[None, :]) / 16.0
        self.register_buffer("kernel", kernel[None, None].repeat(channels, 1, 1, 1))
        self.channels = channels
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        return F.conv2d(
            x, self.kernel, stride=self.stride, padding=1, groups=self.channels
        )


class SelectiveKernelBlock(nn.Module):
    def __init__(
        self, in_channels: int, out_channels: int, downsample: bool = False
    ) -> None:
        super().__init__()
        stride = 2 if downsample else 1
        self.pre_downsample = BlurPool(in_channels, 2) if downsample else nn.Identity()
        self.small = nn.Sequential(
            conv3x3(in_channels, out_channels),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
        self.large = nn.Sequential(
            conv3x3(in_channels, out_channels, dilation=2),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
        hidden = max(16, out_channels // 8)
        self.selector = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(out_channels, hidden, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, out_channels * 2, 1),
        )
        self.skip = (
            nn.Identity()
            if not downsample and in_channels == out_channels
            else nn.Sequential(
                BlurPool(in_channels, stride) if downsample else nn.Identity(),
                conv1x1(in_channels, out_channels),
                nn.BatchNorm2d(out_channels),
            )
        )
        self.activation = nn.ReLU(inplace=True)

    def forward(self, x: Tensor) -> Tensor:
        identity = self.skip(x)
        x = self.pre_downsample(x)
        small, large = self.small(x), self.large(x)
        batch, channels, _, _ = small.shape
        weights = (
            self.selector(small + large).view(batch, 2, channels, 1, 1).softmax(dim=1)
        )
        fused = small * weights[:, 0] + large * weights[:, 1]
        return self.activation(fused + identity)


class AntiAliasedResNetSK(nn.Module):
    def __init__(self, num_classes: int = 18) -> None:
        super().__init__()
        self.stem = nn.Sequential(
            ConvNormAct(3, 32, 3, 1),
            ConvNormAct(32, 48, 3, 1),
            BlurPool(48, 2),
        )
        self.features = nn.Sequential(
            SelectiveKernelBlock(48, 64),
            SelectiveKernelBlock(64, 96, True),
            SelectiveKernelBlock(96, 96),
            SelectiveKernelBlock(96, 192, True),
            SelectiveKernelBlock(192, 192),
            SelectiveKernelBlock(192, 320, True),
            SelectiveKernelBlock(320, 320),
        )
        self.pool = GeMPool2d()
        self.classifier = nn.Sequential(nn.Dropout(0.3), nn.Linear(320, num_classes))
        self.apply(initialize_weights)

    def forward(self, x: Tensor) -> Tensor:
        return self.classifier(self.pool(self.features(self.stem(x))).flatten(1))
