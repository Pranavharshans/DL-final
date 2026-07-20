"""Lightweight Inception-ResNet with multi-scale residual branches."""

from __future__ import annotations

import torch
from torch import Tensor, nn

from .common import ConvNormAct, GeMPool2d, initialize_weights


class InceptionResidualBlock(nn.Module):
    def __init__(self, channels: int, branch_channels: int, scale: float = 0.2) -> None:
        super().__init__()
        self.scale = scale
        self.branch1 = ConvNormAct(channels, branch_channels, 1)
        self.branch3 = nn.Sequential(
            ConvNormAct(channels, branch_channels, 1),
            ConvNormAct(branch_channels, branch_channels, 3),
        )
        self.branch5 = nn.Sequential(
            ConvNormAct(channels, branch_channels, 1),
            ConvNormAct(branch_channels, branch_channels, 3),
            ConvNormAct(branch_channels, branch_channels, 3),
        )
        self.project = nn.Sequential(
            nn.Conv2d(branch_channels * 3, channels, 1, bias=False),
            nn.BatchNorm2d(channels),
        )
        self.activation = nn.ReLU(inplace=True)

    def forward(self, x: Tensor) -> Tensor:
        residual = self.project(
            torch.cat([self.branch1(x), self.branch3(x), self.branch5(x)], dim=1)
        )
        return self.activation(x + residual * self.scale)


class ReductionBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        half = out_channels // 2
        self.conv = ConvNormAct(in_channels, half, 3, 2)
        self.path = nn.Sequential(
            ConvNormAct(in_channels, half // 2, 1),
            ConvNormAct(half // 2, half // 2, 3),
            ConvNormAct(half // 2, half, 3, 2),
        )

    def forward(self, x: Tensor) -> Tensor:
        return torch.cat([self.conv(x), self.path(x)], dim=1)


class InceptionResNetLite(nn.Module):
    def __init__(self, num_classes: int = 18) -> None:
        super().__init__()
        self.stem = nn.Sequential(
            ConvNormAct(3, 32, 3, 2),
            ConvNormAct(32, 48, 3, 2),
            ConvNormAct(48, 64, 3),
        )
        self.stage1 = nn.Sequential(*[InceptionResidualBlock(64, 24) for _ in range(3)])
        self.reduction1 = ReductionBlock(64, 160)
        self.stage2 = nn.Sequential(
            *[InceptionResidualBlock(160, 48) for _ in range(4)]
        )
        self.reduction2 = ReductionBlock(160, 320)
        self.stage3 = nn.Sequential(
            *[InceptionResidualBlock(320, 72) for _ in range(3)]
        )
        self.pool = GeMPool2d()
        self.classifier = nn.Sequential(nn.Dropout(0.35), nn.Linear(320, num_classes))
        self.apply(initialize_weights)

    def forward(self, x: Tensor) -> Tensor:
        x = self.stage1(self.stem(x))
        x = self.stage2(self.reduction1(x))
        x = self.stage3(self.reduction2(x))
        return self.classifier(self.pool(x).flatten(1))
