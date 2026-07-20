"""Dual-branch RGB and fixed Haar-frequency classifier."""

from __future__ import annotations

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from .common import ConvNormAct, GeMPool2d, ResidualBlock, initialize_weights


class HaarTransform(nn.Module):
    """Non-learned 2D Haar transform applied independently to RGB channels."""

    def __init__(self) -> None:
        super().__init__()
        kernels = (
            torch.tensor(
                [
                    [[1.0, 1.0], [1.0, 1.0]],
                    [[-1.0, -1.0], [1.0, 1.0]],
                    [[-1.0, 1.0], [-1.0, 1.0]],
                    [[1.0, -1.0], [-1.0, 1.0]],
                ]
            )
            / 2.0
        )
        weight = kernels[:, None].repeat(3, 1, 1, 1)
        self.register_buffer("weight", weight)

    def forward(self, x: Tensor) -> Tensor:
        return F.conv2d(x, self.weight, stride=2, groups=3)


class SpatialFrequencyNet(nn.Module):
    def __init__(self, num_classes: int = 18) -> None:
        super().__init__()
        self.spatial = nn.Sequential(
            ConvNormAct(3, 32, 3, 2),
            ResidualBlock(32, 64, 2),
            ResidualBlock(64, 96, 2),
            ResidualBlock(96, 160, 2),
            ResidualBlock(160, 224, 2),
        )
        self.haar = HaarTransform()
        self.frequency = nn.Sequential(
            ConvNormAct(12, 32, 3, 2),
            ResidualBlock(32, 48, 2),
            ResidualBlock(48, 80, 2),
            ResidualBlock(80, 128, 2),
        )
        self.fusion = nn.Sequential(
            ConvNormAct(352, 320, 1),
            ResidualBlock(320, 320),
        )
        self.pool = GeMPool2d()
        self.classifier = nn.Sequential(nn.Dropout(0.3), nn.Linear(320, num_classes))
        self.apply(initialize_weights)

    def forward(self, x: Tensor) -> Tensor:
        spatial = self.spatial(x)
        frequency = self.frequency(self.haar(x))
        frequency = F.interpolate(
            frequency, spatial.shape[-2:], mode="bilinear", align_corners=False
        )
        features = self.fusion(torch.cat([spatial, frequency], dim=1))
        return self.classifier(self.pool(features).flatten(1))
