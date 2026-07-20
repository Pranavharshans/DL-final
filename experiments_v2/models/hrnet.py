"""A compact high-resolution network for small geographic cues."""

from __future__ import annotations

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from .common import ConvNormAct, GeMPool2d, ResidualBlock, conv1x1, initialize_weights


class TwoResolutionFusion(nn.Module):
    def __init__(self, high_channels: int, low_channels: int) -> None:
        super().__init__()
        self.high_blocks = nn.Sequential(
            ResidualBlock(high_channels, high_channels),
            ResidualBlock(high_channels, high_channels),
        )
        self.low_blocks = nn.Sequential(
            ResidualBlock(low_channels, low_channels),
            ResidualBlock(low_channels, low_channels),
        )
        self.high_to_low = ConvNormAct(high_channels, low_channels, 3, 2)
        self.low_to_high = nn.Sequential(
            conv1x1(low_channels, high_channels), nn.BatchNorm2d(high_channels)
        )
        self.high_activation = nn.ReLU(inplace=True)
        self.low_activation = nn.ReLU(inplace=True)

    def forward(self, high: Tensor, low: Tensor) -> tuple[Tensor, Tensor]:
        high_features = self.high_blocks(high)
        low_features = self.low_blocks(low)
        low_up = F.interpolate(
            self.low_to_high(low_features),
            size=high_features.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        high_down = self.high_to_low(high_features)
        return self.high_activation(high_features + low_up), self.low_activation(
            low_features + high_down
        )


class HRNetLite(nn.Module):
    def __init__(self, num_classes: int = 18) -> None:
        super().__init__()
        self.stem = nn.Sequential(ConvNormAct(3, 32, 3, 2), ConvNormAct(32, 48, 3, 2))
        self.high_start = ResidualBlock(48, 64)
        self.low_start = ConvNormAct(64, 128, 3, 2)
        self.fusions = nn.ModuleList([TwoResolutionFusion(64, 128) for _ in range(3)])
        self.high_projection = ConvNormAct(64, 128, 3, 2)
        self.low_projection = ConvNormAct(128, 256, 3, 2)
        self.fusion = nn.Sequential(
            ConvNormAct(384, 320, 1),
            ResidualBlock(320, 320),
        )
        self.pool = GeMPool2d()
        self.classifier = nn.Sequential(nn.Dropout(0.3), nn.Linear(320, num_classes))
        self.apply(initialize_weights)

    def forward(self, x: Tensor) -> Tensor:
        high = self.high_start(self.stem(x))
        low = self.low_start(high)
        for fusion in self.fusions:
            high, low = fusion(high, low)
        high = self.high_projection(high)
        low = self.low_projection(low)
        high = F.interpolate(
            high, size=low.shape[-2:], mode="bilinear", align_corners=False
        )
        features = self.fusion(torch.cat([high, low], dim=1))
        return self.classifier(self.pool(features).flatten(1))
