"""Compact Res2Net with efficient channel attention and GeM pooling."""

from __future__ import annotations

import torch
from torch import Tensor, nn

from .common import (
    ConvNormAct,
    EfficientChannelAttention,
    GeMPool2d,
    conv1x1,
    initialize_weights,
)


class Res2Block(nn.Module):
    def __init__(
        self, in_channels: int, out_channels: int, stride: int = 1, scales: int = 4
    ) -> None:
        super().__init__()
        if out_channels % scales:
            raise ValueError("out_channels must be divisible by scales")
        width = out_channels // scales
        self.scales = scales
        self.reduce = ConvNormAct(in_channels, out_channels, 1, stride)
        self.convs = nn.ModuleList(
            [ConvNormAct(width, width, 3) for _ in range(scales - 1)]
        )
        self.fuse = nn.Sequential(
            conv1x1(out_channels, out_channels), nn.BatchNorm2d(out_channels)
        )
        self.attention = EfficientChannelAttention(out_channels)
        self.skip = (
            nn.Identity()
            if stride == 1 and in_channels == out_channels
            else nn.Sequential(
                conv1x1(in_channels, out_channels, stride), nn.BatchNorm2d(out_channels)
            )
        )
        self.activation = nn.ReLU(inplace=True)

    def forward(self, x: Tensor) -> Tensor:
        residual = self.skip(x)
        chunks = self.reduce(x).chunk(self.scales, dim=1)
        outputs = [chunks[0]]
        for index, conv in enumerate(self.convs, start=1):
            branch = chunks[index] if index == 1 else chunks[index] + outputs[-1]
            outputs.append(conv(branch))
        out = self.attention(self.fuse(torch.cat(outputs, dim=1)))
        return self.activation(out + residual)


class Res2NetBackbone(nn.Module):
    output_channels = 320

    def __init__(self) -> None:
        super().__init__()
        self.features = nn.Sequential(
            ConvNormAct(3, 32, 3, 2),
            ConvNormAct(32, 48, 3, 2),
            Res2Block(48, 80),
            Res2Block(80, 80),
            Res2Block(80, 160, 2),
            Res2Block(160, 160),
            Res2Block(160, 320, 2),
            Res2Block(320, 320),
        )
        self.pool = GeMPool2d()

    def forward(self, x: Tensor) -> Tensor:
        return self.pool(self.features(x)).flatten(1)


class Res2NetECAGeM(nn.Module):
    def __init__(self, num_classes: int = 18) -> None:
        super().__init__()
        self.backbone = Res2NetBackbone()
        self.classifier = nn.Sequential(nn.Dropout(0.3), nn.Linear(320, num_classes))
        self.apply(initialize_weights)

    def forward(self, x: Tensor) -> Tensor:
        return self.classifier(self.backbone(x))


class GeoAuxiliaryNet(nn.Module):
    """Image-only classifier with removable metadata-supervision heads.

    ``forward`` always requires only images.  Training code may call
    ``forward_with_aux`` to obtain predictions for provided metadata targets.
    """

    def __init__(self, num_classes: int = 18) -> None:
        super().__init__()
        self.backbone = Res2NetBackbone()
        self.embedding = nn.Sequential(nn.Linear(320, 256), nn.SiLU(), nn.Dropout(0.2))
        self.country_head = nn.Linear(256, num_classes)
        self.latitude_band_head = nn.Linear(256, 12)
        self.longitude_band_head = nn.Linear(256, 24)
        self.hemisphere_head = nn.Linear(256, 4)
        self.coordinate_head = nn.Linear(256, 3)
        self.apply(initialize_weights)

    def encode(self, x: Tensor) -> Tensor:
        return self.embedding(self.backbone(x))

    def forward(self, x: Tensor) -> Tensor:
        return self.country_head(self.encode(x))

    def forward_with_aux(self, x: Tensor) -> dict[str, Tensor]:
        features = self.encode(x)
        return {
            "country": self.country_head(features),
            "latitude_band": self.latitude_band_head(features),
            "longitude_band": self.longitude_band_head(features),
            "hemisphere": self.hemisphere_head(features),
            "coordinate": self.coordinate_head(features),
        }
