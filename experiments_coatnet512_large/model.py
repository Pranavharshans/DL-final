"""A 4.25M-parameter CoAtNet variant trained entirely from scratch."""

from __future__ import annotations

from torch import Tensor, nn

from experiments_v2.models.coatnet import SpatialAttentionBlock
from experiments_v2.models.common import (
    ConvNormAct,
    GeMPool2d,
    MBConv,
    initialize_weights,
)


class CoAtNetLarge4M(nn.Module):
    """Wider/deeper CoAtNet that remains comfortably below five million params."""

    def __init__(self, num_classes: int = 18) -> None:
        super().__init__()
        self.stem = nn.Sequential(
            ConvNormAct(3, 48, 3, 2),
            ConvNormAct(48, 72, 3, 2),
        )
        self.convolutions = nn.Sequential(
            MBConv(72, 88, expansion=2),
            MBConv(88, 136, stride=2, expansion=4),
            MBConv(136, 136, expansion=4, drop_path=0.03),
            MBConv(136, 208, stride=2, expansion=4, drop_path=0.05),
            MBConv(208, 208, expansion=4, drop_path=0.07),
            ConvNormAct(208, 288, 3, 2),
        )
        self.attention = nn.Sequential(
            SpatialAttentionBlock(288, heads=6, drop_path=0.09),
            SpatialAttentionBlock(288, heads=6, drop_path=0.12),
            SpatialAttentionBlock(288, heads=6, drop_path=0.15),
        )
        self.pool = GeMPool2d()
        self.classifier = nn.Sequential(
            nn.LayerNorm(288),
            nn.Dropout(0.35),
            nn.Linear(288, num_classes),
        )
        self.apply(initialize_weights)

    def forward(self, x: Tensor) -> Tensor:
        features = self.attention(self.convolutions(self.stem(x)))
        return self.classifier(self.pool(features).flatten(1))
