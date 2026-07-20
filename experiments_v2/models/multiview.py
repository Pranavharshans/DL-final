"""Shared-backbone global plus local crop classifier."""

from __future__ import annotations

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from .common import ConvNormAct, GeMPool2d, ResidualBlock, initialize_weights


class ViewBackbone(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.features = nn.Sequential(
            ConvNormAct(3, 32, 3, 2),
            ResidualBlock(32, 48, 2),
            ResidualBlock(48, 48),
            ResidualBlock(48, 96, 2),
            ResidualBlock(96, 96),
            ResidualBlock(96, 160, 2),
            ResidualBlock(160, 160),
            ResidualBlock(160, 256, 2),
            ResidualBlock(256, 256),
        )
        self.pool = GeMPool2d()

    def forward(self, x: Tensor) -> Tensor:
        return self.pool(self.features(x)).flatten(1)


class MultiViewCNN(nn.Module):
    """Extract one global view and four detail views from a single input image."""

    def __init__(self, num_classes: int = 18) -> None:
        super().__init__()
        self.backbone = ViewBackbone()
        self.view_attention = nn.Sequential(
            nn.Linear(256, 96),
            nn.Tanh(),
            nn.Linear(96, 1),
        )
        self.classifier = nn.Sequential(
            nn.LayerNorm(256),
            nn.Dropout(0.25),
            nn.Linear(256, num_classes),
        )
        self.apply(initialize_weights)

    @staticmethod
    def _views(x: Tensor) -> Tensor:
        batch, _, height, width = x.shape
        global_view = F.interpolate(
            x, size=(128, 128), mode="bilinear", align_corners=False
        )
        crops = [
            x[:, :, : height // 2, : width // 2],
            x[:, :, : height // 2, width // 2 :],
            x[:, :, height // 2 :, : width // 2],
            x[:, :, height // 2 :, width // 2 :],
        ]
        crops = [
            F.interpolate(crop, (128, 128), mode="bilinear", align_corners=False)
            for crop in crops
        ]
        return torch.cat([global_view, *crops], dim=0).view(5, batch, 3, 128, 128)

    def forward(self, x: Tensor) -> Tensor:
        views = self._views(x)
        features = torch.stack([self.backbone(view) for view in views], dim=1)
        weights = self.view_attention(features).softmax(dim=1)
        fused = (features * weights).sum(dim=1)
        return self.classifier(fused)
