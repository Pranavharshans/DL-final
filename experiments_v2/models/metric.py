"""Country-prototype classifier with an optional ArcFace training margin."""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from .common import ConvNormAct, GeMPool2d, ResidualBlock, initialize_weights


class ArcMarginHead(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        num_classes: int,
        scale: float = 24.0,
        margin: float = 0.25,
    ) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.empty(num_classes, embedding_dim))
        nn.init.xavier_uniform_(self.weight)
        self.scale = scale
        self.cos_margin = math.cos(margin)
        self.sin_margin = math.sin(margin)

    def forward(self, embedding: Tensor, labels: Tensor | None = None) -> Tensor:
        cosine = F.linear(F.normalize(embedding), F.normalize(self.weight)).clamp(
            -1.0, 1.0
        )
        if labels is None:
            return cosine * self.scale
        sine = torch.sqrt((1.0 - cosine.square()).clamp_min(1e-7))
        margin_cosine = cosine * self.cos_margin - sine * self.sin_margin
        one_hot = F.one_hot(labels, num_classes=cosine.shape[1]).to(cosine.dtype)
        return (one_hot * margin_cosine + (1.0 - one_hot) * cosine) * self.scale


class CountryMetricNet(nn.Module):
    def __init__(self, num_classes: int = 18) -> None:
        super().__init__()
        self.features = nn.Sequential(
            ConvNormAct(3, 32, 3, 2),
            ResidualBlock(32, 64, 2),
            ResidualBlock(64, 64),
            ResidualBlock(64, 128, 2),
            ResidualBlock(128, 128),
            ResidualBlock(128, 256, 2),
            ResidualBlock(256, 256),
            ResidualBlock(256, 320, 2),
        )
        self.pool = GeMPool2d()
        self.embedding = nn.Sequential(
            nn.Linear(320, 256),
            nn.BatchNorm1d(256),
            nn.PReLU(256),
            nn.Dropout(0.2),
        )
        self.head = ArcMarginHead(256, num_classes)
        self.apply(initialize_weights)

    def encode(self, x: Tensor) -> Tensor:
        return self.embedding(self.pool(self.features(x)).flatten(1))

    def forward(self, x: Tensor, labels: Tensor | None = None) -> Tensor:
        return self.head(self.encode(x), labels)
