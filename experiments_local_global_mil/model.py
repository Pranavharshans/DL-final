"""Under-5M local-global multi-instance network trained from scratch."""

from __future__ import annotations

import torch
from torch import Tensor, nn

from experiments_coatnet512_large.model import CoAtNetLarge4M
from experiments_v2.models.common import initialize_weights


class LocalGlobalMIL(nn.Module):
    """Encode five image views with one shared backbone and learn their relevance."""

    def __init__(self, num_classes: int = 18, num_views: int = 5) -> None:
        super().__init__()
        base = CoAtNetLarge4M(num_classes)
        self.stem = base.stem
        self.convolutions = base.convolutions
        self.attention = base.attention
        self.pool = base.pool
        self.num_views = num_views
        width = 288

        self.view_embedding = nn.Parameter(torch.zeros(1, num_views, width))
        self.view_norm = nn.LayerNorm(width)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=width,
            nhead=6,
            dim_feedforward=384,
            dropout=0.12,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.fusion = nn.TransformerEncoder(encoder_layer, num_layers=1)
        self.importance = nn.Sequential(
            nn.LayerNorm(width),
            nn.Linear(width, 96),
            nn.GELU(),
            nn.Linear(96, 1),
        )
        self.classifier = nn.Sequential(
            nn.LayerNorm(width * 2),
            nn.Dropout(0.30),
            nn.Linear(width * 2, num_classes),
        )
        self.fusion.apply(initialize_weights)
        self.importance.apply(initialize_weights)
        self.classifier.apply(initialize_weights)
        nn.init.trunc_normal_(self.view_embedding, std=0.02)

    def encode(self, images: Tensor) -> Tensor:
        features = self.attention(self.convolutions(self.stem(images)))
        return self.pool(features).flatten(1)

    def forward(self, views: Tensor) -> Tensor:
        if views.ndim != 5 or views.shape[1] != self.num_views:
            raise ValueError(
                f"expected [batch, {self.num_views}, channels, height, width], "
                f"got {tuple(views.shape)}"
            )
        batch, count, channels, height, width = views.shape
        encoded = self.encode(views.reshape(batch * count, channels, height, width))
        encoded = encoded.reshape(batch, count, -1)
        contextual = self.fusion(self.view_norm(encoded + self.view_embedding))
        weights = torch.softmax(self.importance(contextual), dim=1)
        local_summary = (contextual * weights).sum(dim=1)
        global_feature = contextual[:, 0]
        return self.classifier(torch.cat((global_feature, local_summary), dim=1))
