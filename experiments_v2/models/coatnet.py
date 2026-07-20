"""Convolution-attention hybrid sized for training from scratch."""

from __future__ import annotations

from torch import Tensor, nn

from .common import ConvNormAct, GeMPool2d, MBConv, DropPath, initialize_weights


class SpatialAttentionBlock(nn.Module):
    def __init__(self, channels: int, heads: int = 4, drop_path: float = 0.0) -> None:
        super().__init__()
        if channels % heads:
            raise ValueError("channels must be divisible by heads")
        self.channels = channels
        self.heads = heads
        self.head_dim = channels // heads
        self.scale = self.head_dim**-0.5
        self.norm1 = nn.LayerNorm(channels)
        self.qkv = nn.Linear(channels, channels * 3)
        self.projection = nn.Linear(channels, channels)
        self.norm2 = nn.LayerNorm(channels)
        self.mlp = nn.Sequential(
            nn.Linear(channels, channels * 3),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(channels * 3, channels),
        )
        self.drop_path = DropPath(drop_path)

    def forward(self, x: Tensor) -> Tensor:
        batch, channels, height, width = x.shape
        tokens = x.flatten(2).transpose(1, 2)
        normalized = self.norm1(tokens)
        qkv = self.qkv(normalized).reshape(
            batch, height * width, 3, self.heads, self.head_dim
        )
        qkv = qkv.permute(2, 0, 3, 1, 4)
        query, key, value = qkv.unbind(0)
        attention = (query @ key.transpose(-2, -1) * self.scale).softmax(dim=-1)
        attended = (
            (attention @ value).transpose(1, 2).reshape(batch, height * width, channels)
        )
        tokens = tokens + self.drop_path(self.projection(attended))
        tokens = tokens + self.drop_path(self.mlp(self.norm2(tokens)))
        return tokens.transpose(1, 2).reshape(batch, channels, height, width)


class CoAtNetMicro(nn.Module):
    def __init__(self, num_classes: int = 18) -> None:
        super().__init__()
        self.stem = nn.Sequential(
            ConvNormAct(3, 32, 3, 2),
            ConvNormAct(32, 48, 3, 2),
        )
        self.convolutions = nn.Sequential(
            MBConv(48, 64, expansion=2),
            MBConv(64, 96, stride=2, expansion=4),
            MBConv(96, 96, expansion=4, drop_path=0.02),
            MBConv(96, 160, stride=2, expansion=4, drop_path=0.04),
            MBConv(160, 160, expansion=4, drop_path=0.06),
            ConvNormAct(160, 224, 3, 2),
        )
        self.attention = nn.Sequential(
            SpatialAttentionBlock(224, heads=4, drop_path=0.08),
            SpatialAttentionBlock(224, heads=4, drop_path=0.10),
        )
        self.pool = GeMPool2d()
        self.classifier = nn.Sequential(
            nn.LayerNorm(224), nn.Dropout(0.3), nn.Linear(224, num_classes)
        )
        self.apply(initialize_weights)

    def forward(self, x: Tensor) -> Tensor:
        features = self.attention(self.convolutions(self.stem(x)))
        return self.classifier(self.pool(features).flatten(1))
