"""Micro MaxViT using convolution, window attention, and grid attention."""

from __future__ import annotations

from torch import Tensor, nn

from .common import ConvNormAct, GeMPool2d, MBConv, DropPath, initialize_weights


def _window_partition(
    x: Tensor, block_size: int
) -> tuple[Tensor, tuple[int, int, int, int]]:
    batch, channels, height, width = x.shape
    if height % block_size or width % block_size:
        raise ValueError(
            f"feature map {(height, width)} must be divisible by block size {block_size}"
        )
    windows = x.view(
        batch,
        channels,
        height // block_size,
        block_size,
        width // block_size,
        block_size,
    )
    windows = windows.permute(0, 2, 4, 3, 5, 1).reshape(
        -1, block_size * block_size, channels
    )
    return windows, (batch, channels, height, width)


def _window_reverse(
    windows: Tensor, shape: tuple[int, int, int, int], block_size: int
) -> Tensor:
    batch, channels, height, width = shape
    x = windows.view(
        batch,
        height // block_size,
        width // block_size,
        block_size,
        block_size,
        channels,
    )
    return x.permute(0, 5, 1, 3, 2, 4).reshape(batch, channels, height, width)


def _grid_partition(
    x: Tensor, grid_size: int
) -> tuple[Tensor, tuple[int, int, int, int]]:
    batch, channels, height, width = x.shape
    if height % grid_size or width % grid_size:
        raise ValueError(
            f"feature map {(height, width)} must be divisible by grid size {grid_size}"
        )
    cell_height, cell_width = height // grid_size, width // grid_size
    grids = x.view(batch, channels, grid_size, cell_height, grid_size, cell_width)
    grids = grids.permute(0, 3, 5, 2, 4, 1).reshape(-1, grid_size * grid_size, channels)
    return grids, (batch, channels, height, width)


def _grid_reverse(
    grids: Tensor, shape: tuple[int, int, int, int], grid_size: int
) -> Tensor:
    batch, channels, height, width = shape
    cell_height, cell_width = height // grid_size, width // grid_size
    x = grids.view(batch, cell_height, cell_width, grid_size, grid_size, channels)
    return x.permute(0, 5, 3, 1, 4, 2).reshape(batch, channels, height, width)


class PartitionAttention(nn.Module):
    def __init__(
        self,
        channels: int,
        heads: int,
        partition_size: int,
        mode: str,
        drop_path: float,
    ) -> None:
        super().__init__()
        if mode not in {"window", "grid"}:
            raise ValueError(f"unsupported attention mode: {mode}")
        self.channels = channels
        self.heads = heads
        self.head_dim = channels // heads
        self.scale = self.head_dim**-0.5
        self.partition_size = partition_size
        self.mode = mode
        self.norm1 = nn.LayerNorm(channels)
        self.qkv = nn.Linear(channels, channels * 3)
        self.projection = nn.Linear(channels, channels)
        self.norm2 = nn.LayerNorm(channels)
        self.mlp = nn.Sequential(
            nn.Linear(channels, channels * 3),
            nn.GELU(),
            nn.Linear(channels * 3, channels),
        )
        self.drop_path = DropPath(drop_path)

    def forward(self, x: Tensor) -> Tensor:
        partition = _window_partition if self.mode == "window" else _grid_partition
        reverse = _window_reverse if self.mode == "window" else _grid_reverse
        tokens, shape = partition(x, self.partition_size)
        normalized = self.norm1(tokens)
        qkv = self.qkv(normalized).reshape(
            tokens.shape[0], tokens.shape[1], 3, self.heads, self.head_dim
        )
        query, key, value = qkv.permute(2, 0, 3, 1, 4).unbind(0)
        attention = (query @ key.transpose(-2, -1) * self.scale).softmax(dim=-1)
        attended = (attention @ value).transpose(1, 2).reshape_as(tokens)
        tokens = tokens + self.drop_path(self.projection(attended))
        tokens = tokens + self.drop_path(self.mlp(self.norm2(tokens)))
        return reverse(tokens, shape, self.partition_size)


class MaxViTBlock(nn.Module):
    def __init__(
        self, channels: int, partition_size: int = 4, drop_path: float = 0.0
    ) -> None:
        super().__init__()
        self.mbconv = MBConv(channels, channels, expansion=3, drop_path=drop_path)
        self.window = PartitionAttention(
            channels, 4, partition_size, "window", drop_path
        )
        self.grid = PartitionAttention(channels, 4, partition_size, "grid", drop_path)

    def forward(self, x: Tensor) -> Tensor:
        return self.grid(self.window(self.mbconv(x)))


class MaxViTMicro(nn.Module):
    def __init__(self, num_classes: int = 18) -> None:
        super().__init__()
        self.stem = nn.Sequential(
            ConvNormAct(3, 32, 3, 2),
            ConvNormAct(32, 48, 3, 2),
        )
        self.convolutions = nn.Sequential(
            MBConv(48, 80, stride=2, expansion=3),
            MBConv(80, 80, expansion=3, drop_path=0.02),
            MBConv(80, 160, stride=2, expansion=4, drop_path=0.04),
            MBConv(160, 160, expansion=4, drop_path=0.06),
            ConvNormAct(160, 224, 3, 2),
        )
        self.blocks = nn.Sequential(
            MaxViTBlock(224, partition_size=4, drop_path=0.08),
            MaxViTBlock(224, partition_size=4, drop_path=0.10),
        )
        self.pool = GeMPool2d()
        self.classifier = nn.Sequential(
            nn.LayerNorm(224), nn.Dropout(0.3), nn.Linear(224, num_classes)
        )
        self.apply(initialize_weights)

    def forward(self, x: Tensor) -> Tensor:
        return self.classifier(
            self.pool(self.blocks(self.convolutions(self.stem(x)))).flatten(1)
        )
