"""Conservative image-only augmentation for geographic classification."""

from __future__ import annotations

import math

import torch
from torch import Tensor
import torch.nn.functional as F


class SafeGeoAugment512:
    """Preserve directional/geographic evidence while adding mild robustness."""

    def __init__(self, output_size: int = 512) -> None:
        self.output_size = output_size

    @staticmethod
    def _uniform(low: float, high: float) -> float:
        return torch.empty(1).uniform_(low, high).item()

    def _crop(self, image: Tensor) -> Tensor:
        _, height, width = image.shape
        area = height * width
        for _ in range(10):
            target_area = area * self._uniform(0.90, 1.0)
            aspect = self._uniform(0.94, 1.06)
            crop_width = round(math.sqrt(target_area * aspect))
            crop_height = round(math.sqrt(target_area / aspect))
            if crop_height <= height and crop_width <= width:
                top = int(torch.randint(0, height - crop_height + 1, ()).item())
                left = int(torch.randint(0, width - crop_width + 1, ()).item())
                image = image[:, top : top + crop_height, left : left + crop_width]
                break
        return F.interpolate(
            image.unsqueeze(0),
            (self.output_size, self.output_size),
            mode="bilinear",
            align_corners=False,
            antialias=True,
        ).squeeze(0)

    def _color(self, image: Tensor) -> Tensor:
        if torch.rand(()) >= 0.75:
            return image
        image = image * self._uniform(0.86, 1.14)
        channel_mean = image.mean(dim=(-2, -1), keepdim=True)
        image = (image - channel_mean) * self._uniform(0.86, 1.14) + channel_mean
        gray = image.mean(dim=0, keepdim=True)
        image = (image - gray) * self._uniform(0.92, 1.08) + gray
        return image.clamp(0.0, 1.0)

    @staticmethod
    def _blur(image: Tensor) -> Tensor:
        kernel_1d = image.new_tensor([1.0, 4.0, 6.0, 4.0, 1.0]) / 16.0
        kernel = (kernel_1d[:, None] * kernel_1d[None, :]).view(1, 1, 5, 5)
        kernel = kernel.repeat(image.shape[0], 1, 1, 1)
        return F.conv2d(
            image.unsqueeze(0), kernel, padding=2, groups=image.shape[0]
        ).squeeze(0)

    @staticmethod
    def _erase(image: Tensor) -> Tensor:
        if torch.rand(()) >= 0.08:
            return image
        _, height, width = image.shape
        erase_height = max(1, int(height * torch.empty(1).uniform_(0.04, 0.10).item()))
        erase_width = max(1, int(width * torch.empty(1).uniform_(0.04, 0.10).item()))
        top = int(torch.randint(0, height - erase_height + 1, ()).item())
        left = int(torch.randint(0, width - erase_width + 1, ()).item())
        image = image.clone()
        image[:, top : top + erase_height, left : left + erase_width] = image.mean(
            dim=(-2, -1), keepdim=True
        )
        return image

    def __call__(self, image: Tensor) -> Tensor:
        image = self._color(self._crop(image))
        if torch.rand(()) < 0.05:
            image = self._blur(image)
        return self._erase(image)
