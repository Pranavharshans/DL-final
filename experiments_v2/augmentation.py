"""Conservative online augmentation for street-level geolocation."""

from __future__ import annotations

import math

import torch
from torch import Tensor
import torch.nn.functional as F


class GeoAugmentV1:
    """Augment without mirroring or rotating geographic evidence."""

    def __init__(self, output_size: int = 256) -> None:
        self.output_size = output_size

    @staticmethod
    def _uniform(low: float, high: float) -> float:
        return torch.empty(1).uniform_(low, high).item()

    def _random_resized_crop(self, image: Tensor) -> Tensor:
        _, height, width = image.shape
        area = height * width
        for _ in range(10):
            target_area = area * self._uniform(0.78, 1.0)
            aspect = self._uniform(0.90, 1.10)
            crop_width = round(math.sqrt(target_area * aspect))
            crop_height = round(math.sqrt(target_area / aspect))
            if crop_height <= height and crop_width <= width:
                top = int(torch.randint(0, height - crop_height + 1, ()).item())
                left = int(torch.randint(0, width - crop_width + 1, ()).item())
                crop = image[:, top : top + crop_height, left : left + crop_width]
                return F.interpolate(
                    crop.unsqueeze(0),
                    (self.output_size, self.output_size),
                    mode="bilinear",
                    align_corners=False,
                    antialias=True,
                ).squeeze(0)
        return F.interpolate(
            image.unsqueeze(0),
            (self.output_size, self.output_size),
            mode="bilinear",
            align_corners=False,
            antialias=True,
        ).squeeze(0)

    def _color_jitter(self, image: Tensor) -> Tensor:
        if torch.rand(()) >= 0.8:
            return image
        brightness = self._uniform(0.82, 1.18)
        contrast = self._uniform(0.82, 1.18)
        saturation = self._uniform(0.88, 1.12)
        image = image * brightness
        channel_mean = image.mean(dim=(-2, -1), keepdim=True)
        image = (image - channel_mean) * contrast + channel_mean
        gray = image.mean(dim=0, keepdim=True)
        image = (image - gray) * saturation + gray
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
    def _random_erasing(image: Tensor) -> Tensor:
        if torch.rand(()) >= 0.1:
            return image
        _, height, width = image.shape
        erase_height = max(1, int(height * torch.empty(1).uniform_(0.05, 0.14).item()))
        erase_width = max(1, int(width * torch.empty(1).uniform_(0.05, 0.14).item()))
        top = int(torch.randint(0, height - erase_height + 1, ()).item())
        left = int(torch.randint(0, width - erase_width + 1, ()).item())
        image = image.clone()
        image[:, top : top + erase_height, left : left + erase_width] = image.mean(
            dim=(-2, -1), keepdim=True
        )
        return image

    def __call__(self, image: Tensor) -> Tensor:
        image = self._random_resized_crop(image)
        image = self._color_jitter(image)
        if torch.rand(()) < 0.08:
            image = self._blur(image)
        return self._random_erasing(image)
