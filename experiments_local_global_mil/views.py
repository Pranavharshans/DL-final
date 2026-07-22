"""Construct one global view and four zoomed local views from each image."""

from __future__ import annotations

import torch
from torch import Tensor
import torch.nn.functional as F
from torch.utils.data import Dataset


def _resize(image: Tensor, size: int) -> Tensor:
    return F.interpolate(
        image.unsqueeze(0),
        (size, size),
        mode="bilinear",
        align_corners=False,
        antialias=True,
    ).squeeze(0)


class FiveViewDataset(Dataset):
    """Wrap a normalized image dataset with global and quadrant-local views."""

    def __init__(self, dataset: Dataset, view_size: int = 256, training: bool = False):
        self.dataset = dataset
        self.view_size = view_size
        self.training = training

    def __len__(self) -> int:
        return len(self.dataset)

    def _local_crop(self, image: Tensor, quadrant: int) -> Tensor:
        _, height, width = image.shape
        if self.training:
            fraction = torch.empty(()).uniform_(0.56, 0.72).item()
            crop_h = max(self.view_size, round(height * fraction))
            crop_w = max(self.view_size, round(width * fraction))
            row, column = divmod(quadrant, 2)
            y_anchor = 0 if row == 0 else height - crop_h
            x_anchor = 0 if column == 0 else width - crop_w
            jitter_y = max(1, round((height - crop_h) * 0.30))
            jitter_x = max(1, round((width - crop_w) * 0.30))
            y0 = min(
                max(
                    0, y_anchor + int(torch.randint(-jitter_y, jitter_y + 1, ()).item())
                ),
                height - crop_h,
            )
            x0 = min(
                max(
                    0, x_anchor + int(torch.randint(-jitter_x, jitter_x + 1, ()).item())
                ),
                width - crop_w,
            )
        else:
            crop_h = max(self.view_size, round(height * 0.625))
            crop_w = max(self.view_size, round(width * 0.625))
            row, column = divmod(quadrant, 2)
            y0 = 0 if row == 0 else height - crop_h
            x0 = 0 if column == 0 else width - crop_w
        return _resize(image[:, y0 : y0 + crop_h, x0 : x0 + crop_w], self.view_size)

    def __getitem__(self, index: int):
        image, label, metadata = self.dataset[index]
        views = [_resize(image, self.view_size)]
        views.extend(self._local_crop(image, quadrant) for quadrant in range(4))
        return torch.stack(views), label, metadata
