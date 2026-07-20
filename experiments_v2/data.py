"""Pre-cached split-preserving data pipeline."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch import Tensor
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F

from .augmentation import GeoAugmentV1
from .config import ExperimentConfig


@dataclass(frozen=True)
class DataBundle:
    train: DataLoader
    valid: DataLoader
    test: DataLoader
    countries: list[str]
    country_to_index: dict[str, int]
    mean: Tensor
    std: Tensor


def read_labels(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        required = {"filename", "country", "lat", "lng"}
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            raise ValueError(f"{path} must contain columns {sorted(required)}")
        return list(reader)


def coordinate_targets(latitude: float, longitude: float) -> dict[str, Tensor]:
    latitude_tensor = torch.tensor(latitude, dtype=torch.float32)
    longitude_tensor = torch.tensor(longitude, dtype=torch.float32)
    latitude_radians = torch.deg2rad(latitude_tensor)
    longitude_radians = torch.deg2rad(longitude_tensor)
    spherical = torch.stack(
        [
            torch.cos(latitude_radians) * torch.cos(longitude_radians),
            torch.cos(latitude_radians) * torch.sin(longitude_radians),
            torch.sin(latitude_radians),
        ]
    )
    latitude_band = min(11, max(0, int((latitude + 90.0) / 15.0)))
    longitude_band = min(23, max(0, int((longitude + 180.0) / 15.0)))
    hemisphere = (0 if latitude >= 0 else 2) + (0 if longitude >= 0 else 1)
    return {
        "latitude_band": torch.tensor(latitude_band, dtype=torch.long),
        "longitude_band": torch.tensor(longitude_band, dtype=torch.long),
        "hemisphere": torch.tensor(hemisphere, dtype=torch.long),
        "coordinate": spherical,
    }


class PreCachedGeoDataset(Dataset):
    def __init__(
        self,
        rows: list[dict[str, str]],
        image_dir: Path,
        country_to_index: dict[str, int],
        cache_size: int,
        output_size: int,
        training: bool,
        mean: Tensor | None = None,
        std: Tensor | None = None,
    ) -> None:
        self.training = training
        self.output_size = output_size
        self.augment = GeoAugmentV1(output_size) if training else None
        self.images: list[Tensor] = []
        self.labels: list[int] = []
        self.metadata: list[dict[str, Tensor]] = []
        self.mean = mean
        self.std = std

        for row in rows:
            image_path = image_dir / row["filename"]
            with Image.open(image_path) as source:
                source = source.convert("RGB").resize(
                    (cache_size, cache_size), Image.Resampling.BILINEAR
                )
                image = torch.from_numpy(
                    np.asarray(source, dtype=np.uint8).copy()
                ).permute(2, 0, 1)
            self.images.append(image)
            self.labels.append(country_to_index[row["country"]])
            self.metadata.append(
                coordinate_targets(float(row["lat"]), float(row["lng"]))
            )

    def set_normalization(self, mean: Tensor, std: Tensor) -> None:
        self.mean = mean.view(3, 1, 1)
        self.std = std.view(3, 1, 1)

    def channel_statistics(self) -> tuple[Tensor, Tensor]:
        channel_sum = torch.zeros(3, dtype=torch.float64)
        channel_square_sum = torch.zeros(3, dtype=torch.float64)
        pixel_count = 0
        for image in self.images:
            values = image.to(torch.float64) / 255.0
            channel_sum += values.sum(dim=(1, 2))
            channel_square_sum += values.square().sum(dim=(1, 2))
            pixel_count += values.shape[1] * values.shape[2]
        mean = channel_sum / pixel_count
        variance = channel_square_sum / pixel_count - mean.square()
        return mean.float(), variance.clamp_min(1e-8).sqrt().float()

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor, dict[str, Tensor]]:
        image = self.images[index].to(torch.float32) / 255.0
        if self.augment is not None:
            image = self.augment(image)
        elif image.shape[-1] != self.output_size:
            image = F.interpolate(
                image.unsqueeze(0),
                (self.output_size, self.output_size),
                mode="bilinear",
                align_corners=False,
                antialias=True,
            ).squeeze(0)
        if self.mean is None or self.std is None:
            raise RuntimeError(
                "dataset normalization must be configured before iteration"
            )
        image = (image - self.mean) / self.std
        return (
            image,
            torch.tensor(self.labels[index], dtype=torch.long),
            self.metadata[index],
        )


def create_dataloaders(config: ExperimentConfig) -> DataBundle:
    split_rows = {
        split: read_labels(config.data_root / split / "labels.csv")
        for split in ("train", "valid", "test")
    }
    countries = sorted({row["country"] for row in split_rows["train"]})
    country_to_index = {country: index for index, country in enumerate(countries)}
    for split, rows in split_rows.items():
        unknown = {row["country"] for row in rows}.difference(country_to_index)
        if unknown:
            raise ValueError(f"{split} contains unknown countries: {sorted(unknown)}")

    train_dataset = PreCachedGeoDataset(
        split_rows["train"],
        config.data_root / "train",
        country_to_index,
        config.train_cache_size,
        config.image_size,
        training=True,
    )
    mean, std = train_dataset.channel_statistics()
    train_dataset.set_normalization(mean, std)
    valid_dataset = PreCachedGeoDataset(
        split_rows["valid"],
        config.data_root / "valid",
        country_to_index,
        config.image_size,
        config.image_size,
        training=False,
        mean=mean.view(3, 1, 1),
        std=std.view(3, 1, 1),
    )
    test_dataset = PreCachedGeoDataset(
        split_rows["test"],
        config.data_root / "test",
        country_to_index,
        config.image_size,
        config.image_size,
        training=False,
        mean=mean.view(3, 1, 1),
        std=std.view(3, 1, 1),
    )

    generator = torch.Generator().manual_seed(config.seed)
    common = {
        "batch_size": config.batch_size,
        "num_workers": config.num_workers,
        "pin_memory": torch.cuda.is_available(),
    }
    return DataBundle(
        train=DataLoader(train_dataset, shuffle=True, generator=generator, **common),
        valid=DataLoader(valid_dataset, shuffle=False, **common),
        test=DataLoader(test_dataset, shuffle=False, **common),
        countries=countries,
        country_to_index=country_to_index,
        mean=mean,
        std=std,
    )
