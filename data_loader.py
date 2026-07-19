"""Pre-cached data loader — loads all images into RAM once, then feeds GPU at full speed."""
import csv
import os

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset


def read_labels_csv(csv_path):
    with open(csv_path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


class PreCachedDataset(Dataset):
    """Loads, resizes, and caches all images as float32 tensors at init time."""

    def __init__(self, rows, image_dir, country_to_idx, image_size=256):
        self.country_to_idx = country_to_idx
        self.image_size = image_size
        self.samples = []

        n = len(rows)
        print(f"  Pre-caching {n} images at {image_size}x{image_size} into RAM...", flush=True)
        for i, row in enumerate(rows):
            if (i + 1) % 1000 == 0:
                print(f"    {i+1}/{n}...", flush=True)
            img = Image.open(os.path.join(image_dir, row["filename"])).convert("RGB")
            img = img.resize((image_size, image_size), Image.BILINEAR)
            tensor = torch.from_numpy(np.array(img, dtype=np.float32) / 255.0).permute(2, 0, 1)
            label = country_to_idx[row["country"]]
            lat = float(row.get("lat", 0) or 0)
            lng = float(row.get("lng", 0) or 0)
            self.samples.append((tensor, label, torch.tensor([lat, lng], dtype=torch.float32)))
        print(f"  Done: {n} images cached.", flush=True)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


def create_dataloaders(data_root, image_size=256, batch_size=32, num_workers=2):
    train_rows = read_labels_csv(os.path.join(data_root, "train", "labels.csv"))
    valid_rows = read_labels_csv(os.path.join(data_root, "valid", "labels.csv"))
    test_rows  = read_labels_csv(os.path.join(data_root, "test", "labels.csv"))
    countries = sorted(set(row["country"] for row in train_rows))
    country_to_idx = {c: i for i, c in enumerate(countries)}

    print(f"Countries: {len(countries)}, Train: {len(train_rows)}, Valid: {len(valid_rows)}, Test: {len(test_rows)}", flush=True)

    train_ds = PreCachedDataset(train_rows, os.path.join(data_root, "train"), country_to_idx, image_size)
    valid_ds = PreCachedDataset(valid_rows, os.path.join(data_root, "valid"), country_to_idx, image_size)
    test_ds  = PreCachedDataset(test_rows,  os.path.join(data_root, "test"),  country_to_idx, image_size)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=num_workers, pin_memory=True, persistent_workers=True)
    valid_loader = DataLoader(valid_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True, persistent_workers=True)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True, persistent_workers=True)

    return train_loader, valid_loader, test_loader, countries, country_to_idx
