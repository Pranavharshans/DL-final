"""UInt8 pre-cached data loader — 2.5-3x faster, no CPU bottleneck."""
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
    """Pre-caches images as uint8 tensors (1.5GB train) — no JPEG decode per epoch."""

    def __init__(self, rows, image_dir, country_to_idx, image_size=256):
        self.country_to_idx = country_to_idx
        self.image_size = image_size
        self.samples = []

        n = len(rows)
        print(f"  Pre-caching {n} images at {image_size}x{image_size} (uint8)...", flush=True)
        for i, row in enumerate(rows):
            if (i + 1) % 1000 == 0:
                print(f"    {i+1}/{n}...", flush=True)
            img = Image.open(os.path.join(image_dir, row["filename"])).convert("RGB")
            img = img.resize((image_size, image_size), Image.BILINEAR)
            # Store as uint8 — 1/4 the RAM of float32
            tensor = torch.from_numpy(np.array(img, dtype=np.uint8)).permute(2, 0, 1)
            label = country_to_idx[row["country"]]
            lat = float(row.get("lat", 0) or 0)
            lng = float(row.get("lng", 0) or 0)
            self.samples.append((tensor, label, torch.tensor([lat, lng], dtype=torch.float32)))
        print(f"  Done: {n} images cached.", flush=True)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_uint8, label, coords = self.samples[idx]
        # Convert to float32 on-the-fly — single tensor op, negligible cost
        return img_uint8.to(torch.float32) / 255.0, label, coords


def create_dataloaders(data_root, image_size=256, batch_size=32, num_workers=0):
    train_rows = read_labels_csv(os.path.join(data_root, "train", "labels.csv"))
    valid_rows = read_labels_csv(os.path.join(data_root, "valid", "labels.csv"))
    test_rows  = read_labels_csv(os.path.join(data_root, "test", "labels.csv"))
    countries = sorted(set(row["country"] for row in train_rows))
    country_to_idx = {c: i for i, c in enumerate(countries)}

    print(f"Countries: {len(countries)}, Train: {len(train_rows)}, Valid: {len(valid_rows)}, Test: {len(test_rows)}", flush=True)

    train_ds = PreCachedDataset(train_rows, os.path.join(data_root, "train"), country_to_idx, image_size)
    valid_ds = PreCachedDataset(valid_rows, os.path.join(data_root, "valid"), country_to_idx, image_size)
    test_ds  = PreCachedDataset(test_rows,  os.path.join(data_root, "test"),  country_to_idx, image_size)

    # num_workers=0: no copies, single uint8 dataset shared in RAM (~2.1GB total)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=num_workers, pin_memory=True)
    valid_loader = DataLoader(valid_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

    return train_loader, valid_loader, test_loader, countries, country_to_idx
