"""Data loading — shared across all models."""
import csv
import os

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset


def read_labels_csv(csv_path):
    with open(csv_path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def build_country_mapping(rows):
    countries = sorted(set(row["country"] for row in rows))
    country_to_idx = {c: i for i, c in enumerate(countries)}
    return countries, country_to_idx


class CountryDataset(Dataset):
    def __init__(self, rows, image_dir, country_to_idx, image_size=512):
        self.rows = rows
        self.image_dir = image_dir
        self.country_to_idx = country_to_idx
        self.image_size = image_size

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        row = self.rows[idx]
        img = Image.open(os.path.join(self.image_dir, row["filename"])).convert("RGB")
        img = img.resize((self.image_size, self.image_size), Image.BILINEAR)
        tensor = torch.from_numpy(np.array(img, dtype=np.float32) / 255.0).permute(2, 0, 1)
        label = self.country_to_idx[row["country"]]
        lat = float(row.get("lat", 0) or 0)
        lng = float(row.get("lng", 0) or 0)
        return tensor, label, torch.tensor([lat, lng], dtype=torch.float32)


def create_dataloaders(data_root, image_size=512, batch_size=32, num_workers=4):
    train_rows = read_labels_csv(os.path.join(data_root, "train", "labels.csv"))
    valid_rows = read_labels_csv(os.path.join(data_root, "valid", "labels.csv"))
    test_rows  = read_labels_csv(os.path.join(data_root, "test", "labels.csv"))
    countries, country_to_idx = build_country_mapping(train_rows)

    train_ds = CountryDataset(train_rows, os.path.join(data_root, "train"), country_to_idx, image_size)
    valid_ds = CountryDataset(valid_rows, os.path.join(data_root, "valid"), country_to_idx, image_size)
    test_ds  = CountryDataset(test_rows,  os.path.join(data_root, "test"),  country_to_idx, image_size)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=num_workers, pin_memory=True)
    valid_loader = DataLoader(valid_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

    return train_loader, valid_loader, test_loader, countries, country_to_idx
