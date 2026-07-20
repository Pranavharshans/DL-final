#!/usr/bin/env python3
"""Fast preflight check before expensive GPU training."""

from __future__ import annotations

import csv
from collections import Counter
from pathlib import Path

import torch

from experiments_v2.models import MODEL_REGISTRY, build_model
from experiments_v2.models.common import count_parameters


ROOT = Path(__file__).resolve().parent.parent
EXPECTED_COUNTS = {"train": 7560, "valid": 2160, "test": 1080}


def verify_split() -> None:
    seen: set[str] = set()
    expected_countries: set[str] | None = None
    for split, expected_count in EXPECTED_COUNTS.items():
        split_dir = ROOT / split
        with (split_dir / "labels.csv").open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        if len(rows) != expected_count:
            raise ValueError(
                f"{split}: expected {expected_count} rows, found {len(rows)}"
            )
        filenames = {row["filename"] for row in rows}
        if len(filenames) != len(rows):
            raise ValueError(f"{split}: duplicate filenames")
        if filenames.intersection(seen):
            raise ValueError(f"{split}: filename overlap with an earlier split")
        missing = [
            filename for filename in filenames if not (split_dir / filename).is_file()
        ]
        if missing:
            raise FileNotFoundError(f"{split}: {len(missing)} missing images")
        countries = {row["country"] for row in rows}
        expected_countries = (
            countries if expected_countries is None else expected_countries
        )
        if countries != expected_countries:
            raise ValueError(f"{split}: country set differs from training")
        counts = Counter(row["country"] for row in rows)
        if len(set(counts.values())) != 1:
            raise ValueError(f"{split}: class counts are unbalanced")
        seen.update(filenames)


def verify_models() -> None:
    images = torch.randn(2, 3, 128, 128)
    for name in MODEL_REGISTRY:
        model = build_model(name, 18).eval()
        parameters = count_parameters(model)
        if parameters > 5_000_000:
            raise ValueError(f"{name}: {parameters:,} parameters exceeds the limit")
        with torch.no_grad():
            logits = model(images)
        if logits.shape != (2, 18) or not torch.isfinite(logits).all():
            raise ValueError(f"{name}: invalid output {tuple(logits.shape)}")
        print(f"PASS {name:29s} {parameters:>9,} parameters")


if __name__ == "__main__":
    verify_split()
    verify_models()
    print("All split and model contracts passed.")
