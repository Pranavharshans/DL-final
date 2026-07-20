#!/usr/bin/env python3
"""Evaluate one locked v2 checkpoint on the untouched internal test split."""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

import torch
from torch import nn

from experiments_v2.config import ExperimentConfig
from experiments_v2.data import create_dataloaders
from experiments_v2.engine import run_epoch, select_device
from experiments_v2.models import build_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument(
        "--data-root", type=Path, default=Path(__file__).resolve().parent.parent
    )
    parser.add_argument("--batch-size", type=int, default=32)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = select_device()
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    config = replace(
        ExperimentConfig(),
        data_root=args.data_root.resolve(),
        batch_size=args.batch_size,
    )
    data = create_dataloaders(config)
    if data.countries != checkpoint["countries"]:
        raise ValueError("checkpoint country order does not match the dataset")
    model = build_model(checkpoint["model_name"], len(data.countries))
    model.load_state_dict(checkpoint["model_state"])
    model.to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=config.label_smoothing)
    metrics = run_epoch(model, data.test, criterion, device, len(data.countries))
    print(f"Test accuracy: {metrics.accuracy:.4f}")
    print(f"Test loss: {metrics.loss:.4f}")
    for country, accuracy in metrics.per_country_accuracy(data.countries).items():
        print(f"{country:20s} {accuracy:.4f}")


if __name__ == "__main__":
    main()
