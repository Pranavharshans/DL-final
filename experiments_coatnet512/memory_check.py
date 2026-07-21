#!/usr/bin/env python3
"""Measure safe 512x512 CoAtNet micro-batch sizes on the current CUDA GPU."""

from __future__ import annotations

import argparse
import gc

import torch
from torch import nn

from experiments_v2.models.coatnet import CoAtNetMicro
from experiments_v2.models.common import count_parameters


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--batch-sizes",
        type=int,
        nargs="+",
        default=[4, 8, 12, 16, 24, 32],
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required for the memory check")
    device = torch.device("cuda")
    properties = torch.cuda.get_device_properties(device)
    print(f"GPU: {properties.name}")
    print(f"Total memory: {properties.total_memory / 1024**3:.2f} GB")
    print("Testing one full mixed-precision AdamW training step at 512x512...")

    for batch_size in args.batch_sizes:
        model = None
        optimizer = None
        images = None
        labels = None
        loss = None
        scaler = None
        try:
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(device)
            model = CoAtNetMicro(18).to(device)
            optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
            scaler = torch.cuda.amp.GradScaler()
            images = torch.randn(batch_size, 3, 512, 512, device=device)
            labels = torch.randint(0, 18, (batch_size,), device=device)
            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast():
                loss = nn.functional.cross_entropy(model(images), labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            torch.cuda.synchronize()
            allocated = torch.cuda.max_memory_allocated(device) / 1024**3
            reserved = torch.cuda.max_memory_reserved(device) / 1024**3
            print(
                f"PASS batch={batch_size:2d} "
                f"peak_allocated={allocated:.2f}GB peak_reserved={reserved:.2f}GB"
            )
        except torch.OutOfMemoryError:
            print(f"OOM  batch={batch_size:2d}")
        finally:
            del model, optimizer, images, labels, loss, scaler
            gc.collect()
            torch.cuda.empty_cache()

    print(f"Model parameters: {count_parameters(CoAtNetMicro(18)):,}")
    print("Choose the largest passing batch with at least 15% memory headroom.")


if __name__ == "__main__":
    main()
