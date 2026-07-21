#!/usr/bin/env python3
"""Probe safe micro-batches for CoAtNet-Large-4M at 512x512."""

from __future__ import annotations

import argparse
import gc

import torch
from torch import nn

from experiments_v2.models.common import count_parameters

from .model import CoAtNetLarge4M


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=[4, 8, 12, 16])
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU required")
    device = torch.device("cuda")
    properties = torch.cuda.get_device_properties(device)
    print(f"GPU: {properties.name} ({properties.total_memory / 1024**3:.2f} GB)")

    for batch_size in args.batch_sizes:
        model = optimizer = images = labels = loss = scaler = None
        try:
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(device)
            model = CoAtNetLarge4M(18).to(device)
            optimizer = torch.optim.AdamW(model.parameters(), lr=8e-4)
            scaler = torch.cuda.amp.GradScaler()
            images = torch.randn(batch_size, 3, 512, 512, device=device)
            labels = torch.randint(0, 18, (batch_size,), device=device)
            with torch.cuda.amp.autocast():
                loss = nn.functional.cross_entropy(model(images), labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            torch.cuda.synchronize()
            allocated = torch.cuda.max_memory_allocated(device) / 1024**3
            reserved = torch.cuda.max_memory_reserved(device) / 1024**3
            print(
                f"PASS batch={batch_size:2d} allocated={allocated:.2f}GB "
                f"reserved={reserved:.2f}GB"
            )
        except torch.OutOfMemoryError:
            print(f"OOM  batch={batch_size:2d}")
        finally:
            del model, optimizer, images, labels, loss, scaler
            gc.collect()
            torch.cuda.empty_cache()

    print(f"Parameters: {count_parameters(CoAtNetLarge4M(18)):,}")
    print("Keep at least 15% memory headroom for the real data pipeline.")


if __name__ == "__main__":
    main()
