#!/usr/bin/env python3
"""Run all 20 models sequentially on the GPU, updating results.md after each."""
import sys
import time
import subprocess
from pathlib import Path

import torch

from models import MODEL_REGISTRY, count_params
from train_model import train_and_eval

DATA_ROOT = "/workspace/DL-final"
RESULTS_FILE = Path(DATA_ROOT) / "results.md"
NUM_MODELS = 20


def append_result(result, rank):
    """Append one model's results to the markdown table."""
    line = (
        f"| {rank} | {result['model']} | {result['params']:,} | "
        f"{result['train_acc']:.4f} | {result['valid_acc']:.4f} | {result['test_acc']:.4f} | "
        f"{result['time_sec']:.0f}s |"
    )
    with open(RESULTS_FILE, "a") as f:
        f.write(line + "\n")


def init_results_file():
    header = """# Model Results — Image Country Classification

All models trained on RTX 2060 12GB, 512×512 images, 50 epochs, batch_size=32.

| Rank | Model | Params | Train Acc | Valid Acc | Test Acc | Time |
|------|-------|--------|-----------|-----------|----------|------|
"""
    with open(RESULTS_FILE, "w") as f:
        f.write(header)


def git_push():
    """Commit results.md and push to GitHub."""
    subprocess.run(["git", "-C", DATA_ROOT, "add", "results.md"], check=False)
    subprocess.run(["git", "-C", DATA_ROOT, "commit", "-m", "Update results"], check=False)
    subprocess.run(["git", "-C", DATA_ROOT, "push"], check=False)


def log(msg):
    print(msg, flush=True)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log(f"Device: {device}")
    log(f"Models to run: {len(MODEL_REGISTRY)}")
    log(f"Data root: {DATA_ROOT}")

    init_results_file()
    results = []

    for rank, (name, build_fn) in enumerate(MODEL_REGISTRY.items(), 1):
        log(f"\n{'#'*60}")
        log(f"# MODEL {rank}/{NUM_MODELS}: {name}")
        log(f"{'#'*60}")

        result = train_and_eval(name, build_fn, DATA_ROOT, device, epochs=50, batch_size=32, image_size=256)
        results.append(result)
        append_result(result, rank)
        git_push()
        log(f"Pushed results for {name}")

    # Print final ranking
    results.sort(key=lambda r: r["test_acc"], reverse=True)
    log(f"\n{'='*60}")
    log("FINAL RANKING (by Test Accuracy)")
    log(f"{'='*60}")
    for i, r in enumerate(results, 1):
        log(f"  {i:2d}. {r['model']:30s} Test: {r['test_acc']:.4f}  Valid: {r['valid_acc']:.4f}  Train: {r['train_acc']:.4f}  ({r['params']:,} params, {r['time_sec']:.0f}s)")


if __name__ == "__main__":
    main()
