#!/usr/bin/env python3
"""Train the 4.25M CoAtNet using the exact 2M baseline augmentation contract."""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import replace
from pathlib import Path

import torch
from torch import Tensor, nn

from experiments_coatnet512.train import (
    MAX_PARAMETERS,
    RunConfig,
    cosine_with_warmup,
    deterministic_train_loader,
    evaluate,
    print_confusion_matrix,
    seed_everything,
    train_epoch,
    validate_config,
)
from experiments_coatnet512_large.model import CoAtNetLarge4M
from experiments_v2.augmentation import GeoAugmentV1
from experiments_v2.config import ExperimentConfig
from experiments_v2.data import create_dataloaders
from experiments_v2.engine import select_device
from experiments_v2.models.common import count_parameters


ROOT = Path(__file__).resolve().parent
EXPECTED_PARAMETERS = 4_252_785


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=ROOT.parent)
    parser.add_argument("--output-dir", type=Path, default=ROOT)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=16, dest="micro_batch_size")
    parser.add_argument("--accumulation-steps", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--label-smoothing", type=float, default=0.08)
    parser.add_argument("--warmup-epochs", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--allow-cpu", action="store_true")
    return parser.parse_args()


def build_augmentation(output_size: int) -> GeoAugmentV1:
    """Make the exact augmentation class used by the current 2M experiment."""
    return GeoAugmentV1(output_size)


def write_markdown(report: dict[str, object], path: Path) -> None:
    train = report["train"]
    valid = report["valid"]
    test = report["test"]
    lines = [
        "# CoAtNet-Large-4M Size-Control Results",
        "",
        f"- Parameters: {int(report['parameters']):,}",
        f"- Best epoch: {int(report['best_epoch'])}",
        f"- Augmentation: {report['augmentation']}",
        f"- Inference: {report['inference']}",
        f"- Training time: {float(report['training_seconds']):.1f}s",
        "",
        "| Split | Loss | Accuracy | Correct | Total |",
        "|---|---:|---:|---:|---:|",
        f"| Train | {train['loss']:.4f} | {train['accuracy']:.4f} | {train['correct']} | {train['total']} |",
        f"| Valid | {valid['loss']:.4f} | {valid['accuracy']:.4f} | {valid['correct']} | {valid['total']} |",
        f"| Test | {test['loss']:.4f} | {test['accuracy']:.4f} | {test['correct']} | {test['total']} |",
        "",
        "## Per-country accuracy",
        "",
        "| Country | Valid | Test |",
        "|---|---:|---:|",
    ]
    valid_country = valid["per_country_accuracy"]
    test_country = test["per_country_accuracy"]
    for country in report["countries"]:
        lines.append(
            f"| {country} | {valid_country[country]:.4f} | {test_country[country]:.4f} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    config = RunConfig(
        data_root=args.data_root.resolve(),
        output_dir=args.output_dir.resolve(),
        epochs=args.epochs,
        micro_batch_size=args.micro_batch_size,
        accumulation_steps=args.accumulation_steps,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        label_smoothing=args.label_smoothing,
        warmup_epochs=args.warmup_epochs,
        seed=args.seed,
        num_workers=args.num_workers,
    )
    validate_config(config)
    seed_everything(config.seed)
    device = select_device()
    if device.type != "cuda" and not args.allow_cpu:
        raise RuntimeError("CUDA GPU required; use --allow-cpu only for smoke tests")

    print("=" * 76)
    print("CoAtNet-Large-4M 512 — size-only controlled ablation")
    print("=" * 76)
    print(f"Device: {device}")
    if device.type == "cuda":
        properties = torch.cuda.get_device_properties(device)
        print(f"GPU: {properties.name}")
        print(f"GPU memory: {properties.total_memory / 1024**3:.2f} GB")
    print(f"Micro-batch: {config.micro_batch_size}")
    print(f"Accumulation: {config.accumulation_steps}")
    print(f"Effective batch: {config.effective_batch_size}")
    print("Augmentation: original GeoAugmentV1 (identical to 2M baseline)")
    print("Inference: single-crop only")
    print("No coordinates, pretrained weights, external data, or multi-crop.")
    print("Pre-caching all splits at 512x512 uint8 (approximately 8.5 GB)...")

    data_config = replace(
        ExperimentConfig(),
        data_root=config.data_root,
        image_size=config.image_size,
        train_cache_size=config.cache_size,
        batch_size=config.micro_batch_size,
        epochs=config.epochs,
        learning_rate=config.learning_rate,
        weight_decay=config.weight_decay,
        label_smoothing=config.label_smoothing,
        warmup_epochs=config.warmup_epochs,
        num_workers=config.num_workers,
        seed=config.seed,
    )
    data = create_dataloaders(data_config)
    # Set this explicitly so a future shared-loader change cannot invalidate the ablation.
    data.train.dataset.augment = build_augmentation(config.image_size)
    if data.train.generator is not None:
        data.train.generator.manual_seed(config.seed)

    seed_everything(config.seed)
    model = CoAtNetLarge4M(len(data.countries))
    parameters = count_parameters(model)
    if parameters != EXPECTED_PARAMETERS or parameters > MAX_PARAMETERS:
        raise ValueError(
            f"Parameter contract changed: {parameters:,} != {EXPECTED_PARAMETERS:,}"
        )
    model.to(device)
    seed_everything(config.seed + 10_000)

    criterion = nn.CrossEntropyLoss(label_smoothing=config.label_smoothing)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda epoch: cosine_with_warmup(epoch, config.warmup_epochs, config.epochs),
    )
    scaler = torch.cuda.amp.GradScaler(enabled=device.type == "cuda")
    best_accuracy = -1.0
    best_epoch = 0
    best_state: dict[str, Tensor] | None = None
    history: list[dict[str, float | int]] = []
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    started = time.monotonic()

    for epoch in range(1, config.epochs + 1):
        train_loss, train_accuracy = train_epoch(
            model,
            data.train,
            criterion,
            optimizer,
            scaler,
            device,
            config.accumulation_steps,
            epoch,
        )
        valid_metrics = evaluate(model, data.valid, criterion, device, data.countries)
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "train_accuracy": train_accuracy,
                "valid_loss": valid_metrics.loss,
                "valid_accuracy": valid_metrics.accuracy,
                "learning_rate": optimizer.param_groups[0]["lr"],
            }
        )
        if valid_metrics.accuracy > best_accuracy:
            best_accuracy = valid_metrics.accuracy
            best_epoch = epoch
            best_state = {
                name: value.detach().cpu().clone()
                for name, value in model.state_dict().items()
            }
        scheduler.step()
        print(
            f"EPOCH {epoch:02d}/{config.epochs} train_loss={train_loss:.4f} "
            f"train_acc={train_accuracy:.4f} valid_loss={valid_metrics.loss:.4f} "
            f"valid_acc={valid_metrics.accuracy:.4f} "
            f"best={best_accuracy:.4f}@{best_epoch:02d}",
            flush=True,
        )

    if best_state is None:
        raise RuntimeError("Training completed without a checkpoint")
    training_seconds = time.monotonic() - started
    model.load_state_dict(best_state)
    peak_allocated = (
        torch.cuda.max_memory_allocated(device) / 1024**3
        if device.type == "cuda"
        else 0.0
    )
    peak_reserved = (
        torch.cuda.max_memory_reserved(device) / 1024**3
        if device.type == "cuda"
        else 0.0
    )

    print("\nBest checkpoint loaded. Running final single-crop evaluation...")
    train_loader = deterministic_train_loader(data, config)
    train_metrics = evaluate(model, train_loader, criterion, device, data.countries)
    valid_metrics = evaluate(model, data.valid, criterion, device, data.countries)
    # The test split is touched exactly once after validation selects the checkpoint.
    test_metrics = evaluate(model, data.test, criterion, device, data.countries)

    config.output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = config.output_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = checkpoint_dir / "coatnet_large4m_control_512_best.pt"
    torch.save(
        {
            "model_name": "CoAtNet-Large-4M-512-Control",
            "model_state": best_state,
            "countries": data.countries,
            "normalization_mean": data.mean,
            "normalization_std": data.std,
            "config": config.serializable(),
            "parameters": parameters,
            "best_epoch": best_epoch,
            "best_valid_accuracy": best_accuracy,
            "augmentation": "GeoAugmentV1",
            "inference": "single-crop",
        },
        checkpoint_path,
    )
    report: dict[str, object] = {
        "experiment": "CoAtNet-Large-4M 512 size-only control",
        "parameters": parameters,
        "augmentation": "GeoAugmentV1",
        "inference": "single-crop",
        "best_epoch": best_epoch,
        "best_valid_accuracy": best_accuracy,
        "training_seconds": training_seconds,
        "peak_cuda_allocated_gb": peak_allocated,
        "peak_cuda_reserved_gb": peak_reserved,
        "effective_batch_size": config.effective_batch_size,
        "countries": data.countries,
        "config": config.serializable(),
        "train": train_metrics.serializable(),
        "valid": valid_metrics.serializable(),
        "test": test_metrics.serializable(),
        "history": history,
        "checkpoint": str(checkpoint_path),
    }
    results_json = config.output_dir / "results.json"
    results_markdown = config.output_dir / "results.md"
    results_json.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    write_markdown(report, results_markdown)

    print("\n" + "=" * 76)
    print("FINAL RESULTS — CoAtNet-Large-4M size-only control")
    print("=" * 76)
    print(f"Parameters: {parameters:,} / {MAX_PARAMETERS:,}")
    print(f"Best epoch: {best_epoch}/{config.epochs}")
    print(f"Training time: {training_seconds / 3600:.2f}h")
    print(f"Peak CUDA allocated/reserved: {peak_allocated:.2f}/{peak_reserved:.2f} GB")
    print(f"Train single: {train_metrics.accuracy:.4f}")
    print(f"Valid single: {valid_metrics.accuracy:.4f}")
    print(f"Test single: {test_metrics.accuracy:.4f}")
    print("\nPer-country validation/test single-crop accuracy:")
    for country in data.countries:
        print(
            f"  {country:20s} valid={valid_metrics.per_country_accuracy[country]:.4f} "
            f"test={test_metrics.per_country_accuracy[country]:.4f}"
        )
    print_confusion_matrix(test_metrics.confusion_matrix, data.countries)
    print(f"\nCheckpoint: {checkpoint_path}")
    print(f"JSON results: {results_json}")
    print(f"Markdown results: {results_markdown}")


if __name__ == "__main__":
    main()
