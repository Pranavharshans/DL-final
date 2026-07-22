#!/usr/bin/env python3
"""Train an image-only local-global MIL network from scratch."""

from __future__ import annotations

import argparse
import copy
import json
import time
from dataclasses import replace
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader

from experiments_coatnet512.train import (
    MAX_PARAMETERS,
    RunConfig,
    SplitMetrics,
    autocast_context,
    cosine_with_warmup,
    print_confusion_matrix,
    seed_everything,
    validate_config,
)
from experiments_coatnet512_large.augmentation import SafeGeoAugment512
from experiments_v2.config import ExperimentConfig
from experiments_v2.data import create_dataloaders
from experiments_v2.engine import select_device
from experiments_v2.models.common import count_parameters

from .model import LocalGlobalMIL
from .views import FiveViewDataset


ROOT = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=ROOT.parent)
    parser.add_argument("--output-dir", type=Path, default=ROOT)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=8, dest="micro_batch_size")
    parser.add_argument("--accumulation-steps", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=8e-4)
    parser.add_argument("--weight-decay", type=float, default=1.5e-4)
    parser.add_argument("--label-smoothing", type=float, default=0.06)
    parser.add_argument("--warmup-epochs", type=int, default=5)
    parser.add_argument("--view-size", type=int, default=256)
    parser.add_argument("--ema-decay", type=float, default=0.999)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--allow-cpu", action="store_true")
    return parser.parse_args()


def make_loader(dataset, batch_size: int, workers: int, shuffle: bool, seed: int):
    generator = torch.Generator().manual_seed(seed) if shuffle else None
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        generator=generator,
        num_workers=workers,
        pin_memory=torch.cuda.is_available(),
    )


@torch.no_grad()
def update_ema(ema: nn.Module, model: nn.Module, decay: float) -> None:
    model_values = model.state_dict()
    for name, value in ema.state_dict().items():
        source = model_values[name].detach()
        if value.is_floating_point():
            value.mul_(decay).add_(source, alpha=1.0 - decay)
        else:
            value.copy_(source)


def train_epoch(
    model,
    loader,
    criterion,
    optimizer,
    scaler,
    device,
    accumulation,
    epoch,
    ema,
    ema_decay,
):
    model.train()
    optimizer.zero_grad(set_to_none=True)
    total_loss = total_correct = total = 0
    for batch_index, (views, labels, _) in enumerate(loader):
        views, labels = (
            views.to(device, non_blocking=True),
            labels.to(device, non_blocking=True),
        )
        group_start = (batch_index // accumulation) * accumulation
        group_size = min(accumulation, len(loader) - group_start)
        with autocast_context(device):
            logits = model(views)
            raw_loss = criterion(logits, labels)
            loss = raw_loss / group_size
        if scaler.is_enabled():
            scaler.scale(loss).backward()
        else:
            loss.backward()
        should_step = (batch_index + 1) % accumulation == 0 or batch_index + 1 == len(
            loader
        )
        if should_step:
            if scaler.is_enabled():
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            update_ema(ema, model, ema_decay)
        count = labels.shape[0]
        total_loss += raw_loss.detach().item() * count
        total_correct += logits.detach().argmax(1).eq(labels).sum().item()
        total += count
        if (batch_index + 1) % 50 == 0 or batch_index + 1 == len(loader):
            print(
                f"  epoch={epoch:03d} batch={batch_index + 1:04d}/{len(loader):04d} loss={total_loss / total:.4f} accuracy={total_correct / total:.4f}",
                flush=True,
            )
    return total_loss / total, total_correct / total


@torch.inference_mode()
def evaluate(model, loader, criterion, device, countries) -> SplitMetrics:
    model.eval()
    total_loss = total_correct = total = 0
    correct = [0] * len(countries)
    counts = [0] * len(countries)
    confusion = [[0] * len(countries) for _ in countries]
    for views, labels, _ in loader:
        views, labels = (
            views.to(device, non_blocking=True),
            labels.to(device, non_blocking=True),
        )
        with autocast_context(device):
            logits = model(views)
            loss = criterion(logits, labels)
        predictions = logits.argmax(1)
        count = labels.shape[0]
        total_loss += loss.item() * count
        total_correct += predictions.eq(labels).sum().item()
        total += count
        for truth, prediction in zip(
            labels.tolist(), predictions.tolist(), strict=True
        ):
            counts[truth] += 1
            correct[truth] += int(truth == prediction)
            confusion[truth][prediction] += 1
    return SplitMetrics(
        total_loss / total,
        total_correct / total,
        total_correct,
        total,
        {
            country: correct[i] / max(1, counts[i])
            for i, country in enumerate(countries)
        },
        confusion,
    )


def main() -> None:
    args = parse_args()
    config = RunConfig(
        args.data_root.resolve(),
        args.output_dir.resolve(),
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
    if not 192 <= args.view_size <= 320:
        raise ValueError("view size must be between 192 and 320")
    if not 0.9 <= args.ema_decay < 1.0:
        raise ValueError("EMA decay must be in [0.9, 1.0)")
    seed_everything(config.seed)
    device = select_device()
    if device.type != "cuda" and not args.allow_cpu:
        raise RuntimeError("CUDA GPU required; use --allow-cpu only for smoke tests")

    print("=" * 76)
    print("Local-Global MIL — 1 global + 4 learned local views")
    print("=" * 76)
    print(f"Device: {device}; effective batch: {config.effective_batch_size}")
    print("7,560 unique train images; five generated views per sample per epoch.")
    print("No geo metadata, external data, or pretrained weights.")
    data_config = replace(
        ExperimentConfig(),
        data_root=config.data_root,
        image_size=512,
        train_cache_size=512,
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
    data.train.dataset.augment = SafeGeoAugment512(512)
    train_loader = make_loader(
        FiveViewDataset(data.train.dataset, args.view_size, True),
        config.micro_batch_size,
        config.num_workers,
        True,
        config.seed,
    )
    valid_loader = make_loader(
        FiveViewDataset(data.valid.dataset, args.view_size, False),
        config.micro_batch_size,
        config.num_workers,
        False,
        config.seed,
    )
    test_loader = make_loader(
        FiveViewDataset(data.test.dataset, args.view_size, False),
        config.micro_batch_size,
        config.num_workers,
        False,
        config.seed,
    )

    seed_everything(config.seed)
    model = LocalGlobalMIL(len(data.countries)).to(device)
    parameters = count_parameters(model)
    if parameters > MAX_PARAMETERS:
        raise ValueError(
            f"Model has {parameters:,} parameters; limit is {MAX_PARAMETERS:,}"
        )
    ema = copy.deepcopy(model).eval()
    for parameter in ema.parameters():
        parameter.requires_grad_(False)
    criterion = nn.CrossEntropyLoss(label_smoothing=config.label_smoothing)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda epoch: cosine_with_warmup(epoch, config.warmup_epochs, config.epochs),
    )
    scaler = torch.cuda.amp.GradScaler(enabled=device.type == "cuda")
    best_accuracy, best_epoch, best_state = -1.0, 0, None
    history = []
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    started = time.monotonic()
    for epoch in range(1, config.epochs + 1):
        train_loss, train_accuracy = train_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            scaler,
            device,
            config.accumulation_steps,
            epoch,
            ema,
            args.ema_decay,
        )
        valid = evaluate(ema, valid_loader, criterion, device, data.countries)
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "train_accuracy": train_accuracy,
                "valid_ema_loss": valid.loss,
                "valid_ema_accuracy": valid.accuracy,
                "learning_rate": optimizer.param_groups[0]["lr"],
            }
        )
        if valid.accuracy > best_accuracy:
            best_accuracy, best_epoch = valid.accuracy, epoch
            best_state = {
                name: value.detach().cpu().clone()
                for name, value in ema.state_dict().items()
            }
        scheduler.step()
        print(
            f"EPOCH {epoch:03d}/{config.epochs} train_loss={train_loss:.4f} train_acc={train_accuracy:.4f} valid_ema_loss={valid.loss:.4f} valid_ema_acc={valid.accuracy:.4f} best={best_accuracy:.4f}@{best_epoch:03d}",
            flush=True,
        )

    if best_state is None:
        raise RuntimeError("Training completed without a checkpoint")
    ema.load_state_dict(best_state)
    training_seconds = time.monotonic() - started
    # Disable augmentation before reporting deterministic train accuracy.
    data.train.dataset.augment = None
    train_eval_loader = make_loader(
        FiveViewDataset(data.train.dataset, args.view_size, False),
        config.micro_batch_size,
        config.num_workers,
        False,
        config.seed,
    )
    train_metrics = evaluate(ema, train_eval_loader, criterion, device, data.countries)
    valid_metrics = evaluate(ema, valid_loader, criterion, device, data.countries)
    # Test is evaluated exactly once after model selection on validation.
    test_metrics = evaluate(ema, test_loader, criterion, device, data.countries)
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

    config.output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = config.output_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = checkpoint_dir / "local_global_mil_best.pt"
    torch.save(
        {
            "model_name": "LocalGlobalMIL",
            "model_state": best_state,
            "countries": data.countries,
            "normalization_mean": data.mean,
            "normalization_std": data.std,
            "config": config.serializable(),
            "view_size": args.view_size,
            "views": 5,
            "ema_decay": args.ema_decay,
            "parameters": parameters,
            "best_epoch": best_epoch,
            "best_valid_accuracy": best_accuracy,
        },
        checkpoint_path,
    )
    report = {
        "experiment": "Local-Global MIL",
        "parameters": parameters,
        "unique_train_images": len(data.train.dataset),
        "views_per_sample": 5,
        "augmented_presentations": len(data.train.dataset) * config.epochs,
        "view_tensors_processed": len(data.train.dataset) * config.epochs * 5,
        "best_epoch": best_epoch,
        "best_valid_accuracy": best_accuracy,
        "training_seconds": training_seconds,
        "peak_cuda_allocated_gb": peak_allocated,
        "peak_cuda_reserved_gb": peak_reserved,
        "countries": data.countries,
        "config": config.serializable(),
        "train": train_metrics.serializable(),
        "valid": valid_metrics.serializable(),
        "test": test_metrics.serializable(),
        "history": history,
        "checkpoint": str(checkpoint_path),
    }
    results_json = config.output_dir / "results.json"
    results_json.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    results_md = config.output_dir / "results.md"
    results_md.write_text(
        f"# Local-Global MIL Results\n\n- Parameters: {parameters:,}\n- Best epoch: {best_epoch}/{config.epochs}\n- Train accuracy: {train_metrics.accuracy:.4f}\n- Valid accuracy: {valid_metrics.accuracy:.4f}\n- Test accuracy: {test_metrics.accuracy:.4f}\n",
        encoding="utf-8",
    )

    print("\n" + "=" * 76)
    print("FINAL RESULTS — Local-Global MIL")
    print("=" * 76)
    print(f"Parameters: {parameters:,} / {MAX_PARAMETERS:,}")
    print(f"Best epoch: {best_epoch}/{config.epochs}")
    print(f"Training time: {training_seconds / 3600:.2f}h")
    print(f"Peak CUDA allocated/reserved: {peak_allocated:.2f}/{peak_reserved:.2f} GB")
    print(f"Train: {train_metrics.accuracy:.4f}")
    print(f"Valid: {valid_metrics.accuracy:.4f}")
    print(f"Test:  {test_metrics.accuracy:.4f}")
    print("\nPer-country validation/test accuracy:")
    for country in data.countries:
        print(
            f"  {country:20s} valid={valid_metrics.per_country_accuracy[country]:.4f} test={test_metrics.per_country_accuracy[country]:.4f}"
        )
    print_confusion_matrix(test_metrics.confusion_matrix, data.countries)
    print(f"\nCheckpoint: {checkpoint_path}")
    print(f"JSON results: {results_json}")
    print(f"Markdown results: {results_md}")


if __name__ == "__main__":
    main()
