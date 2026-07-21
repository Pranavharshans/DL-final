#!/usr/bin/env python3
"""Train CoAtNet-Micro from scratch at 512x512 and print final split metrics."""

from __future__ import annotations

import argparse
import json
import math
import random
import time
from contextlib import nullcontext
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import numpy as np
import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader

from experiments_v2.config import ExperimentConfig
from experiments_v2.data import DataBundle, create_dataloaders
from experiments_v2.engine import select_device
from experiments_v2.models.coatnet import CoAtNetMicro
from experiments_v2.models.common import count_parameters


ROOT = Path(__file__).resolve().parent
MAX_PARAMETERS = 5_000_000


@dataclass(frozen=True)
class RunConfig:
    data_root: Path
    output_dir: Path
    image_size: int = 512
    cache_size: int = 512
    epochs: int = 50
    micro_batch_size: int = 8
    accumulation_steps: int = 4
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    label_smoothing: float = 0.08
    warmup_epochs: int = 3
    seed: int = 42
    num_workers: int = 0

    @property
    def effective_batch_size(self) -> int:
        return self.micro_batch_size * self.accumulation_steps

    def serializable(self) -> dict[str, object]:
        values = asdict(self)
        values["data_root"] = str(self.data_root)
        values["output_dir"] = str(self.output_dir)
        values["effective_batch_size"] = self.effective_batch_size
        return values


@dataclass
class SplitMetrics:
    loss: float
    accuracy: float
    correct: int
    total: int
    per_country_accuracy: dict[str, float]
    confusion_matrix: list[list[int]]

    def serializable(self) -> dict[str, object]:
        return asdict(self)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-root",
        type=Path,
        default=ROOT.parent,
        help="Directory containing train/, valid/, and test/",
    )
    parser.add_argument("--output-dir", type=Path, default=ROOT)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=8, dest="micro_batch_size")
    parser.add_argument("--accumulation-steps", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--label-smoothing", type=float, default=0.08)
    parser.add_argument("--warmup-epochs", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument(
        "--allow-cpu",
        action="store_true",
        help="Allow a CPU/MPS run; intended only for smoke testing",
    )
    return parser.parse_args()


def validate_config(config: RunConfig) -> None:
    if config.epochs < 1:
        raise ValueError("epochs must be positive")
    if config.micro_batch_size < 1 or config.accumulation_steps < 1:
        raise ValueError("batch size and accumulation steps must be positive")
    if config.warmup_epochs >= config.epochs:
        raise ValueError("warmup epochs must be smaller than total epochs")
    for split in ("train", "valid", "test"):
        split_dir = config.data_root / split
        if not split_dir.is_dir() or not (split_dir / "labels.csv").is_file():
            raise FileNotFoundError(f"Expected {split_dir}/ and labels.csv")


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def autocast_context(device: torch.device):
    if device.type == "cuda":
        return torch.cuda.amp.autocast()
    return nullcontext()


def cosine_with_warmup(epoch: int, warmup_epochs: int, total_epochs: int) -> float:
    if epoch < warmup_epochs:
        return (epoch + 1) / max(1, warmup_epochs)
    progress = (epoch - warmup_epochs) / max(1, total_epochs - warmup_epochs - 1)
    return 0.5 * (1.0 + math.cos(math.pi * progress))


def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: torch.cuda.amp.GradScaler,
    device: torch.device,
    accumulation_steps: int,
    epoch: int,
) -> tuple[float, float]:
    model.train()
    optimizer.zero_grad(set_to_none=True)
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    total_batches = len(loader)

    for batch_index, (images, labels, _) in enumerate(loader):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        group_start = (batch_index // accumulation_steps) * accumulation_steps
        group_size = min(accumulation_steps, total_batches - group_start)

        with autocast_context(device):
            logits = model(images)
            raw_loss = criterion(logits, labels)
            scaled_loss = raw_loss / group_size

        if scaler.is_enabled():
            scaler.scale(scaled_loss).backward()
        else:
            scaled_loss.backward()

        should_step = (
            batch_index + 1
        ) % accumulation_steps == 0 or batch_index + 1 == total_batches
        if should_step:
            if scaler.is_enabled():
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            optimizer.zero_grad(set_to_none=True)

        batch_size = labels.shape[0]
        total_loss += raw_loss.detach().item() * batch_size
        total_correct += logits.detach().argmax(dim=1).eq(labels).sum().item()
        total_samples += batch_size
        if (batch_index + 1) % 50 == 0 or batch_index + 1 == total_batches:
            print(
                f"  epoch={epoch:02d} batch={batch_index + 1:04d}/{total_batches:04d} "
                f"loss={total_loss / total_samples:.4f} "
                f"accuracy={total_correct / total_samples:.4f}",
                flush=True,
            )

    return total_loss / total_samples, total_correct / total_samples


@torch.inference_mode()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    countries: list[str],
) -> SplitMetrics:
    model.eval()
    num_classes = len(countries)
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    per_country_correct = [0] * num_classes
    per_country_total = [0] * num_classes
    confusion = [[0] * num_classes for _ in range(num_classes)]

    for images, labels, _ in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        with autocast_context(device):
            logits = model(images)
            loss = criterion(logits, labels)
        predictions = logits.argmax(dim=1)
        batch_size = labels.shape[0]
        total_loss += loss.item() * batch_size
        total_correct += predictions.eq(labels).sum().item()
        total_samples += batch_size
        for truth, prediction in zip(
            labels.tolist(), predictions.tolist(), strict=True
        ):
            per_country_total[truth] += 1
            per_country_correct[truth] += int(truth == prediction)
            confusion[truth][prediction] += 1

    return SplitMetrics(
        loss=total_loss / total_samples,
        accuracy=total_correct / total_samples,
        correct=total_correct,
        total=total_samples,
        per_country_accuracy={
            country: per_country_correct[index] / max(1, per_country_total[index])
            for index, country in enumerate(countries)
        },
        confusion_matrix=confusion,
    )


def deterministic_train_loader(data: DataBundle, config: RunConfig) -> DataLoader:
    dataset = data.train.dataset
    if hasattr(dataset, "augment"):
        dataset.augment = None
    return DataLoader(
        dataset,
        batch_size=config.micro_batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=torch.cuda.is_available(),
    )


def print_confusion_matrix(matrix: list[list[int]], countries: list[str]) -> None:
    print("\nTest confusion matrix (rows=true, columns=predicted):")
    print("     " + " ".join(f"{index:4d}" for index in range(len(countries))))
    for index, row in enumerate(matrix):
        print(f"{index:02d} | " + " ".join(f"{value:4d}" for value in row))
    print("Country index:")
    for index, country in enumerate(countries):
        print(f"  {index:02d}: {country}")


def write_markdown(report: dict[str, object], path: Path) -> None:
    train = report["train"]
    valid = report["valid"]
    test = report["test"]
    lines = [
        "# CoAtNet-Micro 512x512 Results",
        "",
        f"- Parameters: {int(report['parameters']):,}",
        f"- Best epoch: {int(report['best_epoch'])}",
        f"- Effective batch size: {int(report['effective_batch_size'])}",
        f"- Training time: {float(report['training_seconds']):.1f}s",
        f"- Peak CUDA allocation: {float(report['peak_cuda_allocated_gb']):.2f} GB",
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
        raise RuntimeError(
            "CUDA GPU required. Pass --allow-cpu only for smoke testing."
        )

    print("=" * 72)
    print("CoAtNet-Micro 512x512 experiment")
    print("=" * 72)
    print(f"Device: {device}")
    if device.type == "cuda":
        properties = torch.cuda.get_device_properties(device)
        print(f"GPU: {properties.name}")
        print(f"GPU memory: {properties.total_memory / 1024**3:.2f} GB")
    print(f"Micro-batch size: {config.micro_batch_size}")
    print(f"Accumulation steps: {config.accumulation_steps}")
    print(f"Effective batch size: {config.effective_batch_size}")
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
    if data.train.generator is not None:
        data.train.generator.manual_seed(config.seed)

    seed_everything(config.seed)
    model = CoAtNetMicro(len(data.countries))
    parameters = count_parameters(model)
    if parameters > MAX_PARAMETERS:
        raise ValueError(
            f"Model has {parameters:,} parameters; limit is {MAX_PARAMETERS:,}"
        )
    model.to(device)
    # Separate model initialization randomness from augmentation/dropout randomness.
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
        scheduler.step()
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
        print(
            f"EPOCH {epoch:02d}/{config.epochs} "
            f"train_loss={train_loss:.4f} train_acc={train_accuracy:.4f} "
            f"valid_loss={valid_metrics.loss:.4f} "
            f"valid_acc={valid_metrics.accuracy:.4f} "
            f"best={best_accuracy:.4f}@{best_epoch:02d}",
            flush=True,
        )

    if best_state is None:
        raise RuntimeError("Training completed without producing a checkpoint")
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

    print("\nBest checkpoint loaded. Running final deterministic evaluation...")
    train_eval_loader = deterministic_train_loader(data, config)
    train_metrics = evaluate(
        model, train_eval_loader, criterion, device, data.countries
    )
    valid_metrics = evaluate(model, data.valid, criterion, device, data.countries)
    # Test is evaluated exactly once, after validation has selected the checkpoint.
    test_metrics = evaluate(model, data.test, criterion, device, data.countries)

    config.output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = config.output_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = checkpoint_dir / "coatnet_micro_512_best.pt"
    torch.save(
        {
            "model_name": "25_CoAtNet-Micro",
            "model_state": best_state,
            "countries": data.countries,
            "normalization_mean": data.mean,
            "normalization_std": data.std,
            "config": config.serializable(),
            "parameters": parameters,
            "best_epoch": best_epoch,
            "best_valid_accuracy": best_accuracy,
        },
        checkpoint_path,
    )

    report: dict[str, object] = {
        "experiment": "CoAtNet-Micro 512x512",
        "parameters": parameters,
        "best_epoch": best_epoch,
        "best_valid_accuracy": best_accuracy,
        "training_seconds": training_seconds,
        "peak_cuda_allocated_gb": peak_allocated,
        "peak_cuda_reserved_gb": peak_reserved,
        "effective_batch_size": config.effective_batch_size,
        "countries": data.countries,
        "config": config.serializable(),
        "normalization_mean": data.mean.tolist(),
        "normalization_std": data.std.tolist(),
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

    print("\n" + "=" * 72)
    print("FINAL RESULTS — CoAtNet-Micro 512x512")
    print("=" * 72)
    print(f"Parameters: {parameters:,} / {MAX_PARAMETERS:,}")
    print(f"Best epoch: {best_epoch}/{config.epochs}")
    print(f"Training time: {training_seconds:.1f}s ({training_seconds / 3600:.2f}h)")
    print(f"Peak CUDA allocated: {peak_allocated:.2f} GB")
    print(f"Peak CUDA reserved: {peak_reserved:.2f} GB")
    print(
        f"Train: loss={train_metrics.loss:.4f} "
        f"accuracy={train_metrics.accuracy:.4f} "
        f"({train_metrics.correct}/{train_metrics.total})"
    )
    print(
        f"Valid: loss={valid_metrics.loss:.4f} "
        f"accuracy={valid_metrics.accuracy:.4f} "
        f"({valid_metrics.correct}/{valid_metrics.total})"
    )
    print(
        f"Test:  loss={test_metrics.loss:.4f} "
        f"accuracy={test_metrics.accuracy:.4f} "
        f"({test_metrics.correct}/{test_metrics.total})"
    )
    print("\nPer-country validation/test accuracy:")
    for country in data.countries:
        print(
            f"  {country:20s} "
            f"valid={valid_metrics.per_country_accuracy[country]:.4f} "
            f"test={test_metrics.per_country_accuracy[country]:.4f}"
        )
    print_confusion_matrix(test_metrics.confusion_matrix, data.countries)
    print(f"\nCheckpoint: {checkpoint_path}")
    print(f"JSON results: {results_json}")
    print(f"Markdown results: {results_markdown}")


if __name__ == "__main__":
    main()
