"""Shared training and evaluation engine for every v2 architecture."""

from __future__ import annotations

import copy
import json
import math
import random
import time
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader

from .config import ExperimentConfig
from .models.metric import CountryMetricNet
from .models.res2net import GeoAuxiliaryNet


@dataclass
class EpochMetrics:
    loss: float
    accuracy: float
    per_country_correct: list[int]
    per_country_total: list[int]

    def per_country_accuracy(self, countries: list[str]) -> dict[str, float]:
        return {
            country: self.per_country_correct[index]
            / max(1, self.per_country_total[index])
            for index, country in enumerate(countries)
        }


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def select_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _autocast(device: torch.device):
    if device.type == "cuda":
        return torch.cuda.amp.autocast()
    return nullcontext()


def _move_metadata(
    metadata: dict[str, Tensor], device: torch.device
) -> dict[str, Tensor]:
    return {
        name: value.to(device, non_blocking=True) for name, value in metadata.items()
    }


def _forward_and_loss(
    model: nn.Module,
    images: Tensor,
    labels: Tensor,
    metadata: dict[str, Tensor],
    criterion: nn.CrossEntropyLoss,
    training: bool,
) -> tuple[Tensor, Tensor]:
    if isinstance(model, GeoAuxiliaryNet) and training:
        outputs = model.forward_with_aux(images)
        country_loss = criterion(outputs["country"], labels)
        latitude_loss = nn.functional.cross_entropy(
            outputs["latitude_band"], metadata["latitude_band"]
        )
        longitude_loss = nn.functional.cross_entropy(
            outputs["longitude_band"], metadata["longitude_band"]
        )
        hemisphere_loss = nn.functional.cross_entropy(
            outputs["hemisphere"], metadata["hemisphere"]
        )
        coordinate_loss = nn.functional.smooth_l1_loss(
            outputs["coordinate"], metadata["coordinate"]
        )
        loss = (
            country_loss
            + 0.10 * latitude_loss
            + 0.10 * longitude_loss
            + 0.05 * hemisphere_loss
            + 0.10 * coordinate_loss
        )
        return outputs["country"], loss
    if isinstance(model, CountryMetricNet):
        logits = model(images, labels if training else None)
    else:
        logits = model(images)
    return logits, criterion(logits, labels)


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.CrossEntropyLoss,
    device: torch.device,
    num_classes: int,
    optimizer: torch.optim.Optimizer | None = None,
    scaler: torch.cuda.amp.GradScaler | None = None,
) -> EpochMetrics:
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    per_country_correct = [0] * num_classes
    per_country_total = [0] * num_classes

    context = nullcontext() if training else torch.no_grad()
    with context:
        for images, labels, metadata in loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            metadata = _move_metadata(metadata, device)
            if training:
                optimizer.zero_grad(set_to_none=True)
            with _autocast(device):
                logits, loss = _forward_and_loss(
                    model, images, labels, metadata, criterion, training
                )
            if training:
                if scaler is not None and scaler.is_enabled():
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()

            predictions = logits.argmax(dim=1)
            batch_size = labels.shape[0]
            total_loss += loss.detach().item() * batch_size
            total_correct += predictions.eq(labels).sum().item()
            total_samples += batch_size
            for class_index in range(num_classes):
                mask = labels == class_index
                per_country_total[class_index] += mask.sum().item()
                per_country_correct[class_index] += (
                    predictions[mask].eq(labels[mask]).sum().item()
                )

    if total_samples == 0:
        raise RuntimeError("cannot evaluate an empty dataloader")
    return EpochMetrics(
        loss=total_loss / total_samples,
        accuracy=total_correct / total_samples,
        per_country_correct=per_country_correct,
        per_country_total=per_country_total,
    )


def cosine_with_warmup(epoch: int, warmup_epochs: int, total_epochs: int) -> float:
    if epoch < warmup_epochs:
        return (epoch + 1) / max(1, warmup_epochs)
    progress = (epoch - warmup_epochs) / max(1, total_epochs - warmup_epochs - 1)
    return 0.5 * (1.0 + math.cos(math.pi * progress))


def train_model(
    model_name: str,
    model: nn.Module,
    train_loader: DataLoader,
    valid_loader: DataLoader,
    countries: list[str],
    config: ExperimentConfig,
    checkpoint_path: Path,
    device: torch.device,
) -> dict[str, object]:
    criterion = nn.CrossEntropyLoss(label_smoothing=config.label_smoothing)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
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
    start = time.monotonic()

    for epoch in range(1, config.epochs + 1):
        train_metrics = run_epoch(
            model,
            train_loader,
            criterion,
            device,
            len(countries),
            optimizer,
            scaler,
        )
        valid_metrics = run_epoch(
            model, valid_loader, criterion, device, len(countries)
        )
        scheduler.step()
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_metrics.loss,
                "train_accuracy": train_metrics.accuracy,
                "valid_loss": valid_metrics.loss,
                "valid_accuracy": valid_metrics.accuracy,
                "learning_rate": optimizer.param_groups[0]["lr"],
            }
        )
        if valid_metrics.accuracy > best_accuracy:
            best_accuracy = valid_metrics.accuracy
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
        print(
            f"{model_name} epoch {epoch:02d}/{config.epochs} "
            f"train={train_metrics.accuracy:.4f} valid={valid_metrics.accuracy:.4f}",
            flush=True,
        )

    if best_state is None:
        raise RuntimeError("training completed without a checkpoint")
    model.load_state_dict(best_state)
    final_valid = run_epoch(model, valid_loader, criterion, device, len(countries))
    elapsed = time.monotonic() - start
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_name": model_name,
            "model_state": best_state,
            "countries": countries,
            "config": config.to_dict(),
            "best_epoch": best_epoch,
            "best_valid_accuracy": best_accuracy,
        },
        checkpoint_path,
    )
    return {
        "model": model_name,
        "best_epoch": best_epoch,
        "valid_accuracy": final_valid.accuracy,
        "valid_loss": final_valid.loss,
        "per_country_accuracy": final_valid.per_country_accuracy(countries),
        "training_seconds": elapsed,
        "history": history,
        "checkpoint": str(checkpoint_path),
    }


def save_json(payload: object, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")
