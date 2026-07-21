"""Deterministic image-only multi-crop inference."""

from __future__ import annotations

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from experiments_coatnet512.train import SplitMetrics, autocast_context


def multicrop_views(images: Tensor, crop_size: int = 448) -> list[Tensor]:
    """Return full, center, and four corner views, each at model resolution."""
    _, _, height, width = images.shape
    if crop_size >= min(height, width):
        raise ValueError("crop_size must be smaller than the input image")
    bottom = height - crop_size
    right = width - crop_size
    center_y = bottom // 2
    center_x = right // 2
    positions = [
        (center_y, center_x),
        (0, 0),
        (0, right),
        (bottom, 0),
        (bottom, right),
    ]
    views = [images]
    for top, left in positions:
        crop = images[:, :, top : top + crop_size, left : left + crop_size]
        views.append(
            F.interpolate(
                crop,
                size=(height, width),
                mode="bilinear",
                align_corners=False,
                antialias=True,
            )
        )
    return views


@torch.inference_mode()
def evaluate_multicrop(
    model: nn.Module,
    loader,
    device: torch.device,
    countries: list[str],
    crop_size: int = 448,
) -> SplitMetrics:
    """Average probabilities across views without increasing peak batch memory."""
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
        probabilities = None
        for view in multicrop_views(images, crop_size):
            with autocast_context(device):
                view_probabilities = model(view).softmax(dim=1)
            probabilities = (
                view_probabilities
                if probabilities is None
                else probabilities + view_probabilities
            )
        if probabilities is None:
            raise RuntimeError("multi-crop evaluation produced no views")
        probabilities = probabilities / 6.0
        loss = F.nll_loss(probabilities.clamp_min(1e-8).log(), labels)
        predictions = probabilities.argmax(dim=1)
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
