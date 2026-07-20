#!/usr/bin/env python3
"""Train the ten v2 models using one controlled protocol."""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

import torch

from experiments_v2.config import ExperimentConfig
from experiments_v2.data import create_dataloaders
from experiments_v2.engine import save_json, seed_everything, select_device, train_model
from experiments_v2.models import MODEL_REGISTRY, build_model
from experiments_v2.models.common import count_parameters


ROOT = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        choices=list(MODEL_REGISTRY),
        help="Train one model instead of all ten",
    )
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--data-root", type=Path, default=ROOT.parent)
    parser.add_argument(
        "--resume-results",
        action="store_true",
        help="Skip models already present in results.json",
    )
    return parser.parse_args()


def write_markdown(results: list[dict[str, object]], path: Path) -> None:
    ranked = sorted(
        results, key=lambda result: float(result["valid_accuracy"]), reverse=True
    )
    lines = [
        "# V2 Model Results",
        "",
        "All rankings use validation accuracy only. The test split is not read by this runner.",
        "",
        "| Rank | Model | Parameters | Best epoch | Valid accuracy | Time |",
        "|---:|---|---:|---:|---:|---:|",
    ]
    for rank, result in enumerate(ranked, start=1):
        lines.append(
            f"| {rank} | {result['model']} | {int(result['parameters']):,} | "
            f"{int(result['best_epoch'])} | {float(result['valid_accuracy']):.4f} | "
            f"{float(result['training_seconds']):.0f}s |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    config = replace(
        ExperimentConfig(),
        data_root=args.data_root.resolve(),
        epochs=args.epochs,
        batch_size=args.batch_size,
    )
    seed_everything(config.seed)
    device = select_device()
    print(f"Device: {device}")
    print("Pre-caching immutable dataset splits...")
    data = create_dataloaders(config)
    print(f"Training normalization: mean={data.mean.tolist()} std={data.std.tolist()}")

    results_path = ROOT / "results.json"
    results: list[dict[str, object]] = []
    if results_path.exists():
        import json

        results = json.loads(results_path.read_text(encoding="utf-8"))
    completed = {str(result["model"]) for result in results}
    names = [args.model] if args.model else list(MODEL_REGISTRY)

    for name in names:
        if name in completed:
            if args.resume_results:
                print(f"Skipping completed model: {name}")
                continue
            results = [result for result in results if result["model"] != name]
        seed_everything(config.seed)
        model = build_model(name, len(data.countries))
        parameters = count_parameters(model)
        if parameters > config.max_parameters:
            raise ValueError(
                f"{name} has {parameters:,} parameters; limit is {config.max_parameters:,}"
            )
        model = model.to(device)
        result = train_model(
            name,
            model,
            data.train,
            data.valid,
            data.countries,
            config,
            ROOT / "checkpoints" / f"{name}.pt",
            device,
        )
        result["parameters"] = parameters
        results.append(result)
        save_json(results, results_path)
        write_markdown(results, ROOT / "results.md")
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
