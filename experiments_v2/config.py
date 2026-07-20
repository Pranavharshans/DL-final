"""Single source of truth for comparable v2 experiments."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True)
class ExperimentConfig:
    data_root: Path = Path(__file__).resolve().parents[1]
    image_size: int = 256
    train_cache_size: int = 256
    batch_size: int = 32
    epochs: int = 50
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    label_smoothing: float = 0.08
    warmup_epochs: int = 3
    num_workers: int = 0
    seed: int = 42
    max_parameters: int = 5_000_000

    def to_dict(self) -> dict[str, object]:
        values = asdict(self)
        values["data_root"] = str(self.data_root)
        return values
