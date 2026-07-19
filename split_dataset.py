"""Create a deterministic, stratified train/validation/test dataset split."""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import shutil
import subprocess
from collections import Counter, defaultdict
from pathlib import Path


DEFAULT_SEED = 42
SPLIT_COUNTS = {"train": 420, "valid": 120, "test": 60}


def clone_or_copy(source: Path, destination: Path) -> None:
    """Clone a file on supported filesystems, otherwise make a regular copy."""
    if shutil.which("cp"):
        result = subprocess.run(
            ["cp", "-c", "-p", str(source), str(destination)],
            capture_output=True,
            check=False,
        )
        if result.returncode == 0:
            return
    shutil.copy2(source, destination)


def read_rows(labels_path: Path) -> tuple[list[dict[str, str]], list[str]]:
    with labels_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"No CSV header found in {labels_path}")
        required = {"filename", "country"}
        missing = required.difference(reader.fieldnames)
        if missing:
            raise ValueError(f"Missing required CSV columns: {sorted(missing)}")
        return list(reader), list(reader.fieldnames)


def create_split(
    source_root: Path,
    output_root: Path,
    seed: int = DEFAULT_SEED,
) -> None:
    source_images = source_root / "train"
    labels_path = source_root / "train_labels.csv"
    temporary_root = output_root.with_name(f"{output_root.name}.tmp")

    if output_root.exists() or temporary_root.exists():
        raise FileExistsError(
            f"Refusing to overwrite an existing split: {output_root} or {temporary_root}"
        )
    if not source_images.is_dir() or not labels_path.is_file():
        raise FileNotFoundError("Expected source train/ and train_labels.csv")

    rows, fieldnames = read_rows(labels_path)
    rows_by_country: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        rows_by_country[row["country"]].append(row)

    expected_per_country = sum(SPLIT_COUNTS.values())
    invalid_counts = {
        country: len(country_rows)
        for country, country_rows in rows_by_country.items()
        if len(country_rows) != expected_per_country
    }
    if invalid_counts:
        raise ValueError(
            f"Expected {expected_per_country} samples per country; got {invalid_counts}"
        )

    rng = random.Random(seed)
    split_rows: dict[str, list[dict[str, str]]] = {
        split_name: [] for split_name in SPLIT_COUNTS
    }
    for country in sorted(rows_by_country):
        country_rows = rows_by_country[country][:]
        rng.shuffle(country_rows)
        start = 0
        for split_name, count in SPLIT_COUNTS.items():
            split_rows[split_name].extend(country_rows[start : start + count])
            start += count

    for rows_for_split in split_rows.values():
        rng.shuffle(rows_for_split)

    try:
        temporary_root.mkdir(parents=True)
        for split_name, rows_for_split in split_rows.items():
            split_dir = temporary_root / split_name
            split_dir.mkdir()
            with (split_dir / "labels.csv").open(
                "w", newline="", encoding="utf-8"
            ) as handle:
                writer = csv.DictWriter(handle, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(rows_for_split)

            for row in rows_for_split:
                filename = row["filename"]
                source = source_images / filename
                if not source.is_file():
                    raise FileNotFoundError(f"Missing source image: {source}")
                clone_or_copy(source, split_dir / filename)

        manifest = {
            "source": str(source_root.resolve()),
            "seed": seed,
            "ratios": {"train": 0.70, "valid": 0.20, "test": 0.10},
            "counts": {name: len(values) for name, values in split_rows.items()},
            "countries": len(rows_by_country),
            "samples_per_country": {
                name: count for name, count in SPLIT_COUNTS.items()
            },
        }
        with (temporary_root / "split_manifest.json").open(
            "w", encoding="utf-8"
        ) as handle:
            json.dump(manifest, handle, indent=2)
            handle.write("\n")

        verify_split(temporary_root, rows, rows_by_country)
        os.rename(temporary_root, output_root)
    except Exception:
        shutil.rmtree(temporary_root, ignore_errors=True)
        raise


def verify_split(
    split_root: Path,
    source_rows: list[dict[str, str]],
    rows_by_country: dict[str, list[dict[str, str]]],
) -> None:
    source_filenames = {row["filename"] for row in source_rows}
    seen_filenames: set[str] = set()

    for split_name, expected_per_country in SPLIT_COUNTS.items():
        split_dir = split_root / split_name
        split_rows, _ = read_rows(split_dir / "labels.csv")
        filenames = [row["filename"] for row in split_rows]
        image_filenames = {
            path.name for path in split_dir.glob("*.jpg") if path.is_file()
        }
        expected_total = expected_per_country * len(rows_by_country)
        country_counts = Counter(row["country"] for row in split_rows)

        if len(split_rows) != expected_total:
            raise ValueError(f"Wrong row count in {split_name}: {len(split_rows)}")
        if len(filenames) != len(set(filenames)):
            raise ValueError(f"Duplicate filenames within {split_name}")
        if set(filenames) != image_filenames:
            raise ValueError(f"Image/label mismatch in {split_name}")
        if set(country_counts.values()) != {expected_per_country}:
            raise ValueError(f"Unbalanced countries in {split_name}: {country_counts}")
        if seen_filenames.intersection(filenames):
            raise ValueError(f"Filename overlap involving {split_name}")
        seen_filenames.update(filenames)

    if seen_filenames != source_filenames:
        raise ValueError("The split does not cover the complete labeled dataset")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        type=Path,
        default=Path(__file__).resolve().parent / "geo_dataset",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parent / "geo_dataset_split",
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    create_split(arguments.source, arguments.output, arguments.seed)
    print(f"Created and verified split at {arguments.output}")
