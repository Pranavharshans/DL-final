"""Unit tests for augmentation and metadata target construction."""

from __future__ import annotations

import unittest

import torch

from experiments_v2.augmentation import GeoAugmentV1
from experiments_v2.data import coordinate_targets


class DataContractTests(unittest.TestCase):
    def test_augmentation_preserves_shape_and_range(self) -> None:
        torch.manual_seed(42)
        image = torch.rand(3, 288, 288)
        augmented = GeoAugmentV1(256)(image)
        self.assertEqual(tuple(augmented.shape), (3, 256, 256))
        self.assertGreaterEqual(augmented.min().item(), 0.0)
        self.assertLessEqual(augmented.max().item(), 1.0)

    def test_coordinate_targets_are_bounded(self) -> None:
        targets = coordinate_targets(51.0, 7.0)
        self.assertIn(targets["latitude_band"].item(), range(12))
        self.assertIn(targets["longitude_band"].item(), range(24))
        self.assertIn(targets["hemisphere"].item(), range(4))
        self.assertAlmostEqual(targets["coordinate"].norm().item(), 1.0, places=5)


if __name__ == "__main__":
    unittest.main()
