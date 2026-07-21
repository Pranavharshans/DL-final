from __future__ import annotations

import unittest

import torch

from experiments_coatnet512_large.model import CoAtNetLarge4M
from experiments_coatnet512_large_control.train import (
    EXPECTED_PARAMETERS,
    build_augmentation,
)
from experiments_v2.augmentation import GeoAugmentV1
from experiments_v2.models.common import count_parameters


class LargeControlTests(unittest.TestCase):
    def test_exact_model_and_augmentation_contract(self) -> None:
        model = CoAtNetLarge4M(18).eval()
        self.assertEqual(count_parameters(model), EXPECTED_PARAMETERS)
        self.assertIsInstance(build_augmentation(512), GeoAugmentV1)

    def test_512_forward(self) -> None:
        model = CoAtNetLarge4M(18).eval()
        with torch.inference_mode():
            logits = model(torch.randn(1, 3, 512, 512))
        self.assertEqual(tuple(logits.shape), (1, 18))


if __name__ == "__main__":
    unittest.main()
