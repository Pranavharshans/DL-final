from __future__ import annotations

import unittest

import torch
from torch import nn

from experiments_coatnet512_large.augmentation import SafeGeoAugment512
from experiments_coatnet512_large.model import CoAtNetLarge4M
from experiments_coatnet512_large.multicrop import evaluate_multicrop, multicrop_views
from experiments_v2.models.common import count_parameters


class CoAtNetLargeTests(unittest.TestCase):
    def test_parameter_contract_and_512_forward(self) -> None:
        model = CoAtNetLarge4M(18).eval()
        self.assertEqual(count_parameters(model), 4_252_785)
        self.assertLessEqual(count_parameters(model), 5_000_000)
        with torch.inference_mode():
            logits = model(torch.randn(1, 3, 512, 512))
        self.assertEqual(tuple(logits.shape), (1, 18))
        self.assertTrue(torch.isfinite(logits).all())

    def test_multicrop_has_full_center_and_four_corners(self) -> None:
        images = torch.randn(2, 3, 512, 512)
        views = multicrop_views(images, crop_size=448)
        self.assertEqual(len(views), 6)
        self.assertTrue(all(tuple(view.shape) == (2, 3, 512, 512) for view in views))
        self.assertTrue(torch.equal(views[0], images))

    def test_safe_augmentation_preserves_shape_and_finite_values(self) -> None:
        augmented = SafeGeoAugment512()(torch.rand(3, 512, 512))
        self.assertEqual(tuple(augmented.shape), (3, 512, 512))
        self.assertTrue(torch.isfinite(augmented).all())

    def test_multicrop_evaluator_returns_complete_metrics(self) -> None:
        model = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(3, 2),
        ).eval()
        loader = [(torch.rand(2, 3, 512, 512), torch.tensor([0, 1]), {})]
        metrics = evaluate_multicrop(
            model, loader, torch.device("cpu"), ["a", "b"], crop_size=448
        )
        self.assertEqual(metrics.total, 2)
        self.assertEqual(len(metrics.confusion_matrix), 2)
        self.assertTrue(0.0 <= metrics.accuracy <= 1.0)


if __name__ == "__main__":
    unittest.main()
