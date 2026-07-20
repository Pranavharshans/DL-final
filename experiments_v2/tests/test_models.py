"""Architecture contract tests; run with ``python -m unittest``."""

from __future__ import annotations

import unittest

import torch

from experiments_v2.models import MODEL_REGISTRY, build_model
from experiments_v2.models.common import count_parameters
from experiments_v2.models.res2net import GeoAuxiliaryNet


class ModelContractTests(unittest.TestCase):
    def test_registry_contains_exactly_ten_models(self) -> None:
        self.assertEqual(len(MODEL_REGISTRY), 10)

    def test_all_models_are_image_only_and_under_parameter_limit(self) -> None:
        images = torch.randn(2, 3, 128, 128)
        for name in MODEL_REGISTRY:
            with self.subTest(model=name):
                model = build_model(name, num_classes=18).eval()
                self.assertLessEqual(count_parameters(model), 5_000_000)
                with torch.no_grad():
                    logits = model(images)
                self.assertEqual(tuple(logits.shape), (2, 18))
                self.assertTrue(torch.isfinite(logits).all())

    def test_geo_auxiliary_outputs_are_training_only_extras(self) -> None:
        model = GeoAuxiliaryNet(18).eval()
        images = torch.randn(2, 3, 128, 128)
        with torch.no_grad():
            logits = model(images)
            auxiliary = model.forward_with_aux(images)
        self.assertEqual(tuple(logits.shape), (2, 18))
        self.assertEqual(tuple(auxiliary["coordinate"].shape), (2, 3))
        self.assertTrue(torch.equal(logits, auxiliary["country"]))

    def test_all_model_parameters_receive_finite_gradients(self) -> None:
        images = torch.randn(2, 3, 128, 128)
        labels = torch.tensor([0, 1])
        for name in MODEL_REGISTRY:
            with self.subTest(model=name):
                model = build_model(name, num_classes=18).train()
                if isinstance(model, GeoAuxiliaryNet):
                    outputs = model.forward_with_aux(images)
                    loss = torch.nn.functional.cross_entropy(outputs["country"], labels)
                    loss += torch.nn.functional.cross_entropy(
                        outputs["latitude_band"], torch.tensor([0, 1])
                    )
                    loss += torch.nn.functional.cross_entropy(
                        outputs["longitude_band"], torch.tensor([0, 1])
                    )
                    loss += torch.nn.functional.cross_entropy(
                        outputs["hemisphere"], torch.tensor([0, 1])
                    )
                    loss += torch.nn.functional.smooth_l1_loss(
                        outputs["coordinate"], torch.randn(2, 3)
                    )
                elif name == "29_CountryMetric-Net":
                    loss = torch.nn.functional.cross_entropy(
                        model(images, labels), labels
                    )
                else:
                    loss = torch.nn.functional.cross_entropy(model(images), labels)
                loss.backward()
                gradients = [
                    parameter.grad
                    for parameter in model.parameters()
                    if parameter.requires_grad
                ]
                self.assertTrue(gradients)
                self.assertTrue(
                    all(
                        gradient is not None and torch.isfinite(gradient).all()
                        for gradient in gradients
                    )
                )


if __name__ == "__main__":
    unittest.main()
