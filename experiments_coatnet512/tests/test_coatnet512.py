from __future__ import annotations

import unittest
from pathlib import Path

import torch

from experiments_coatnet512.train import RunConfig, cosine_with_warmup
from experiments_v2.models.coatnet import CoAtNetMicro
from experiments_v2.models.common import count_parameters


class CoAtNet512Tests(unittest.TestCase):
    def test_effective_batch_size(self) -> None:
        config = RunConfig(
            Path("."), Path("."), micro_batch_size=8, accumulation_steps=4
        )
        self.assertEqual(config.effective_batch_size, 32)

    def test_scheduler_starts_with_warmup_and_ends_at_zero(self) -> None:
        self.assertAlmostEqual(cosine_with_warmup(0, 3, 50), 1 / 3)
        self.assertAlmostEqual(cosine_with_warmup(2, 3, 50), 1.0)
        self.assertAlmostEqual(cosine_with_warmup(49, 3, 50), 0.0)

    def test_model_contract_at_512(self) -> None:
        model = CoAtNetMicro(18).eval()
        self.assertLessEqual(count_parameters(model), 5_000_000)
        with torch.inference_mode():
            logits = model(torch.randn(1, 3, 512, 512))
        self.assertEqual(tuple(logits.shape), (1, 18))
        self.assertTrue(torch.isfinite(logits).all())


if __name__ == "__main__":
    unittest.main()
