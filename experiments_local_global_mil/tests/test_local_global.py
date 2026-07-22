import torch

from experiments_local_global_mil.model import LocalGlobalMIL
from experiments_local_global_mil.views import FiveViewDataset
from experiments_v2.models.common import count_parameters


class TinyDataset:
    def __len__(self):
        return 1

    def __getitem__(self, index):
        return torch.rand(3, 512, 512), torch.tensor(2), {}


def test_parameter_limit_and_forward():
    model = LocalGlobalMIL()
    assert 4_000_000 < count_parameters(model) < 5_000_000
    with torch.inference_mode():
        output = model(torch.randn(1, 5, 3, 256, 256))
    assert output.shape == (1, 18)


def test_five_views_are_constructed():
    views, label, _ = FiveViewDataset(TinyDataset(), training=False)[0]
    assert views.shape == (5, 3, 256, 256)
    assert label.item() == 2
