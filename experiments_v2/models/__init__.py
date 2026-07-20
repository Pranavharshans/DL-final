"""Registry for all ten v2 architectures."""

from __future__ import annotations

from collections.abc import Callable

from torch import nn

from .antialiased import AntiAliasedResNetSK
from .coatnet import CoAtNetMicro
from .hrnet import HRNetLite
from .inception_resnet import InceptionResNetLite
from .maxvit import MaxViTMicro
from .metric import CountryMetricNet
from .multiview import MultiViewCNN
from .res2net import GeoAuxiliaryNet, Res2NetECAGeM
from .spatial_frequency import SpatialFrequencyNet


ModelBuilder = Callable[[int], nn.Module]

MODEL_REGISTRY: dict[str, ModelBuilder] = {
    "21_MultiView-CNN": MultiViewCNN,
    "22_Res2Net-ECA-GeM": Res2NetECAGeM,
    "23_GeoAuxiliary-ImageOnly": GeoAuxiliaryNet,
    "24_HRNet-Lite": HRNetLite,
    "25_CoAtNet-Micro": CoAtNetMicro,
    "26_InceptionResNet-Lite": InceptionResNetLite,
    "27_AntiAliased-ResNet-SK": AntiAliasedResNetSK,
    "28_SpatialFrequency-Net": SpatialFrequencyNet,
    "29_CountryMetric-Net": CountryMetricNet,
    "30_MaxViT-Micro": MaxViTMicro,
}


def build_model(name: str, num_classes: int = 18) -> nn.Module:
    try:
        builder = MODEL_REGISTRY[name]
    except KeyError as error:
        choices = ", ".join(MODEL_REGISTRY)
        raise ValueError(
            f"Unknown model {name!r}. Available models: {choices}"
        ) from error
    return builder(num_classes)


__all__ = ["MODEL_REGISTRY", "build_model"]
