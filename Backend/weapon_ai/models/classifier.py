"""Classifier model construction for weapon inference."""

from __future__ import annotations

import torch.nn as nn
from torchvision import models


def build_model(arch: str, num_classes: int = 2) -> nn.Module:
    arch = arch.lower()
    if arch == "resnet18":
        weights = models.ResNet18_Weights.IMAGENET1K_V1
        model = models.resnet18(weights=weights)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        return model
    if arch == "mobilenet_v3_small":
        weights = models.MobileNet_V3_Small_Weights.IMAGENET1K_V1
        model = models.mobilenet_v3_small(weights=weights)
        model.classifier[3] = nn.Linear(model.classifier[3].in_features, num_classes)
        return model
    raise ValueError(f"Unknown arch: {arch}. Use resnet18 or mobilenet_v3_small.")


def build_gun_prob_model(arch: str) -> nn.Module:
    """Single-logit BCE head for P(gun)."""
    return build_model(arch, num_classes=1)
