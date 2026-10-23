from __future__ import annotations

import torch
from torch import nn
from torchvision.models import ResNet18_Weights, resnet18


def build_model(num_classes: int = 6) -> nn.Module:
    """Build the mandated architecture with random initialization only."""
    model = resnet18(weights=None, num_classes=num_classes)
    if any(parameter.requires_grad is False for parameter in model.parameters()):
        raise RuntimeError("Scratch model unexpectedly contains frozen parameters")
    return model


def build_feature_model(num_classes: int = 6) -> tuple[nn.Module, nn.Module]:
    model = build_model(num_classes)
    classifier = model.fc
    model.fc = nn.Identity()
    return model, classifier