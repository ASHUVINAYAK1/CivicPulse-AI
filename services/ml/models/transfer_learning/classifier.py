"""
CivicPulse AI - Transfer Learning Model Architectures

Supports MobileNetV3-Small, EfficientNet-B0, and ResNet-18 backbones
with custom classification heads for road damage triage.
"""

from typing import Tuple, Dict, Any, Optional
import torch
import torch.nn as nn
from torchvision import models


class RoadDamageClassifier(nn.Module):
    """Transfer learning classifier for road damage identification (D00, D10, D20, D40)."""

    SUPPORTED_BACKBONES = ["mobilenet_v3_small", "efficientnet_b0", "resnet18"]

    def __init__(
        self,
        backbone_name: str = "mobilenet_v3_small",
        num_classes: int = 4,
        pretrained: bool = True,
        dropout_rate: float = 0.2,
    ):
        super().__init__()
        self.backbone_name = backbone_name
        self.num_classes = num_classes

        if backbone_name == "mobilenet_v3_small":
            weights = models.MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
            backbone = models.mobilenet_v3_small(weights=weights)
            in_features = backbone.classifier[0].in_features
            # Replace final classifier head
            backbone.classifier = nn.Sequential(
                nn.Linear(in_features, 1024),
                nn.Hardswish(),
                nn.Dropout(p=dropout_rate),
                nn.Linear(1024, num_classes),
            )
            self.model = backbone

        elif backbone_name == "efficientnet_b0":
            weights = models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
            backbone = models.efficientnet_b0(weights=weights)
            in_features = backbone.classifier[1].in_features
            backbone.classifier = nn.Sequential(
                nn.Dropout(p=dropout_rate),
                nn.Linear(in_features, num_classes),
            )
            self.model = backbone

        elif backbone_name == "resnet18":
            weights = models.ResNet18_Weights.DEFAULT if pretrained else None
            backbone = models.resnet18(weights=weights)
            in_features = backbone.fc.in_features
            backbone.fc = nn.Sequential(
                nn.Dropout(p=dropout_rate),
                nn.Linear(in_features, num_classes),
            )
            self.model = backbone

        else:
            raise ValueError(f"Unsupported backbone '{backbone_name}'. Choose from: {self.SUPPORTED_BACKBONES}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def freeze_backbone(self) -> None:
        """Freezes all feature extractor layers for initial head training."""
        for name, param in self.model.named_parameters():
            if "classifier" not in name and "fc" not in name:
                param.requires_grad = False

    def unfreeze_backbone(self) -> None:
        """Unfreezes all layers for fine-tuning."""
        for param in self.model.parameters():
            param.requires_grad = True

