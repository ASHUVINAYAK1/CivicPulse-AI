"""
CivicPulse AI - Road Damage Object Detection Architecture

Implements a lightweight transfer-learning object detector based on
SSDLite with MobileNetV3-Large backbone, customized for RDD2022 damage boxes.
"""

from typing import List, Dict, Any, Tuple, Optional
import torch
import torch.nn as nn
from torchvision.models.detection import ssdlite320_mobilenet_v3_large


class RoadDamageDetector(nn.Module):
    """
    Object detector predicting bounding boxes and damage categories (D00, D10, D20, D40).
    Background class is index 0; damage classes are indices 1..4.
    """

    NUM_CLASSES = 5  # 0: background, 1: D00, 2: D10, 3: D20, 4: D40
    CLASS_MAP = {1: "D00", 2: "D10", 3: "D20", 4: "D40"}

    def __init__(self, pretrained: bool = True, num_classes: int = NUM_CLASSES):
        super().__init__()
        weights_backbone = "DEFAULT" if pretrained else None
        self.model = ssdlite320_mobilenet_v3_large(
            weights=None,
            weights_backbone=weights_backbone,
            num_classes=num_classes,
        )

    def forward(self, images: List[torch.Tensor], targets: Optional[List[Dict[str, torch.Tensor]]] = None):
        """
        In training: returns loss dict (bbox_regression, classification).
        In inference: returns list of dicts with 'boxes', 'scores', 'labels'.
        """
        return self.model(images, targets)
