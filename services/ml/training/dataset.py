"""
CivicPulse AI - PyTorch Dataset & DataLoader Pipelines

Loads road damage samples with ImageNet transformations,
multi-label ground-truth vectors, and bounding-box crops.
"""

from typing import List, Tuple, Optional, Callable
from pathlib import Path
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from PIL import Image

from preprocessing.rdd2022_parser import RoadDamageSample
from preprocessing.image_pipeline import ImagePreprocessor


CLASS_TO_IDX = {"D00": 0, "D10": 1, "D20": 2, "D40": 3}
IDX_TO_CLASS = {0: "D00", 1: "D10", 2: "D20", 3: "D40"}


class RoadDamageDataset(Dataset):
    """PyTorch Dataset for multi-label road damage classification."""

    def __init__(
        self,
        samples: List[RoadDamageSample],
        target_size: Tuple[int, int] = (224, 224),
        augment: bool = False,
        crop_boxes: bool = False,
    ):
        self.samples = samples
        self.preprocessor = ImagePreprocessor(target_size=target_size, normalize=True)
        self.augment = augment
        self.crop_boxes = crop_boxes

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        sample = self.samples[idx]
        img_path = Path(sample.image_path)

        if img_path.exists():
            with Image.open(img_path) as img:
                pil_img = img.convert("RGB")
        else:
            # Fallback synthetic placeholder canvas if raw image is not downloaded
            pil_img = Image.new("RGB", (sample.width or 640, sample.height or 480), color=(128, 128, 128))

        # Optional: Crop around largest bounding box if crop_boxes is enabled
        if self.crop_boxes and sample.boxes:
            box = sample.boxes[0]
            pil_img = pil_img.crop((box.xmin, box.ymin, box.xmax, box.ymax))

        tensor_np = self.preprocessor.preprocess_image(pil_img, augment=self.augment)
        image_tensor = torch.from_numpy(tensor_np)

        # Build multi-hot target vector of length 4: [D00, D10, D20, D40]
        target = torch.zeros(4, dtype=torch.float32)
        for dt in sample.damage_types:
            if dt in CLASS_TO_IDX:
                target[CLASS_TO_IDX[dt]] = 1.0

        return image_tensor, target


def create_dataloader(
    samples: List[RoadDamageSample],
    batch_size: int = 16,
    shuffle: bool = True,
    num_workers: int = 0,
    augment: bool = False,
) -> DataLoader:
    dataset = RoadDamageDataset(samples, augment=augment)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )

