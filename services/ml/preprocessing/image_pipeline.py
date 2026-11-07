"""
CivicPulse AI - Computer Vision Preprocessing Pipeline

Handles image validation, resizing, aspect-ratio preservation,
normalization (ImageNet standard), and data augmentation.
"""

from typing import Tuple, Optional, Union
import numpy as np
from PIL import Image, ImageOps
import io


# ImageNet baseline statistics for transfer learning
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

TARGET_IMAGE_SIZE = (224, 224)


class ImagePreprocessor:
    """Preprocesses road damage imagery for CNN classification and transfer learning."""

    def __init__(
        self,
        target_size: Tuple[int, int] = TARGET_IMAGE_SIZE,
        normalize: bool = True,
    ):
        self.target_size = target_size
        self.normalize = normalize

    def load_from_bytes(self, image_bytes: bytes) -> Image.Image:
        """Loads and validates a PIL Image from raw bytes."""
        try:
            image = Image.open(io.BytesIO(image_bytes))
            # Handle EXIF orientation tags automatically
            image = ImageOps.exif_transpose(image)
            # Ensure RGB format (strips alpha channels and grayscale)
            return image.convert("RGB")
        except Exception as e:
            raise ValueError(f"Invalid image format or corrupted file: {str(e)}")

    def preprocess_image(
        self,
        image: Union[Image.Image, bytes],
        augment: bool = False,
    ) -> np.ndarray:
        """
        Transforms input PIL image or raw bytes into a normalized tensor.
        Returns:
            np.ndarray of shape (3, H, W) normalized to [0, 1] or ImageNet standardization.
        """
        if isinstance(image, bytes):
            pil_img = self.load_from_bytes(image)
        else:
            pil_img = image.convert("RGB")

        # Resize with high-quality Lanczos resampling
        pil_img = pil_img.resize(self.target_size, Image.Resampling.LANCZOS)

        # Optional data augmentation for training experiments
        if augment:
            if np.random.rand() > 0.5:
                pil_img = pil_img.transpose(Image.Transpose.FLIP_LEFT_RIGHT)

        # Convert to numpy float32 array in [0.0, 1.0]
        img_array = np.asarray(pil_img, dtype=np.float32) / 255.0

        if self.normalize:
            img_array = (img_array - IMAGENET_MEAN) / IMAGENET_STD

        # Transpose from (H, W, C) to (C, H, W) standard tensor format
        return np.transpose(img_array, (2, 0, 1))

    def batch_preprocess(self, images: list[Union[Image.Image, bytes]]) -> np.ndarray:
        """Preprocesses a batch of images into a single array (N, C, H, W)."""
        processed = [self.preprocess_image(img) for img in images]
        return np.stack(processed, axis=0)
