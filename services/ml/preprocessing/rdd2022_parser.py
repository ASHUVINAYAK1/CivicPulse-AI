"""
CivicPulse AI - RDD2022 Road Damage Dataset Parser & Preparation Suite

Provides Pascal VOC XML annotation parsing, bounding-box extraction,
dataset validation, duplicate frame detection, class distribution profiling,
and group-based train/val/test splitting to prevent spatial leakage.
"""

from dataclasses import dataclass, field, asdict
from typing import List, Dict, Tuple, Optional, Any, Set
from pathlib import Path
import xml.etree.ElementTree as ET
import hashlib
import json
from PIL import Image


# Canonical RDD2022 Class Codes mapped to domain categories
RDD2022_CLASSES = {
    "D00": "Longitudinal Crack",
    "D10": "Transverse Crack",
    "D20": "Alligator Crack",
    "D40": "Pothole",
}

VALID_CLASS_NAMES = set(RDD2022_CLASSES.keys())


@dataclass
class BoundingBox:
    """Represents a single bounding box annotation."""
    label: str
    xmin: int
    ymin: int
    xmax: int
    ymax: int
    confidence: Optional[float] = 1.0

    @property
    def width(self) -> int:
        return max(0, self.xmax - self.xmin)

    @property
    def height(self) -> int:
        return max(0, self.ymax - self.ymin)

    @property
    def area(self) -> int:
        return self.width * self.height

    def to_yolo(self, img_width: int, img_height: int) -> Tuple[float, float, float, float]:
        """Converts box to normalized (center_x, center_y, width, height)."""
        x_c = ((self.xmin + self.xmax) / 2.0) / img_width
        y_c = ((self.ymin + self.ymax) / 2.0) / img_height
        w = self.width / img_width
        h = self.height / img_height
        return (
            max(0.0, min(1.0, x_c)),
            max(0.0, min(1.0, y_c)),
            max(0.0, min(1.0, w)),
            max(0.0, min(1.0, h)),
        )


@dataclass
class RoadDamageSample:
    """Represents a road damage image and its associated annotations."""
    image_id: str
    image_path: str
    width: int
    height: int
    depth: int = 3
    boxes: List[BoundingBox] = field(default_factory=list)
    group_id: str = "default_group"
    file_hash: Optional[str] = None

    @property
    def damage_types(self) -> List[str]:
        return sorted(list({b.label for b in self.boxes}))

    @property
    def is_multi_label(self) -> bool:
        return len(self.damage_types) > 1

    @property
    def has_damage(self) -> bool:
        return len(self.boxes) > 0


class RDDAnnotationParser:
    """Parses Pascal VOC XML and YOLO TXT files from RDD2022 distributions."""

    YOLO_CLASS_MAP = {
        0: "D00",
        1: "D10",
        2: "D20",
        3: "D40",
    }

    @staticmethod
    def parse_yolo_file(txt_path: Path, image_path: Path, img_width: int = 720, img_height: int = 720) -> RoadDamageSample:
        """Parses a YOLO format text file (<class_id> <x_c> <y_c> <w> <h>) into a RoadDamageSample."""
        boxes: List[BoundingBox] = []
        if txt_path.exists():
            content = txt_path.read_text().strip()
            if content:
                for line in content.splitlines():
                    tokens = line.strip().split()
                    if len(tokens) >= 5:
                        cls_idx = int(tokens[0])
                        label = RDDAnnotationParser.YOLO_CLASS_MAP.get(cls_idx, f"C{cls_idx}")
                        x_c = float(tokens[1]) * img_width
                        y_c = float(tokens[2]) * img_height
                        w = float(tokens[3]) * img_width
                        h = float(tokens[4]) * img_height

                        xmin = max(0, int(x_c - w / 2))
                        ymin = max(0, int(y_c - h / 2))
                        xmax = min(img_width, int(x_c + w / 2))
                        ymax = min(img_height, int(y_c + h / 2))

                        if xmin < xmax and ymin < ymax:
                            boxes.append(BoundingBox(label=label, xmin=xmin, ymin=ymin, xmax=xmax, ymax=ymax))

        parts = image_path.stem.split("_")
        group_id = parts[0] if len(parts) > 1 else "seq_default"

        return RoadDamageSample(
            image_id=image_path.stem,
            image_path=str(image_path),
            width=img_width,
            height=img_height,
            boxes=boxes,
            group_id=group_id,
        )

    @staticmethod
    def load_yolo_split(split_dir: Path) -> List[RoadDamageSample]:
        """Loads all samples from a split directory with images/ and labels/."""
        img_dir = split_dir / "images"
        lbl_dir = split_dir / "labels"
        samples = []
        for img_path in sorted(list(img_dir.glob("*.jpg"))):
            txt_path = lbl_dir / f"{img_path.stem}.txt"
            samples.append(RDDAnnotationParser.parse_yolo_file(txt_path, img_path))
        return samples

    @staticmethod
    def parse_xml(xml_path: Path, image_dir: Optional[Path] = None) -> RoadDamageSample:
        """Parses a single VOC XML file into a RoadDamageSample."""
        tree = ET.parse(xml_path)
        root = tree.getroot()

        filename = root.findtext("filename", default=f"{xml_path.stem}.jpg")
        size_node = root.find("size")
        width = int(size_node.findtext("width", default="0")) if size_node is not None else 0
        height = int(size_node.findtext("height", default="0")) if size_node is not None else 0
        depth = int(size_node.findtext("depth", default="3")) if size_node is not None else 3

        boxes: List[BoundingBox] = []
        for obj in root.findall("object"):
            name = obj.findtext("name", default="").strip()
            if name in VALID_CLASS_NAMES:
                bndbox = obj.find("bndbox")
                if bndbox is not None:
                    xmin = int(float(bndbox.findtext("xmin", default="0")))
                    ymin = int(float(bndbox.findtext("ymin", default="0")))
                    xmax = int(float(bndbox.findtext("xmax", default="0")))
                    ymax = int(float(bndbox.findtext("ymax", default="0")))

                    # Sanity check: ensure valid coordinate ordering
                    if xmin < xmax and ymin < ymax:
                        boxes.append(BoundingBox(label=name, xmin=xmin, ymin=ymin, xmax=xmax, ymax=ymax))

        # Determine sequence/road group identifier to prevent cross-split leakage
        # (e.g. 'India_001' from 'India_001234.xml')
        parts = xml_path.stem.split("_")
        group_id = parts[0] if len(parts) > 1 else "seq_default"

        img_path_str = str(image_dir / filename) if image_dir else str(xml_path.with_suffix(".jpg"))

        return RoadDamageSample(
            image_id=xml_path.stem,
            image_path=img_path_str,
            width=width,
            height=height,
            depth=depth,
            boxes=boxes,
            group_id=group_id,
        )


class DatasetValidator:
    """Validates image decodability, dimension consistency, and duplicate detection."""

    @staticmethod
    def compute_sha256(file_path: Path) -> str:
        """Computes SHA-256 hash of image file for exact duplicate elimination."""
        hasher = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(65536), b""):
                hasher.update(chunk)
        return hasher.hexdigest()

    @staticmethod
    def validate_sample(sample: RoadDamageSample) -> Tuple[bool, Optional[str]]:
        """Verifies image file decodability and box coordinates."""
        img_path = Path(sample.image_path)
        if not img_path.exists():
            return False, f"Image file not found: {img_path}"

        try:
            with Image.open(img_path) as img:
                real_width, real_height = img.size
                # If XML had missing size, update from image
                if sample.width == 0 or sample.height == 0:
                    sample.width = real_width
                    sample.height = real_height

                # Check box containment
                for b in sample.boxes:
                    if b.xmax > real_width + 10 or b.ymax > real_height + 10:
                        return False, f"Bounding box exceeds image bounds: {b} in {img.size}"

                sample.file_hash = DatasetValidator.compute_sha256(img_path)
                return True, None
        except Exception as e:
            return False, f"Corrupted or invalid image: {str(e)}"


class RDDDataSplitter:
    """Group-based dataset splitter ensuring frames from same inspection road do not leak."""

    @staticmethod
    def split_by_group(
        samples: List[RoadDamageSample],
        train_ratio: float = 0.70,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15,
        seed: int = 42,
    ) -> Dict[str, List[RoadDamageSample]]:
        import random

        rng = random.Random(seed)

        # Group samples by group_id
        groups: Dict[str, List[RoadDamageSample]] = {}
        for s in samples:
            groups.setdefault(s.group_id, []).append(s)

        group_keys = list(groups.keys())
        rng.shuffle(group_keys)

        total_samples = len(samples)
        target_train = int(total_samples * train_ratio)
        target_val = int(total_samples * val_ratio)

        train_set: List[RoadDamageSample] = []
        val_set: List[RoadDamageSample] = []
        test_set: List[RoadDamageSample] = []

        for gk in group_keys:
            grp_samples = groups[gk]
            if len(train_set) < target_train:
                train_set.extend(grp_samples)
            elif len(val_set) < target_val:
                val_set.extend(grp_samples)
            else:
                test_set.extend(grp_samples)

        return {
            "train": train_set,
            "val": val_set,
            "test": test_set,
        }

