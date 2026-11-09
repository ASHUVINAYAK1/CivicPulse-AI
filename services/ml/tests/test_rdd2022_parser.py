"""
Tests for RDD2022 annotation parser, validator, and group splitter.
"""

from pathlib import Path
import tempfile
import xml.etree.ElementTree as ET
from PIL import Image

from preprocessing.rdd2022_parser import (
    RDDAnnotationParser,
    DatasetValidator,
    RDDDataSplitter,
    BoundingBox,
    RoadDamageSample,
)


def create_sample_voc_xml(file_path: Path, filename: str, boxes: list):
    root = ET.Element("annotation")
    fn = ET.SubElement(root, "filename")
    fn.text = filename
    size = ET.SubElement(root, "size")
    w = ET.SubElement(size, "width")
    w.text = "720"
    h = ET.SubElement(size, "height")
    h.text = "720"
    d = ET.SubElement(size, "depth")
    d.text = "3"

    for label, xmin, ymin, xmax, ymax in boxes:
        obj = ET.SubElement(root, "object")
        name = ET.SubElement(obj, "name")
        name.text = label
        bnd = ET.SubElement(obj, "bndbox")
        for tag, val in [("xmin", xmin), ("ymin", ymin), ("xmax", xmax), ("ymax", ymax)]:
            sub = ET.SubElement(bnd, tag)
            sub.text = str(val)

    tree = ET.ElementTree(root)
    tree.write(file_path)


def test_rdd_xml_parser():
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        xml_file = tmp_path / "India_001234.xml"
        create_sample_voc_xml(
            xml_file,
            "India_001234.jpg",
            [("D40", 100, 150, 250, 300), ("D00", 50, 60, 90, 400)],
        )

        sample = RDDAnnotationParser.parse_xml(xml_file)
        assert sample.image_id == "India_001234"
        assert sample.group_id == "India"
        assert len(sample.boxes) == 2
        assert sample.damage_types == ["D00", "D40"]
        assert sample.is_multi_label is True


def test_bounding_box_yolo_normalization():
    box = BoundingBox(label="D40", xmin=100, ymin=100, xmax=300, ymax=400)
    xc, yc, w, h = box.to_yolo(1000, 1000)
    assert xc == 0.2
    assert yc == 0.25
    assert w == 0.2
    assert h == 0.3


def test_dataset_validator_and_hash():
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        img_file = tmp_path / "test.jpg"
        img = Image.new("RGB", (720, 720), color=(100, 100, 100))
        img.save(img_file)

        sample = RoadDamageSample(
            image_id="test",
            image_path=str(img_file),
            width=720,
            height=720,
            boxes=[BoundingBox(label="D40", xmin=10, ymin=10, xmax=100, ymax=100)],
        )

        is_valid, err = DatasetValidator.validate_sample(sample)
        assert is_valid is True
        assert err is None
        assert sample.file_hash is not None


def test_group_based_splitter():
    samples = [
        RoadDamageSample("India_001", "p1", 720, 720, group_id="India_roadA"),
        RoadDamageSample("India_002", "p2", 720, 720, group_id="India_roadA"),
        RoadDamageSample("India_003", "p3", 720, 720, group_id="India_roadB"),
        RoadDamageSample("India_004", "p4", 720, 720, group_id="India_roadB"),
        RoadDamageSample("Japan_001", "p5", 720, 720, group_id="Japan_roadA"),
        RoadDamageSample("Japan_002", "p6", 720, 720, group_id="Japan_roadB"),
    ]

    splits = RDDDataSplitter.split_by_group(samples, train_ratio=0.5, val_ratio=0.25, test_ratio=0.25)
    assert "train" in splits and "val" in splits and "test" in splits
    total_split_samples = len(splits["train"]) + len(splits["val"]) + len(splits["test"])
    assert total_split_samples == 6

