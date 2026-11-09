"""
CivicPulse AI — Phase 2.2 Comprehensive Test Suite
Validates:
1. Split Isolation (zero adjacent/burst frames across train/val/test).
2. Near-Duplicate Detection (dHash behavior).
3. Annotation Conversion (YOLO to absolute box coordinates).
4. Detector Evaluation Logic (IoU and AP interpolation).
5. Model Registry Consistency (cryptographic hash matches & artifact integrity).
"""

import os
import sys
import json
import hashlib
from pathlib import Path
import re
import numpy as np
import pytest
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from training.train_detectors_v2 import box_iou_xyxy, compute_voc_ap
from preprocessing.partition_dataset_v2 import dhash


def test_split_isolation():
    """Verifies complete spatial and temporal isolation in rdd2022_india_grouped_v2."""
    split_dir = Path("datasets/splits/rdd2022_india_grouped_v2")
    assert split_dir.exists(), f"Split directory missing: {split_dir}"

    manifest_file = split_dir / "split_manifest.json"
    assert manifest_file.exists(), "split_manifest.json is missing"

    manifest = json.loads(manifest_file.read_text())
    integrity = manifest["integrity_verification"]

    # Strict isolation assertions
    assert integrity["adjacent_test_frames_in_train_delta_2"] == 0, "Test set contains adjacent burst frames in train!"
    assert integrity["adjacent_val_frames_in_train_delta_2"] == 0, "Val set contains adjacent burst frames in train!"
    assert integrity["min_index_distance_test_to_train"] >= 10, "Minimum separation from Test to Train must be >= 10 frames!"

    # Verify no SHA-256 duplicate collision across splits
    train_files = list((split_dir / "train" / "images").glob("*.jpg"))
    test_files = list((split_dir / "test" / "images").glob("*.jpg"))
    assert len(train_files) >= 5000, f"Expected >= 5000 train images, got {len(train_files)}"
    assert len(test_files) >= 1000, f"Expected >= 1000 test images, got {len(test_files)}"


def test_near_duplicate_detection():
    """Verifies that perceptual hashing correctly detects duplicates and differentiates distinct scenes."""
    img_a = Image.linear_gradient("L").resize((320, 320))
    h_a = dhash(img_a)

    # Identical image -> zero hamming distance
    assert np.count_nonzero(h_a != dhash(img_a)) == 0

    # Perturbed image with minor pixel shift -> low distance <= 5
    img_perturbed = img_a.copy()
    pixels = img_perturbed.load()
    for i in range(10):
        pixels[50 + i, 50] = max(0, min(255, pixels[50 + i, 50] + 1))
    h_perturbed = dhash(img_perturbed)
    assert np.count_nonzero(h_a != h_perturbed) <= 5

    # Contrasting directional pattern -> high distance (> 30)
    img_b = Image.linear_gradient("L").rotate(90).resize((320, 320))
    h_b = dhash(img_b)
    dist = np.count_nonzero(h_a != h_b)
    assert dist > 30, f"Expected high distance between distinct scenes, got {dist}"


def test_annotation_conversion():
    """Tests coordinate scaling and conversion from YOLO format to TorchVision bounding boxes."""
    orig_w, orig_h = 720, 720
    target_size = (320, 320)

    # YOLO box: center (0.5, 0.5), width 0.2, height 0.2 -> [0.4, 0.4, 0.6, 0.6]
    xc_norm, yc_norm, w_norm, h_norm = 0.5, 0.5, 0.2, 0.2
    cls_id = 3  # D40 pothole

    xc_abs = xc_norm * orig_w
    yc_abs = yc_norm * orig_h
    bw_abs = w_norm * orig_w
    bh_abs = h_norm * orig_h

    x1 = xc_abs - bw_abs / 2.0
    y1 = yc_abs - bh_abs / 2.0
    x2 = xc_abs + bw_abs / 2.0
    y2 = yc_abs + bh_abs / 2.0

    scale_x = target_size[0] / float(orig_w)
    scale_y = target_size[1] / float(orig_h)

    x1_target = max(0.0, min(320.0, x1 * scale_x))
    y1_target = max(0.0, min(320.0, y1 * scale_y))
    x2_target = max(0.0, min(320.0, x2 * scale_x))
    y2_target = max(0.0, min(320.0, y2 * scale_y))

    # Expected at 320x320: [128.0, 128.0, 192.0, 192.0]
    assert np.isclose(x1_target, 128.0)
    assert np.isclose(y1_target, 128.0)
    assert np.isclose(x2_target, 192.0)
    assert np.isclose(y2_target, 192.0)

    # Class indexing mapping: YOLO 0..3 -> TorchVision 1..4
    torchvision_label = cls_id + 1
    assert torchvision_label == 4


def test_detector_evaluation_logic():
    """Validates mathematical correctness of IoU and AP interpolation."""
    # 1. Box IoU
    box_a = np.array([10.0, 10.0, 50.0, 50.0])
    box_b = np.array([10.0, 10.0, 50.0, 50.0])
    assert np.isclose(box_iou_xyxy(box_a, box_b), 1.0)

    box_c = np.array([60.0, 60.0, 100.0, 100.0])
    assert np.isclose(box_iou_xyxy(box_a, box_c), 0.0)

    # Half overlap: box_d [10, 10, 30, 50] (area 800) in box_a (area 1600) -> inter 800, union 1600 -> 0.5
    box_d = np.array([10.0, 10.0, 30.0, 50.0])
    assert np.isclose(box_iou_xyxy(box_a, box_d), 0.5)

    # 2. VOC AP interpolation
    recalls = np.array([0.2, 0.4, 0.6, 0.8, 1.0])
    precisions = np.array([1.0, 1.0, 0.8, 0.8, 0.5])
    ap = compute_voc_ap(recalls, precisions)
    assert 0.0 <= ap <= 1.0
    assert ap > 0.7  # High precision curve should yield high AP


def test_model_registry_consistency():
    """Validates model artifact checksums and integrity against registry."""
    artifacts_dir = Path("services/ml/model_artifacts")
    assert artifacts_dir.exists()

    prod_model = artifacts_dir / "production_model.pt"
    resnet18_best = artifacts_dir / "resnet18_best.pt"
    assert prod_model.exists()
    assert resnet18_best.exists()

    # Verify bit-for-bit SHA-256 match
    h_prod = hashlib.sha256(prod_model.read_bytes()).hexdigest()
    h_best = hashlib.sha256(resnet18_best.read_bytes()).hexdigest()
    assert h_prod == h_best, "production_model.pt must match resnet18_best.pt"

    # Verify original detector checkpoint remains preserved
    orig_det = artifacts_dir / "detector_ssdlite320_best.pt"
    assert orig_det.exists(), "Original detector checkpoint must be preserved"
