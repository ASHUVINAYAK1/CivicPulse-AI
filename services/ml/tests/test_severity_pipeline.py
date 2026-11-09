"""
CivicPulse AI — Severity Pipeline Unit Tests
Tests normalized geometry extraction, category normalization, and data readiness gating.
"""

import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from preprocessing.severity_feature_pipeline import (
    normalize_damage_category,
    extract_detection_features,
    extract_image_features,
)
from training.train_severity_experimental import ExperimentalSeverityPipeline


def test_normalize_damage_categories():
    assert normalize_damage_category("D40") == "D40"
    assert normalize_damage_category("Pothole") == "D40"
    assert normalize_damage_category("D20") == "D20"
    assert normalize_damage_category("alligator_crack") == "D20"
    assert normalize_damage_category("D10") == "D10"
    assert normalize_damage_category("D00") == "D00"
    assert normalize_damage_category("longitudinal_crack") == "D00"
    assert normalize_damage_category("debris") == "other"
    assert normalize_damage_category(None) == "unknown"


def test_extract_detection_features():
    # Test normalized box
    box1 = {"label": "pothole", "confidence": 0.88, "width": 0.10, "height": 0.20}
    feat1 = extract_detection_features(box1)
    assert feat1["category"] == "D40"
    assert feat1["normalized_width"] == 0.10
    assert feat1["normalized_height"] == 0.20
    assert feat1["normalized_area"] == 0.02
    assert feat1["aspect_ratio"] == 0.5
    assert feat1["confidence"] == 0.88

    # Test pixel coordinates box
    box2 = {"label": "D20", "confidence": 0.95, "xmin": 100, "xmax": 200, "ymin": 50, "ymax": 150}
    feat2 = extract_detection_features(box2, image_width=1000, image_height=1000)
    assert feat2["category"] == "D20"
    assert feat2["normalized_width"] == 0.10
    assert feat2["normalized_height"] == 0.10
    assert feat2["normalized_area"] == 0.01


def test_extract_image_features_empty():
    res = extract_image_features([])
    assert res["total_defects"] == 0
    assert res["sum_normalized_area"] == 0.0
    assert res["category_diversity"] == 0


def test_extract_image_features_multiple():
    boxes = [
        {"label": "D40", "confidence": 0.85, "width": 0.1, "height": 0.1},
        {"label": "D40", "confidence": 0.90, "width": 0.2, "height": 0.2},
        {"label": "D20", "confidence": 0.75, "width": 0.1, "height": 0.1},
    ]
    res = extract_image_features(boxes)
    assert res["total_defects"] == 3
    assert res["category_diversity"] == 2
    assert res["category_counts"]["D40"] == 2
    assert res["category_counts"]["D20"] == 1
    assert res["max_normalized_area"] == 0.04


def test_data_readiness_gate_insufficient():
    pipeline = ExperimentalSeverityPipeline(min_threshold=50)
    # Only 6 samples
    sample_records = [
        {"verifiedSeverity": "critical"},
        {"verifiedSeverity": "high"},
        {"verifiedSeverity": "medium"},
        {"verifiedSeverity": "low"},
        {"verifiedSeverity": "critical"},
        {"verifiedSeverity": "critical"},
    ]
    audit = pipeline.audit_verified_dataset(sample_records)
    assert audit["is_sufficient"] is False
    assert audit["status"] == "DATA_INSUFFICIENT"
    assert audit["total_verified_samples"] == 6

    run_res = pipeline.run_training_experiment(sample_records)
    assert run_res["success"] is False
    assert run_res["status"] == "DATA_INSUFFICIENT"

