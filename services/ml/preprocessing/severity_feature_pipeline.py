"""
CivicPulse AI — Python Severity Feature Extraction Pipeline
Extracts detection-level, image-level, and incident-level geometric features.
"""

from typing import List, Dict, Any, Optional
import numpy as np


def normalize_damage_category(cat: Optional[str]) -> str:
    """Normalizes raw category strings to canonical codes."""
    if not cat:
        return "unknown"
    c = cat.lower().strip()
    if "40" in c or "pothole" in c:
        return "D40"
    if "20" in c or "alligator" in c:
        return "D20"
    if "10" in c or "transverse" in c:
        return "D10"
    if "00" in c or "longitudinal" in c or "crack" in c:
        return "D00"
    return "other"


def extract_detection_features(
    box: Dict[str, Any], image_width: int = 720, image_height: int = 720
) -> Dict[str, Any]:
    """Extracts normalized geometric metrics from a single bounding box."""
    label = box.get("label") or box.get("class") or "unknown"
    conf = float(box.get("confidence") or box.get("score") or 0.0)

    # Normalized width and height
    w = float(box.get("width") or 0.0)
    h = float(box.get("height") or 0.0)

    if w > 1.0 or h > 1.0:
        w = w / max(image_width, 1)
        h = h / max(image_height, 1)
    elif "xmin" in box and "xmax" in box and "ymin" in box and "ymax" in box:
        raw_w = max(0.0, float(box["xmax"]) - float(box["xmin"]))
        raw_h = max(0.0, float(box["ymax"]) - float(box["ymin"]))
        w = raw_w / max(image_width, 1) if raw_w > 1.0 else raw_w
        h = raw_h / max(image_height, 1) if raw_h > 1.0 else raw_h

    w = float(np.clip(w, 0.0, 1.0))
    h = float(np.clip(h, 0.0, 1.0))
    area = float(np.clip(w * h, 0.0, 1.0))
    aspect = float(w / h) if h > 1e-6 else 1.0

    return {
        "category": normalize_damage_category(label),
        "normalized_width": round(w, 4),
        "normalized_height": round(h, 4),
        "normalized_area": round(area, 5),
        "aspect_ratio": round(aspect, 3),
        "confidence": round(float(np.clip(conf, 0.0, 1.0)), 4),
    }


def extract_image_features(
    boxes: List[Dict[str, Any]], image_width: int = 720, image_height: int = 720
) -> Dict[str, Any]:
    """Aggregates bounding boxes into image-level spatial distress features."""
    if not boxes:
        return {
            "total_defects": 0,
            "category_counts": {},
            "sum_normalized_area": 0.0,
            "max_normalized_area": 0.0,
            "mean_normalized_area": 0.0,
            "category_diversity": 0,
            "detections": [],
        }

    extracted = [extract_detection_features(b, image_width, image_height) for b in boxes]
    counts: Dict[str, int] = {}
    sum_area = 0.0
    max_area = 0.0

    for det in extracted:
        cat = det["category"]
        counts[cat] = counts.get(cat, 0) + 1
        sum_area += det["normalized_area"]
        if det["normalized_area"] > max_area:
            max_area = det["normalized_area"]

    n = len(extracted)
    mean_area = sum_area / n if n > 0 else 0.0

    return {
        "total_defects": n,
        "category_counts": counts,
        "sum_normalized_area": round(sum_area, 4),
        "max_normalized_area": round(max_area, 4),
        "mean_normalized_area": round(mean_area, 4),
        "category_diversity": len(counts),
        "detections": extracted,
    }

