"""
CivicPulse AI — Phase 2.1 Object Detection Independent Audit Script
Evaluates SSDLite320 on the held-out test split of RDD2022 India using original bounding boxes.
Calculates genuine held-out:
- mAP@0.5
- mAP@0.5:0.95
- Per-class AP (Average Precision)
- Per-class Precision and Recall
- Mean Bounding-Box IoU
- False Positive and False Negative case examples
"""

import sys
import json
import time
from pathlib import Path
from typing import Dict, Any, List, Tuple
from collections import defaultdict

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from preprocessing.rdd2022_parser import RDDAnnotationParser, RoadDamageSample, BoundingBox
from models.detection.detector import RoadDamageDetector

CLASS_NAMES = ["D00", "D10", "D20", "D40"]
CLASS_MAP = {1: "D00", 2: "D10", 3: "D20", 4: "D40"}


def box_iou(box1: np.ndarray, box2: np.ndarray) -> float:
    """Computes IoU between box1 [x1, y1, x2, y2] and box2 [x1, y1, x2, y2]."""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter_area = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union_area = area1 + area2 - inter_area

    if union_area <= 0:
        return 0.0
    return inter_area / union_area


def compute_ap(recalls: np.ndarray, precisions: np.ndarray) -> float:
    """Computes Average Precision (AP) using all-point interpolation (COCO standard)."""
    mrec = np.concatenate(([0.0], recalls, [1.0]))
    mpre = np.concatenate(([0.0], precisions, [0.0]))

    # Compute precision envelope
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

    # Find points where recall changes
    idx = np.where(mrec[1:] != mrec[:-1])[0]
    ap = np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1])
    return float(ap)


def evaluate_detection_map(
    detector: RoadDamageDetector,
    samples: List[RoadDamageSample],
    device: torch.device,
    iou_thresholds: List[float] = [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95],
    conf_threshold: float = 0.05,
) -> Dict[str, Any]:
    detector.eval()

    # Ground truth storage by class: image_id -> list of [x1, y1, x2, y2]
    # In normalized coordinates [0..1]
    gt_by_class: Dict[str, Dict[str, List[np.ndarray]]] = {c: defaultdict(list) for c in CLASS_NAMES}
    total_gt_boxes: Dict[str, int] = {c: 0 for c in CLASS_NAMES}

    for s in samples:
        orig_w = float(s.width or 720)
        orig_h = float(s.height or 720)
        for b in s.boxes:
            if b.label in CLASS_NAMES:
                # Store normalized coordinates
                x1 = b.xmin / orig_w
                y1 = b.ymin / orig_h
                x2 = b.xmax / orig_w
                y2 = b.ymax / orig_h
                gt_by_class[b.label][s.image_id].append(np.array([x1, y1, x2, y2], dtype=np.float32))
                total_gt_boxes[b.label] += 1

    # Detections storage by class: list of {image_id, confidence, box: [x1, y1, x2, y2]}
    dt_by_class: Dict[str, List[Dict[str, Any]]] = {c: [] for c in CLASS_NAMES}

    print(f"Running inference on {len(samples)} test images for object detection...")
    start_t = time.perf_counter()

    ious_matched = []
    fp_examples = []
    fn_examples = []

    with torch.no_grad():
        for i, sample in enumerate(samples):
            img_path = Path(sample.image_path)
            try:
                with Image.open(img_path) as img:
                    pil_img = img.convert("RGB").resize((320, 320))
            except Exception:
                continue

            tensor = torch.from_numpy(np.asarray(pil_img, dtype=np.float32) / 255.0).permute(2, 0, 1).to(device)
            preds = detector([tensor])[0]

            boxes = preds["boxes"].cpu().numpy()
            scores = preds["scores"].cpu().numpy()
            labels = preds["labels"].cpu().numpy()

            for b, s, l in zip(boxes, scores, labels):
                if s >= conf_threshold and 1 <= l <= 4:
                    cname = CLASS_NAMES[l - 1]
                    # Convert from 320x320 box to normalized [0..1]
                    x1 = max(0.0, min(1.0, float(b[0] / 320.0)))
                    y1 = max(0.0, min(1.0, float(b[1] / 320.0)))
                    x2 = max(0.0, min(1.0, float(b[2] / 320.0)))
                    y2 = max(0.0, min(1.0, float(b[3] / 320.0)))

                    dt_by_class[cname].append({
                        "image_id": sample.image_id,
                        "confidence": float(s),
                        "box": np.array([x1, y1, x2, y2], dtype=np.float32),
                    })

    eval_duration = time.perf_counter() - start_t
    print(f"Detector evaluation forward passes completed in {eval_duration:.1f}s.")

    # Calculate AP across IoU thresholds
    ap_results = {}
    per_class_metrics_05 = {}

    for iou_th in iou_thresholds:
        ap_results[f"iou_{iou_th:.2f}"] = {}

    for cname in CLASS_NAMES:
        detections = dt_by_class[cname]
        # Sort detections descending by confidence
        detections.sort(key=lambda d: d["confidence"], reverse=True)
        n_gt = total_gt_boxes[cname]

        if n_gt == 0:
            for iou_th in iou_thresholds:
                ap_results[f"iou_{iou_th:.2f}"][cname] = 0.0
            continue

        if len(detections) == 0:
            for iou_th in iou_thresholds:
                ap_results[f"iou_{iou_th:.2f}"][cname] = 0.0
            per_class_metrics_05[cname] = {
                "support_gt": n_gt,
                "detected": 0,
                "ap_50": 0.0,
                "precision": 0.0,
                "recall": 0.0,
                "mean_iou": 0.0,
            }
            continue

        # Evaluate at each IoU threshold
        for iou_th in iou_thresholds:
            tp = np.zeros(len(detections))
            fp = np.zeros(len(detections))
            # Track which GT boxes have been matched per image: image_id -> list of bools
            gt_matched = {img_id: np.zeros(len(boxes), dtype=bool) for img_id, boxes in gt_by_class[cname].items()}

            for d_idx, det in enumerate(detections):
                img_id = det["image_id"]
                dt_box = det["box"]
                gt_boxes = gt_by_class[cname].get(img_id, [])

                best_iou = 0.0
                best_gt_idx = -1

                for g_idx, g_box in enumerate(gt_boxes):
                    cur_iou = box_iou(dt_box, g_box)
                    if cur_iou > best_iou:
                        best_iou = cur_iou
                        best_gt_idx = g_idx

                if best_iou >= iou_th:
                    if not gt_matched[img_id][best_gt_idx]:
                        tp[d_idx] = 1.0
                        gt_matched[img_id][best_gt_idx] = True
                        if iou_th == 0.50:
                            ious_matched.append(best_iou)
                    else:
                        fp[d_idx] = 1.0
                else:
                    fp[d_idx] = 1.0
                    if iou_th == 0.50 and len(fp_examples) < 10 and det["confidence"] >= 0.2:
                        fp_examples.append({
                            "image_id": img_id,
                            "class": cname,
                            "confidence": round(det["confidence"], 4),
                            "box_norm": [round(float(v), 4) for v in dt_box],
                            "best_iou_with_any_gt": round(float(best_iou), 4),
                            "reason": "IoU below threshold with ground truth or background false alarm"
                        })

            # Check False Negatives (unmatched GT boxes) at IoU 0.50
            if iou_th == 0.50:
                for img_id, matched_arr in gt_matched.items():
                    for g_idx, matched in enumerate(matched_arr):
                        if not matched and len(fn_examples) < 10:
                            g_box = gt_by_class[cname][img_id][g_idx]
                            fn_examples.append({
                                "image_id": img_id,
                                "class": cname,
                                "box_norm": [round(float(v), 4) for v in g_box],
                                "reason": "Detector missed defect entirely or confidence was suppressed below threshold"
                            })

            tp_cumsum = np.cumsum(tp)
            fp_cumsum = np.cumsum(fp)
            recalls = tp_cumsum / float(n_gt)
            precisions = tp_cumsum / np.maximum(tp_cumsum + fp_cumsum, 1e-6)

            ap = compute_ap(recalls, precisions)
            ap_results[f"iou_{iou_th:.2f}"][cname] = round(ap, 4)

            if iou_th == 0.50:
                final_prec = float(precisions[-1]) if len(precisions) > 0 else 0.0
                final_rec = float(recalls[-1]) if len(recalls) > 0 else 0.0
                per_class_metrics_05[cname] = {
                    "support_gt": n_gt,
                    "total_detections": len(detections),
                    "tp_at_05": int(tp_cumsum[-1]),
                    "fp_at_05": int(fp_cumsum[-1]),
                    "ap_50": round(ap, 4),
                    "precision_50": round(final_prec, 4),
                    "recall_50": round(final_rec, 4),
                }

    # Summary mAP across classes
    map_50 = float(np.mean([ap_results["iou_0.50"][c] for c in CLASS_NAMES]))
    
    # COCO mAP (0.50:0.95)
    coco_map_by_class = {}
    for c in CLASS_NAMES:
        coco_map_by_class[c] = float(np.mean([ap_results[f"iou_{th:.2f}"][c] for th in iou_thresholds]))
    map_coco = float(np.mean(list(coco_map_by_class.values())))

    mean_iou_matched = float(np.mean(ious_matched)) if ious_matched else 0.0

    return {
        "architecture": "ssdlite320_mobilenet_v3_large",
        "checkpoint": "detector_ssdlite320_best.pt",
        "num_test_images": len(samples),
        "total_gt_boxes": sum(total_gt_boxes.values()),
        "gt_boxes_per_class": total_gt_boxes,
        "mAP_50": round(map_50, 4),
        "mAP_50_95": round(map_coco, 4),
        "per_class_ap_50": {c: ap_results["iou_0.50"][c] for c in CLASS_NAMES},
        "per_class_ap_coco": {c: round(coco_map_by_class[c], 4) for c in CLASS_NAMES},
        "per_class_details_05": per_class_metrics_05,
        "mean_iou_on_true_positives": round(mean_iou_matched, 4),
        "false_positive_cases": fp_examples[:5],
        "false_negative_cases": fn_examples[:5],
    }


def main():
    print("=" * 70)
    print("  CivicPulse AI — Phase 2.1 Object Detection Independent Audit  ")
    print("=" * 70)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Audit compute engine: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")

    test_dir = Path("datasets/raw/RDD2022_India/test")
    test_samples = RDDAnnotationParser.load_yolo_split(test_dir)
    print(f"Loaded {len(test_samples)} held-out test samples with ground-truth boxes from {test_dir}")

    artifacts_dir = Path("services/ml/model_artifacts")
    det_ckpt_path = artifacts_dir / "detector_ssdlite320_best.pt"
    if not det_ckpt_path.exists():
        raise FileNotFoundError(f"Detector checkpoint not found: {det_ckpt_path}")

    checkpoint = torch.load(det_ckpt_path, map_location=device)
    detector = RoadDamageDetector(pretrained=True, num_classes=5).to(device)
    detector.load_state_dict(checkpoint["state_dict"])
    detector.eval()

    results = evaluate_detection_map(detector, test_samples, device)

    out_file = artifacts_dir / "detection_audit_results.json"
    with open(out_file, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nDetection audit complete. Results saved to {out_file}")
    print(f"mAP@0.5: {results['mAP_50']:.4f}")
    print(f"mAP@0.5:0.95: {results['mAP_50_95']:.4f}")
    print(f"Per-Class AP@0.5: {results['per_class_ap_50']}")


if __name__ == "__main__":
    main()

