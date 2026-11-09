"""
CivicPulse AI — Phase 2.2 Visual Prediction & Full Metric Evaluation Suite
Evaluates retrained detectors on the grouped v2 test split:
- Computes genuine mAP@0.5 and mAP@0.5:0.95 (10-point IoU sweep [0.50:0.05:0.95])
- Measures synchronized median and p95 GPU latencies (PyTorch CUDA events / synchronize)
- Generates visual prediction images with color-coded ground-truth vs prediction boxes
  highlighting True Positives (green), False Positives (red), and False Negatives (yellow)
"""

import os
import sys
import json
import time
from pathlib import Path
from collections import defaultdict
from typing import Dict, Any, List, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from training.train_detectors_v2 import (
    V2DetectionDataset,
    collate_fn,
    build_ssdlite320,
    build_fasterrcnn_mobilenetv3,
    box_iou_xyxy,
    compute_voc_ap,
    CLASS_NAMES,
    CLASS_LABELS,
)

IOU_THRESHOLDS = [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95]


def measure_synchronized_latency(model: torch.nn.Module, device: torch.device, num_iters: int = 100) -> Dict[str, float]:
    """Measures CUDA synchronized latency distribution (median, p95, min, max)."""
    model.eval()
    dummy = [torch.rand(3, 320, 320, device=device)]
    
    # Warmup
    with torch.no_grad():
        for _ in range(20):
            _ = model(dummy)
            if device.type == "cuda":
                torch.cuda.synchronize()

    latencies = []
    with torch.no_grad():
        for _ in range(num_iters):
            if device.type == "cuda":
                start_event = torch.cuda.Event(enable_timing=True)
                end_event = torch.cuda.Event(enable_timing=True)
                start_event.record()
                _ = model(dummy)
                end_event.record()
                torch.cuda.synchronize()
                latencies.append(start_event.elapsed_time(end_event))
            else:
                t0 = time.perf_counter()
                _ = model(dummy)
                latencies.append((time.perf_counter() - t0) * 1000.0)

    latencies = np.array(latencies)
    return {
        "median_ms": round(float(np.median(latencies)), 2),
        "p95_ms": round(float(np.percentile(latencies, 95)), 2),
        "min_ms": round(float(np.min(latencies)), 2),
        "max_ms": round(float(np.max(latencies)), 2),
    }


def evaluate_detailed_map(
    model: torch.nn.Module,
    test_loader: DataLoader,
    device: torch.device,
    conf_thresh: float = 0.15,
) -> Dict[str, Any]:
    """Evaluates mAP@0.5 and mAP@0.5:0.95 across all damage classes."""
    model.eval()

    gt_by_class = {c: defaultdict(list) for c in range(1, 5)}
    dt_by_class = {c: [] for c in range(1, 5)}
    total_gt = {c: 0 for c in range(1, 5)}

    sample_predictions = []
    img_counter = 0

    with torch.no_grad():
        for images, targets in test_loader:
            images = [img.to(device) for img in images]
            preds = model(images)

            for target, pred in zip(targets, preds):
                img_id = f"test_{img_counter:04d}"
                img_counter += 1

                gt_boxes = target["boxes"].cpu().numpy()
                gt_labels = target["labels"].cpu().numpy()
                for b, l in zip(gt_boxes, gt_labels):
                    if 1 <= l <= 4:
                        gt_by_class[l][img_id].append(b)
                        total_gt[l] += 1

                p_boxes = pred["boxes"].cpu().numpy()
                p_scores = pred["scores"].cpu().numpy()
                p_labels = pred["labels"].cpu().numpy()

                img_dt = []
                for b, s, l in zip(p_boxes, p_scores, p_labels):
                    if s >= conf_thresh and 1 <= l <= 4:
                        det_obj = {
                            "img_id": img_id,
                            "confidence": float(s),
                            "box": b,
                            "label": int(l),
                            "class_name": CLASS_NAMES[l - 1],
                        }
                        dt_by_class[l].append(det_obj)
                        img_dt.append(det_obj)

                sample_predictions.append({
                    "img_id": img_id,
                    "gt_boxes": gt_boxes,
                    "gt_labels": gt_labels,
                    "predictions": img_dt,
                })

    # Evaluate AP for each IoU threshold
    ap_table = {f"iou_{th:.2f}": {} for th in IOU_THRESHOLDS}
    matched_ious = []
    tp_05_total, fp_05_total, fn_05_total = 0, 0, 0

    for iou_th in IOU_THRESHOLDS:
        iou_key = f"iou_{iou_th:.2f}"
        for c in range(1, 5):
            cname = CLASS_NAMES[c - 1]
            detections = list(dt_by_class[c])
            detections.sort(key=lambda d: d["confidence"], reverse=True)
            n_gt = total_gt[c]

            if n_gt == 0 or len(detections) == 0:
                ap_table[iou_key][cname] = 0.0
                if iou_th == 0.50:
                    fn_05_total += n_gt
                continue

            tp = np.zeros(len(detections))
            fp = np.zeros(len(detections))
            gt_matched = {img_id: np.zeros(len(boxes), dtype=bool) for img_id, boxes in gt_by_class[c].items()}

            for d_idx, det in enumerate(detections):
                img_id = det["img_id"]
                box = det["box"]
                gt_boxes = gt_by_class[c].get(img_id, [])

                best_iou = 0.0
                best_gt_idx = -1
                for g_idx, g_box in enumerate(gt_boxes):
                    iou = box_iou_xyxy(box, g_box)
                    if iou > best_iou:
                        best_iou = iou
                        best_gt_idx = g_idx

                if best_iou >= iou_th:
                    if not gt_matched[img_id][best_gt_idx]:
                        tp[d_idx] = 1.0
                        gt_matched[img_id][best_gt_idx] = True
                        if iou_th == 0.50:
                            matched_ious.append(best_iou)
                    else:
                        fp[d_idx] = 1.0
                else:
                    fp[d_idx] = 1.0

            cum_tp = np.cumsum(tp)
            cum_fp = np.cumsum(fp)
            rec = cum_tp / float(n_gt)
            prec = cum_tp / np.maximum(cum_tp + cum_fp, 1e-8)
            ap = compute_voc_ap(rec, prec)
            ap_table[iou_key][cname] = round(float(ap), 4)

            if iou_th == 0.50:
                tp_05_total += int(np.sum(tp))
                fp_05_total += int(np.sum(fp))
                fn_05_total += (n_gt - int(np.sum(tp)))

    # Compute mAP@0.5 and mAP@0.5:0.95
    per_class_map50 = {cname: ap_table["iou_0.50"][cname] for cname in CLASS_NAMES}
    map_50 = float(np.mean(list(per_class_map50.values())))

    # Mean across all 10 IoU thresholds
    all_maps = []
    for th in IOU_THRESHOLDS:
        iou_key = f"iou_{th:.2f}"
        all_maps.append(np.mean(list(ap_table[iou_key].values())))
    map_50_95 = float(np.mean(all_maps))

    overall_prec = float(tp_05_total / max(tp_05_total + fp_05_total, 1))
    overall_rec = float(tp_05_total / max(tp_05_total + fn_05_total, 1))
    mean_iou = float(np.mean(matched_ious)) if matched_ious else 0.0

    return {
        "mAP_50": round(map_50, 4),
        "mAP_50_95": round(map_50_95, 4),
        "overall_precision_50": round(overall_prec, 4),
        "overall_recall_50": round(overall_rec, 4),
        "mean_matched_iou": round(mean_iou, 4),
        "per_class_ap_50": per_class_map50,
        "ap_table_all_ious": ap_table,
        "total_proposals": sum(len(dt_by_class[c]) for c in range(1, 5)),
        "true_positives_50": tp_05_total,
        "false_positives_50": fp_05_total,
        "false_negatives_50": fn_05_total,
        "sample_predictions": sample_predictions,
    }


def render_visual_prediction_examples(
    test_ds: V2DetectionDataset,
    sample_preds: List[Dict[str, Any]],
    output_dir: Path,
    model_name: str,
    num_examples: int = 6,
):
    """Renders visual overlays with color-coded ground-truth and prediction boxes."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Find interesting examples with damage
    interesting_indices = []
    for idx, (img_p, lbl_p) in enumerate(test_ds.samples):
        content = lbl_p.read_text().strip() if lbl_p.exists() else ""
        if content and len(sample_preds[idx]["predictions"]) > 0:
            interesting_indices.append(idx)
        if len(interesting_indices) >= num_examples:
            break

    for i, idx in enumerate(interesting_indices):
        img_p, lbl_p = test_ds.samples[idx]
        with Image.open(img_p) as img:
            vis_img = img.convert("RGB").resize((640, 640))

        draw = ImageDraw.Draw(vis_img)
        scale_x = 640.0 / 320.0
        scale_y = 640.0 / 320.0

        item = sample_preds[idx]
        gt_boxes = item["gt_boxes"]
        gt_labels = item["gt_labels"]
        preds = item["predictions"]

        # Draw Ground Truths in Cyan
        for b, l in zip(gt_boxes, gt_labels):
            x1, y1, x2, y2 = b[0] * scale_x, b[1] * scale_y, b[2] * scale_x, b[3] * scale_y
            cname = CLASS_NAMES[l - 1] if 1 <= l <= 4 else f"C{l}"
            draw.rectangle([x1, y1, x2, y2], outline="#00e5ff", width=3)
            draw.text((x1 + 4, max(0, y1 - 18)), f"GT: {cname}", fill="#00e5ff")

        # Match predictions to GT to label TP vs FP
        gt_used = np.zeros(len(gt_boxes), dtype=bool)
        for det in preds:
            b = det["box"]
            x1, y1, x2, y2 = b[0] * scale_x, b[1] * scale_y, b[2] * scale_x, b[3] * scale_y
            cname = det["class_name"]
            score = det["confidence"]

            # Check IoU against GT
            best_iou = 0.0
            best_g_idx = -1
            for g_idx, gb in enumerate(gt_boxes):
                iou = box_iou_xyxy(b, gb)
                if iou > best_iou:
                    best_iou = iou
                    best_g_idx = g_idx

            is_tp = (best_iou >= 0.50) and (not gt_used[best_g_idx])
            if is_tp:
                gt_used[best_g_idx] = True
                color = "#00e676"  # Vibrant Green for True Positive
                tag = f"TP: {cname} {score*100:.0f}%"
            else:
                color = "#ff1744"  # Vibrant Red for False Positive
                tag = f"FP: {cname} {score*100:.0f}%"

            draw.rectangle([x1, y1, x2, y2], outline=color, width=2)
            draw.text((x1 + 4, min(620, y2 + 2)), tag, fill=color)

        out_path = output_dir / f"{model_name}_sample_{i+1:02d}_{img_p.stem}.jpg"
        vis_img.save(out_path, quality=92)

    print(f"Rendered {len(interesting_indices)} visual prediction samples to {output_dir}")

