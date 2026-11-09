"""
CivicPulse AI — Phase 2.2 Detector Retraining Suite
Retrains SSDLite320 (MobileNetV3-Large) and Faster R-CNN (MobileNetV3-Large-320 FPN)
on the grouped v2 dataset using full COCO transfer initialization, data augmentation,
cosine annealing LR schedules, and validation-based early stopping.
"""

import os
import sys
import time
import json
import random
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from PIL import Image, ImageEnhance
import torchvision
from torchvision.models.detection import (
    ssdlite320_mobilenet_v3_large,
    SSDLite320_MobileNet_V3_Large_Weights,
    fasterrcnn_mobilenet_v3_large_320_fpn,
    FasterRCNN_MobileNet_V3_Large_320_FPN_Weights,
)
from torchvision.models.detection.ssdlite import SSDLiteClassificationHead
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

CLASS_NAMES = ["D00", "D10", "D20", "D40"]
CLASS_LABELS = {
    0: "D00",
    1: "D10",
    2: "D20",
    3: "D40"
}


def box_iou_xyxy(box1: np.ndarray, box2: np.ndarray) -> float:
    """Computes IoU between two [x1, y1, x2, y2] boxes."""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    a1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    a2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = a1 + a2 - inter
    return inter / max(union, 1e-8)


def compute_voc_ap(recalls: np.ndarray, precisions: np.ndarray) -> float:
    """Computes all-point interpolated AP (COCO/VOC standard)."""
    mrec = np.concatenate(([0.0], recalls, [1.0]))
    mpre = np.concatenate(([0.0], precisions, [0.0]))
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])
    idx = np.where(mrec[1:] != mrec[:-1])[0]
    return float(np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1]))


class V2DetectionDataset(Dataset):
    """Dataset for torchvision detectors with bounding box coordinates and augmentations."""

    def __init__(
        self,
        split_dir: Path,
        target_size: Tuple[int, int] = (320, 320),
        augment: bool = False,
        background_ratio: float = 0.35,
    ):
        self.target_size = target_size
        self.augment = augment
        img_dir = split_dir / "images"
        lbl_dir = split_dir / "labels"

        all_imgs = sorted(list(img_dir.glob("*.jpg")))
        positives = []
        negatives = []

        for img_p in all_imgs:
            lbl_p = lbl_dir / f"{img_p.stem}.txt"
            content = lbl_p.read_text().strip() if lbl_p.exists() else ""
            if content:
                positives.append((img_p, lbl_p))
            else:
                negatives.append((img_p, lbl_p))

        # Sample negatives to teach background rejection without dominating positive gradients
        if augment and background_ratio > 0:
            n_neg = int(len(positives) * background_ratio)
            random.seed(42)
            sampled_neg = random.sample(negatives, min(n_neg, len(negatives)))
            self.samples = positives + sampled_neg
            random.shuffle(self.samples)
        else:
            self.samples = positives + negatives

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        img_p, lbl_p = self.samples[idx]

        with Image.open(img_p) as img:
            pil_img = img.convert("RGB")

        orig_w, orig_h = pil_img.size

        # Parse boxes
        raw_boxes = []
        raw_labels = []
        if lbl_p.exists():
            content = lbl_p.read_text().strip()
            if content:
                for line in content.splitlines():
                    tokens = line.strip().split()
                    if len(tokens) >= 5:
                        cls_id = int(tokens[0]) + 1  # 1..4 in torchvision (0 is background)
                        xc = float(tokens[1]) * orig_w
                        yc = float(tokens[2]) * orig_h
                        bw = float(tokens[3]) * orig_w
                        bh = float(tokens[4]) * orig_h
                        x1 = max(0.0, xc - bw / 2.0)
                        y1 = max(0.0, yc - bh / 2.0)
                        x2 = min(float(orig_w), xc + bw / 2.0)
                        y2 = min(float(orig_h), yc + bh / 2.0)
                        if x2 > x1 + 2 and y2 > y1 + 2:
                            raw_boxes.append([x1, y1, x2, y2])
                            raw_labels.append(cls_id)

        # Apply augmentation: horizontal flip
        do_flip = self.augment and (random.random() > 0.5)
        if do_flip:
            pil_img = pil_img.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
            flipped_boxes = []
            for b in raw_boxes:
                flipped_boxes.append([orig_w - b[2], b[1], orig_w - b[0], b[3]])
            raw_boxes = flipped_boxes

        # Apply photometric augmentation
        if self.augment:
            if random.random() > 0.5:
                enhancer = ImageEnhance.Brightness(pil_img)
                pil_img = enhancer.enhance(random.uniform(0.8, 1.2))
            if random.random() > 0.5:
                enhancer = ImageEnhance.Contrast(pil_img)
                pil_img = enhancer.enhance(random.uniform(0.8, 1.2))

        # Resize to target size
        tw, th = self.target_size
        pil_img = pil_img.resize((tw, th), Image.Resampling.BILINEAR)
        scale_x = tw / float(orig_w)
        scale_y = th / float(orig_h)

        scaled_boxes = []
        scaled_labels = []
        for b, l in zip(raw_boxes, raw_labels):
            x1 = max(0.0, min(float(tw), b[0] * scale_x))
            y1 = max(0.0, min(float(th), b[1] * scale_y))
            x2 = max(0.0, min(float(tw), b[2] * scale_x))
            y2 = max(0.0, min(float(th), b[3] * scale_y))
            if x2 > x1 + 1 and y2 > y1 + 1:
                scaled_boxes.append([x1, y1, x2, y2])
                scaled_labels.append(l)

        # Convert to tensors (Image in [0, 1] float tensor)
        img_np = np.asarray(pil_img, dtype=np.float32) / 255.0
        img_tensor = torch.from_numpy(img_np).permute(2, 0, 1)

        if len(scaled_boxes) > 0:
            target = {
                "boxes": torch.as_tensor(scaled_boxes, dtype=torch.float32),
                "labels": torch.as_tensor(scaled_labels, dtype=torch.int64),
            }
        else:
            target = {
                "boxes": torch.zeros((0, 4), dtype=torch.float32),
                "labels": torch.zeros((0,), dtype=torch.int64),
            }

        return img_tensor, target


def collate_fn(batch):
    return tuple(zip(*batch))


def build_ssdlite320(num_classes: int = 5) -> nn.Module:
    """Builds SSDLite320 with MobileNetV3-Large backbone and COCO transfer weights."""
    model = ssdlite320_mobilenet_v3_large(weights=SSDLite320_MobileNet_V3_Large_Weights.DEFAULT)
    in_channels = [layer[0][0].in_channels for layer in model.head.classification_head.module_list]
    num_anchors = model.anchor_generator.num_anchors_per_location()
    new_head = SSDLiteClassificationHead(
        in_channels=in_channels,
        num_anchors=num_anchors,
        num_classes=num_classes,
        norm_layer=nn.BatchNorm2d,
    )
    model.head.classification_head = new_head
    return model


def build_fasterrcnn_mobilenetv3(num_classes: int = 5) -> nn.Module:
    """Builds Faster R-CNN MobileNetV3-Large-320 FPN with COCO transfer weights."""
    model = fasterrcnn_mobilenet_v3_large_320_fpn(weights=FasterRCNN_MobileNet_V3_Large_320_FPN_Weights.DEFAULT)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes=num_classes)
    return model


def evaluate_detector_map(
    model: nn.Module,
    val_loader: DataLoader,
    device: torch.device,
    conf_thresh: float = 0.05,
    iou_thresh: float = 0.50,
) -> Dict[str, Any]:
    """Calculates VOC/COCO mAP@0.5 and per-class AP on validation/test set."""
    model.eval()

    gt_by_class = {c: defaultdict(list) for c in range(1, 5)}
    dt_by_class = {c: [] for c in range(1, 5)}
    total_gt = {c: 0 for c in range(1, 5)}

    img_counter = 0
    with torch.no_grad():
        for images, targets in val_loader:
            images = [img.to(device) for img in images]
            preds = model(images)

            for target, pred in zip(targets, preds):
                img_id = f"img_{img_counter}"
                img_counter += 1

                # Ground truths
                gt_boxes = target["boxes"].cpu().numpy()
                gt_labels = target["labels"].cpu().numpy()
                for b, l in zip(gt_boxes, gt_labels):
                    if 1 <= l <= 4:
                        gt_by_class[l][img_id].append(b)
                        total_gt[l] += 1

                # Predictions
                p_boxes = pred["boxes"].cpu().numpy()
                p_scores = pred["scores"].cpu().numpy()
                p_labels = pred["labels"].cpu().numpy()

                for b, s, l in zip(p_boxes, p_scores, p_labels):
                    if s >= conf_thresh and 1 <= l <= 4:
                        dt_by_class[l].append({
                            "img_id": img_id,
                            "confidence": float(s),
                            "box": b,
                        })

    per_class_ap = {}
    matched_ious = []
    tp_total, fp_total, fn_total = 0, 0, 0

    for c in range(1, 5):
        cname = CLASS_NAMES[c - 1]
        detections = dt_by_class[c]
        detections.sort(key=lambda d: d["confidence"], reverse=True)
        n_gt = total_gt[c]

        if n_gt == 0:
            per_class_ap[cname] = 0.0
            continue

        if len(detections) == 0:
            per_class_ap[cname] = 0.0
            fn_total += n_gt
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

            if best_iou >= iou_thresh:
                if not gt_matched[img_id][best_gt_idx]:
                    tp[d_idx] = 1.0
                    gt_matched[img_id][best_gt_idx] = True
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
        per_class_ap[cname] = round(float(ap), 4)

        tp_total += int(np.sum(tp))
        fp_total += int(np.sum(fp))
        fn_total += (n_gt - int(np.sum(tp)))

    map_50 = float(np.mean(list(per_class_ap.values())))
    overall_prec = float(tp_total / max(tp_total + fp_total, 1))
    overall_rec = float(tp_total / max(tp_total + fn_total, 1))
    mean_iou = float(np.mean(matched_ious)) if matched_ious else 0.0

    return {
        "mAP_50": round(map_50, 4),
        "overall_precision": round(overall_prec, 4),
        "overall_recall": round(overall_rec, 4),
        "mean_matched_iou": round(mean_iou, 4),
        "total_proposals": sum(len(dt_by_class[c]) for c in range(1, 5)),
        "per_class_ap_50": per_class_ap,
    }


def train_single_detector(
    detector_name: str,
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    test_loader: DataLoader,
    device: torch.device,
    epochs: int = 35,
    initial_lr: float = 2e-4,
    checkpoint_out: Path = Path("services/ml/model_artifacts/detector_v2.pt"),
) -> Dict[str, Any]:
    print(f"\n{'='*70}")
    print(f"  Training Detector: {detector_name.upper()} on {device}")
    print(f"  Epochs: {epochs} | Batch Size: {train_loader.batch_size} | Initial LR: {initial_lr}")
    print(f"{'='*70}")

    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=initial_lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

    best_val_map = -1.0
    best_weights = None
    epochs_no_improve = 0
    patience = 12

    history = []
    start_time = time.time()

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0.0
        n_batches = 0

        for images, targets in train_loader:
            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())

            optimizer.zero_grad()
            losses.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
            optimizer.step()

            epoch_loss += losses.item()
            n_batches += 1

        scheduler.step()
        avg_loss = epoch_loss / max(n_batches, 1)

        # Validation evaluation
        val_metrics = evaluate_detector_map(model, val_loader, device=device)
        val_map = val_metrics["mAP_50"]

        history.append({
            "epoch": epoch,
            "train_loss": round(avg_loss, 4),
            "val_map_50": val_map,
            "val_prec": val_metrics["overall_precision"],
            "val_rec": val_metrics["overall_recall"],
        })

        is_best = val_map > best_val_map
        if is_best:
            best_val_map = val_map
            best_weights = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        print(
            f"Epoch [{epoch:02d}/{epochs:02d}] | Train Loss: {avg_loss:.4f} | "
            f"Val mAP@0.5: {val_map*100:.2f}% | Val Rec: {val_metrics['overall_recall']*100:.1f}% | "
            f"Val Prec: {val_metrics['overall_precision']*100:.1f}% "
            f"{'★ BEST' if is_best else ''}"
        )

        if epochs_no_improve >= patience and epoch >= 20:
            print(f"Early stopping triggered at epoch {epoch} (no improvement for {patience} epochs).")
            break

    total_training_time = time.time() - start_time
    print(f"\nTraining finished in {total_training_time:.1f}s. Best Val mAP@0.5: {best_val_map*100:.2f}%")

    # Load best weights
    model.load_state_dict(best_weights)
    model.to(device)

    # Save checkpoint
    checkpoint_out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "architecture": detector_name,
            "state_dict": best_weights,
            "classes": ["background"] + CLASS_NAMES,
            "val_map_50": best_val_map,
            "training_time_sec": round(total_training_time, 1),
            "epochs_trained": len(history),
        },
        checkpoint_out,
    )
    print(f"Saved best model checkpoint to {checkpoint_out}")

    # Final independent evaluation on TEST set
    print(f"\n--- Final Held-Out Evaluation on Test Split ({len(test_loader.dataset)} images) ---")
    test_metrics = evaluate_detector_map(model, test_loader, device=device)

    # Measure latency
    model.eval()
    latencies = []
    dummy = [torch.rand(3, 320, 320, device=device)]
    with torch.no_grad():
        for _ in range(50):
            t0 = time.perf_counter()
            _ = model(dummy)
            torch.cuda.synchronize() if device.type == "cuda" else None
            latencies.append((time.perf_counter() - t0) * 1000.0)

    med_latency = float(np.median(latencies[5:]))

    print(f"Test mAP@0.5:           {test_metrics['mAP_50']*100:.2f}%")
    print(f"Test Mean IoU:          {test_metrics['mean_matched_iou']*100:.2f}%")
    print(f"Test Precision:         {test_metrics['overall_precision']*100:.2f}%")
    print(f"Test Recall:            {test_metrics['overall_recall']*100:.2f}%")
    print(f"GPU Latency:            {med_latency:.2f} ms")
    print("Per-class AP@0.5 breakdown:")
    for cname, ap in test_metrics["per_class_ap_50"].items():
        print(f"  {cname}: {ap*100:.2f}%")

    return {
        "detector_name": detector_name,
        "checkpoint_path": str(checkpoint_out),
        "training_time_sec": round(total_training_time, 1),
        "epochs_trained": len(history),
        "best_val_map_50": round(best_val_map, 4),
        "test_map_50": test_metrics["mAP_50"],
        "test_precision": test_metrics["overall_precision"],
        "test_recall": test_metrics["overall_recall"],
        "test_mean_matched_iou": test_metrics["mean_matched_iou"],
        "test_per_class_ap_50": test_metrics["per_class_ap_50"],
        "gpu_latency_ms": round(med_latency, 2),
        "history": history,
    }


def main():
    print("=" * 75)
    print("  CivicPulse AI — Phase 2.2 Object Detector Retraining Pipeline  ")
    print("=" * 75)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")

    base_dir = Path("datasets/splits/rdd2022_india_grouped_v2")
    train_dir = base_dir / "train"
    val_dir = base_dir / "val"
    test_dir = base_dir / "test"

    print("Building datasets with augmentations and calibrated background samples...")
    train_ds = V2DetectionDataset(train_dir, target_size=(320, 320), augment=True, background_ratio=0.35)
    val_ds = V2DetectionDataset(val_dir, target_size=(320, 320), augment=False, background_ratio=0.0)
    test_ds = V2DetectionDataset(test_dir, target_size=(320, 320), augment=False, background_ratio=0.0)

    print(f"Dataset sizes: Train={len(train_ds)}, Val={len(val_ds)}, Test={len(test_ds)}")

    train_loader = DataLoader(
        train_ds, batch_size=16, shuffle=True, collate_fn=collate_fn, num_workers=0, pin_memory=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=16, shuffle=False, collate_fn=collate_fn, num_workers=0, pin_memory=True
    )
    test_loader = DataLoader(
        test_ds, batch_size=16, shuffle=False, collate_fn=collate_fn, num_workers=0, pin_memory=True
    )

    results = {}

    # 1. Train SSDLite320 (MobileNetV3-Large)
    torch.cuda.empty_cache()
    ssdlite_model = build_ssdlite320(num_classes=5)
    ssdlite_ckpt = Path("services/ml/model_artifacts/detector_ssdlite320_v2_best.pt")
    ssdlite_results = train_single_detector(
        detector_name="ssdlite320_mobilenet_v3_large_v2",
        model=ssdlite_model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        device=device,
        epochs=35,
        initial_lr=2e-4,
        checkpoint_out=ssdlite_ckpt,
    )
    results["ssdlite320_v2"] = ssdlite_results

    # 2. Train Faster R-CNN (MobileNetV3-Large-320 FPN)
    torch.cuda.empty_cache()
    fasterrcnn_model = build_fasterrcnn_mobilenetv3(num_classes=5)
    fasterrcnn_ckpt = Path("services/ml/model_artifacts/detector_fasterrcnn_mobilenetv3_best.pt")
    fasterrcnn_results = train_single_detector(
        detector_name="fasterrcnn_mobilenet_v3_large_320_fpn",
        model=fasterrcnn_model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        device=device,
        epochs=35,
        initial_lr=1.5e-4,
        checkpoint_out=fasterrcnn_ckpt,
    )
    results["fasterrcnn_mobilenetv3"] = fasterrcnn_results

    # Save complete benchmark results
    out_file = Path("services/ml/model_artifacts/detector_retraining_benchmark_results.json")
    with open(out_file, "w") as f:
        json.dump(results, f, indent=2)

    print("\n" + "=" * 70)
    print("  Detection Retraining Benchmark Complete!  ")
    print("=" * 70)
    print(f"SSDLite320 v2 Test mAP@0.5:      {ssdlite_results['test_map_50']*100:.2f}% (Audited v1 was 8.98%)")
    print(f"Faster R-CNN Test mAP@0.5:        {fasterrcnn_results['test_map_50']*100:.2f}%")
    print(f"Results saved to {out_file}")


if __name__ == "__main__":
    main()

