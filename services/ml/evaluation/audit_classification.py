"""
CivicPulse AI — Phase 2.1 Classification Independent Audit Script
Evaluates saved model checkpoints on the held-out test split of RDD2022 India.
Does not alter models, checkpoints, or dataset.
Computes per-class P/R/F1, Macro/Micro/Weighted F1, Binary metrics, Confusion matrices, and Calibration (ECE).
"""

import sys
import json
import time
from pathlib import Path
from typing import Dict, Any, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from preprocessing.rdd2022_parser import RDDAnnotationParser, RoadDamageSample
from preprocessing.image_pipeline import ImagePreprocessor
from models.transfer_learning.classifier import RoadDamageClassifier

CLASS_NAMES = ["D00", "D10", "D20", "D40"]
CLASS_LABELS = {
    "D00": "Longitudinal Crack",
    "D10": "Transverse Crack",
    "D20": "Alligator Crack",
    "D40": "Pothole",
}


class AuditTestDataset(Dataset):
    def __init__(self, samples: List[RoadDamageSample], target_size=(224, 224)):
        self.samples = samples
        self.preprocessor = ImagePreprocessor(target_size=target_size, normalize=True)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        img_path = Path(sample.image_path)
        with Image.open(img_path) as img:
            pil_img = img.convert("RGB")

        tensor_np = self.preprocessor.preprocess_image(pil_img, augment=False)
        target = torch.zeros(4, dtype=torch.float32)
        for dt in sample.damage_types:
            if dt in CLASS_NAMES:
                target[CLASS_NAMES.index(dt)] = 1.0

        return torch.from_numpy(tensor_np), target, sample.image_id


def compute_calibration_ece(probs: np.ndarray, targets: np.ndarray, n_bins: int = 10) -> float:
    """Computes Expected Calibration Error (ECE) for multi-label binary probabilities."""
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    total_elements = probs.size

    for i in range(n_bins):
        bin_lower = bin_boundaries[i]
        bin_upper = bin_boundaries[i + 1]
        mask = (probs >= bin_lower) & (probs < bin_upper) if i < n_bins - 1 else (probs >= bin_lower) & (probs <= bin_upper)
        bin_size = np.sum(mask)

        if bin_size > 0:
            avg_confidence = np.mean(probs[mask])
            avg_accuracy = np.mean(targets[mask])
            ece += (bin_size / total_elements) * np.abs(avg_accuracy - avg_confidence)

    return float(ece)


def audit_model(backbone_name: str, checkpoint_path: Path, test_loader: DataLoader, device: torch.device) -> Dict[str, Any]:
    print(f"\n--- Auditing Checkpoint: {checkpoint_path.name} ({backbone_name}) ---")
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint["state_dict"] if "state_dict" in checkpoint else checkpoint

    model = RoadDamageClassifier(backbone_name=backbone_name, num_classes=4, pretrained=False).to(device)
    model.load_state_dict(state_dict)
    model.eval()

    all_logits = []
    all_targets = []
    all_image_ids = []

    with torch.no_grad():
        for imgs, tgts, ids in test_loader:
            imgs = imgs.to(device)
            logits = model(imgs)
            all_logits.append(logits.cpu().numpy())
            all_targets.append(tgts.numpy())
            all_image_ids.extend(ids)

    y_logits = np.vstack(all_logits)
    y_true = np.vstack(all_targets)
    # Apply Sigmoid
    y_prob = 1.0 / (1.0 + np.exp(-y_logits))

    # Evaluate at threshold = 0.5
    y_pred = (y_prob >= 0.5).astype(int)

    # 1. Per-Class Metrics and 2x2 Confusion Matrices
    per_class = {}
    tp_total, fp_total, fn_total, tn_total = 0, 0, 0, 0
    f1_list = []
    prec_list = []
    rec_list = []
    support_list = []

    for i, cname in enumerate(CLASS_NAMES):
        tp = int(np.sum((y_pred[:, i] == 1) & (y_true[:, i] == 1)))
        fp = int(np.sum((y_pred[:, i] == 1) & (y_true[:, i] == 0)))
        fn = int(np.sum((y_pred[:, i] == 0) & (y_true[:, i] == 1)))
        tn = int(np.sum((y_pred[:, i] == 0) & (y_true[:, i] == 0)))
        support = int(np.sum(y_true[:, i] == 1))

        prec = float(tp / max(tp + fp, 1e-6))
        rec = float(tp / max(tp + fn, 1e-6))
        f1 = float((2 * prec * rec) / max(prec + rec, 1e-6))
        specificity = float(tn / max(tn + fp, 1e-6))

        # Max, mean, median predicted probabilities for positive samples
        pos_probs = y_prob[y_true[:, i] == 1, i]
        max_prob_pos = float(np.max(pos_probs)) if len(pos_probs) > 0 else 0.0
        mean_prob_pos = float(np.mean(pos_probs)) if len(pos_probs) > 0 else 0.0

        per_class[cname] = {
            "name": CLASS_LABELS[cname],
            "support": support,
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "tn": tn,
            "confusion_matrix_2x2": {"tn": tn, "fp": fp, "fn": fn, "tp": tp},
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "f1_score": round(f1, 4),
            "specificity": round(specificity, 4),
            "max_prob_on_positive": round(max_prob_pos, 4),
            "mean_prob_on_positive": round(mean_prob_pos, 4),
        }

        f1_list.append(f1)
        prec_list.append(prec)
        rec_list.append(rec)
        support_list.append(support)

        tp_total += tp
        fp_total += fp
        fn_total += fn
        tn_total += tn

    # Macro Metrics
    macro_f1 = float(np.mean(f1_list))
    macro_prec = float(np.mean(prec_list))
    macro_rec = float(np.mean(rec_list))

    # Micro Metrics
    micro_prec = float(tp_total / max(tp_total + fp_total, 1e-6))
    micro_rec = float(tp_total / max(tp_total + fn_total, 1e-6))
    micro_f1 = float((2 * micro_prec * micro_rec) / max(micro_prec + micro_rec, 1e-6))

    # Weighted F1
    total_support = max(sum(support_list), 1)
    weighted_f1 = float(sum(f * s for f, s in zip(f1_list, support_list)) / total_support)

    # Exact Match Ratio (All 4 labels match exactly)
    exact_match_ratio = float(np.mean(np.all(y_pred == y_true, axis=1)))

    # Hamming Loss (Fraction of incorrect labels out of total predictions)
    hamming_loss = float(np.mean(y_pred != y_true))

    # Binary Classification (Damage vs Clean road)
    binary_true = (np.sum(y_true, axis=1) > 0).astype(int)
    binary_pred = (np.sum(y_pred, axis=1) > 0).astype(int)

    b_tp = int(np.sum((binary_pred == 1) & (binary_true == 1)))
    b_fp = int(np.sum((binary_pred == 1) & (binary_true == 0)))
    b_fn = int(np.sum((binary_pred == 0) & (binary_true == 1)))
    b_tn = int(np.sum((binary_pred == 0) & (binary_true == 0)))

    b_acc = float(np.mean(binary_pred == binary_true))
    b_prec = float(b_tp / max(b_tp + b_fp, 1e-6))
    b_rec = float(b_tp / max(b_tp + b_fn, 1e-6))
    b_f1 = float((2 * b_prec * b_rec) / max(b_prec + b_rec, 1e-6))

    # Expected Calibration Error (ECE) and Brier Score
    ece = compute_calibration_ece(y_prob, y_true, n_bins=10)
    brier_score = float(np.mean((y_prob - y_true) ** 2))

    # Threshold sensitivity study for D10 (Transverse Crack)
    d10_idx = CLASS_NAMES.index("D10")
    d10_probs = y_prob[:, d10_idx]
    d10_true = y_true[:, d10_idx]
    thresholds = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50]
    d10_sweep = []
    for th in thresholds:
        p_th = (d10_probs >= th).astype(int)
        tp_th = int(np.sum((p_th == 1) & (d10_true == 1)))
        fp_th = int(np.sum((p_th == 1) & (d10_true == 0)))
        fn_th = int(np.sum((p_th == 0) & (d10_true == 1)))
        prec_th = float(tp_th / max(tp_th + fp_th, 1e-6))
        rec_th = float(tp_th / max(tp_th + fn_th, 1e-6))
        f1_th = float((2 * prec_th * rec_th) / max(prec_th + rec_th, 1e-6))
        d10_sweep.append({
            "threshold": th,
            "tp": tp_th,
            "fp": fp_th,
            "fn": fn_th,
            "precision": round(prec_th, 4),
            "recall": round(rec_th, 4),
            "f1": round(f1_th, 4),
        })

    return {
        "model_name": backbone_name,
        "checkpoint": checkpoint_path.name,
        "num_test_samples": len(y_true),
        "overall_metrics": {
            "macro_f1": round(macro_f1, 4),
            "macro_precision": round(macro_prec, 4),
            "macro_recall": round(macro_rec, 4),
            "micro_f1": round(micro_f1, 4),
            "micro_precision": round(micro_prec, 4),
            "micro_recall": round(micro_rec, 4),
            "weighted_f1": round(weighted_f1, 4),
            "exact_match_ratio": round(exact_match_ratio, 4),
            "hamming_loss": round(hamming_loss, 4),
            "ece": round(ece, 4),
            "brier_score": round(brier_score, 4),
        },
        "binary_damage_detection": {
            "accuracy": round(b_acc, 4),
            "precision": round(b_prec, 4),
            "recall": round(b_rec, 4),
            "f1_score": round(b_f1, 4),
            "tp": b_tp,
            "fp": b_fp,
            "fn": b_fn,
            "tn": b_tn,
        },
        "per_class_metrics": per_class,
        "d10_threshold_investigation": {
            "d10_support_in_test": int(np.sum(d10_true == 1)),
            "d10_max_predicted_prob": round(float(np.max(d10_probs)), 4),
            "d10_mean_predicted_prob": round(float(np.mean(d10_probs)), 6),
            "d10_prob_percentiles": {
                "p50": round(float(np.percentile(d10_probs, 50)), 6),
                "p90": round(float(np.percentile(d10_probs, 90)), 6),
                "p99": round(float(np.percentile(d10_probs, 99)), 6),
                "max": round(float(np.max(d10_probs)), 6),
            },
            "threshold_sweep": d10_sweep,
        },
    }


def main():
    print("=" * 70)
    print("  CivicPulse AI — Phase 2.1 Classification Independent Audit  ")
    print("=" * 70)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Audit compute engine: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")

    test_dir = Path("datasets/raw/RDD2022_India/test")
    test_samples = RDDAnnotationParser.load_yolo_split(test_dir)
    print(f"Loaded {len(test_samples)} held-out test samples from {test_dir}")

    test_dataset = AuditTestDataset(test_samples)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=0)

    artifacts_dir = Path("services/ml/model_artifacts")
    models = [
        ("resnet18", artifacts_dir / "resnet18_best.pt"),
        ("efficientnet_b0", artifacts_dir / "efficientnet_b0_best.pt"),
        ("mobilenet_v3_small", artifacts_dir / "mobilenet_v3_small_best.pt"),
        ("resnet18", artifacts_dir / "production_model.pt"),
    ]

    audit_results = {}
    for name, path in models:
        key = f"{name}_{path.stem}"
        if key in audit_results:
            continue
        res = audit_model(name, path, test_loader, device)
        audit_results[key] = res

    # Check identity of production_model.pt vs resnet18_best.pt
    prod_ckpt = torch.load(artifacts_dir / "production_model.pt", map_location="cpu")
    res_ckpt = torch.load(artifacts_dir / "resnet18_best.pt", map_location="cpu")
    weights_identical = True
    for k in prod_ckpt["state_dict"]:
        if not torch.equal(prod_ckpt["state_dict"][k], res_ckpt["state_dict"][k]):
            weights_identical = False
            break
    print(f"\nProduction model state_dict matches resnet18_best.pt exactly: {weights_identical}")
    audit_results["production_identity_verified"] = weights_identical

    out_file = artifacts_dir / "classification_audit_results.json"
    with open(out_file, "w") as f:
        json.dump(audit_results, f, indent=2)

    print(f"\nAudit complete. Results saved to {out_file}")


if __name__ == "__main__":
    main()

