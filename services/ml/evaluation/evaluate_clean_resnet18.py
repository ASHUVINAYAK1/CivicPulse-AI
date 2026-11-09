"""
CivicPulse AI — Phase 2.3: Clean ResNet-18 Generalization Evaluation
Evaluates the clean ResNet-18 model (trained exclusively on grouped v2 train/val)
against the grouped v2 test split (datasets/splits/rdd2022_india_grouped_v2/test).

Computes comprehensive metrics:
- Macro F1, Micro F1, Weighted F1
- Per-class Precision, Recall, F1, Support, Specificity
- Binary Damage Detection (Accuracy, Precision, Recall, F1, Confusion Matrix)
- 2x2 Class Confusion Matrices
- Calibration Metrics (ECE, Brier Score)
- Exact Match Ratio, Hamming Loss
- Detailed 3-Way Comparative Benchmark:
  1. Historical ResNet-18 on Original Random Split
  2. Historical ResNet-18 on Grouped Split (72.7% training overlap)
  3. Clean ResNet-18 on Grouped v2 Split (0.00% overlap, clean lineage)
"""

import sys
import json
import time
import hashlib
from pathlib import Path
from typing import Dict, Any, List

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


def get_file_sha256(filepath: Path) -> str:
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


class GeneralizationDataset(Dataset):
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
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    total_elements = probs.size

    for i in range(n_bins):
        bin_lower = bin_boundaries[i]
        bin_upper = bin_boundaries[i + 1]
        mask = (
            (probs >= bin_lower) & (probs < bin_upper)
            if i < n_bins - 1
            else (probs >= bin_lower) & (probs <= bin_upper)
        )
        bin_size = np.sum(mask)

        if bin_size > 0:
            avg_confidence = np.mean(probs[mask])
            avg_accuracy = np.mean(targets[mask])
            ece += (bin_size / total_elements) * np.abs(avg_accuracy - avg_confidence)

    return float(ece)


def compute_brier_score(probs: np.ndarray, targets: np.ndarray) -> float:
    return float(np.mean((probs - targets) ** 2))


def evaluate_clean_model(
    checkpoint_path: Path = Path("services/ml/model_artifacts/resnet18_clean_grouped_v2_best.pt"),
    test_dir: Path = Path("datasets/splits/rdd2022_india_grouped_v2/test"),
    output_path: Path = Path("services/ml/model_artifacts/resnet18_clean_generalization_results.json"),
) -> Dict[str, Any]:
    print("\n" + "=" * 75)
    print("  CivicPulse AI — Phase 2.3 Clean ResNet-18 Generalization Evaluation")
    print("=" * 75)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Evaluation Hardware: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Clean checkpoint not found at: {checkpoint_path}")

    ckpt_sha256 = get_file_sha256(checkpoint_path)
    print(f"Checkpoint Path: {checkpoint_path}")
    print(f"SHA-256 Checksum: {ckpt_sha256}")

    # Load checkpoint
    ckpt = torch.load(checkpoint_path, map_location=device)
    model = RoadDamageClassifier(backbone_name="resnet18", num_classes=4, pretrained=False).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    print(f"Loaded weights from epoch {ckpt.get('best_epoch', 'N/A')} with Val Macro F1: {ckpt.get('best_val_macro_f1', 'N/A'):.4f}")

    # Load test split
    print(f"Loading test split from: {test_dir}")
    test_samples = RDDAnnotationParser.load_yolo_split(test_dir)
    print(f"Loaded {len(test_samples)} test samples.")

    dataset = GeneralizationDataset(test_samples)
    test_loader = DataLoader(dataset, batch_size=64, shuffle=False, num_workers=0, pin_memory=torch.cuda.is_available())

    # Inference loop
    all_probs = []
    all_targets = []
    all_ids = []
    latencies = []

    print("Executing evaluation forward passes...")
    start_time = time.time()

    with torch.no_grad():
        for imgs, tgts, ids in test_loader:
            imgs = imgs.to(device, non_blocking=True)
            tgts = tgts.to(device, non_blocking=True)

            t0 = time.perf_counter()
            logits = model(imgs)
            torch.cuda.synchronize() if device.type == "cuda" else None
            t1 = time.perf_counter()

            latencies.append((t1 - t0) * 1000.0 / imgs.size(0))
            probs = torch.sigmoid(logits).cpu().numpy()

            all_probs.append(probs)
            all_targets.append(tgts.cpu().numpy())
            all_ids.extend(ids)

    total_eval_time = time.time() - start_time
    y_probs = np.vstack(all_probs)
    y_true = np.vstack(all_targets)
    y_pred = (y_probs >= 0.5).astype(int)

    avg_latency_ms = float(np.mean(latencies))
    p95_latency_ms = float(np.percentile(latencies, 95))
    print(f"Inference finished in {total_eval_time:.2f}s (Mean latency: {avg_latency_ms:.2f} ms/frame, p95: {p95_latency_ms:.2f} ms)")

    # Per-Class Metrics
    per_class = {}
    f1_list, prec_list, rec_list, supports = [], [], [], []

    for i, cname in enumerate(CLASS_NAMES):
        tp = int(np.sum((y_pred[:, i] == 1) & (y_true[:, i] == 1)))
        fp = int(np.sum((y_pred[:, i] == 1) & (y_true[:, i] == 0)))
        fn = int(np.sum((y_pred[:, i] == 0) & (y_true[:, i] == 1)))
        tn = int(np.sum((y_pred[:, i] == 0) & (y_true[:, i] == 0)))
        support = int(np.sum(y_true[:, i] == 1))

        prec = float(tp / max(tp + fp, 1e-6))
        rec = float(tp / max(tp + fn, 1e-6))
        f1 = float((2 * prec * rec) / max(prec + rec, 1e-6))
        spec = float(tn / max(tn + fp, 1e-6))

        f1_list.append(f1)
        prec_list.append(prec)
        rec_list.append(rec)
        supports.append(support)

        # Probabilities on true positives
        pos_indices = np.where(y_true[:, i] == 1)[0]
        max_prob_pos = float(np.max(y_probs[pos_indices, i])) if len(pos_indices) > 0 else 0.0
        mean_prob_pos = float(np.mean(y_probs[pos_indices, i])) if len(pos_indices) > 0 else 0.0

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
            "specificity": round(spec, 4),
            "mean_prob_on_positive": round(mean_prob_pos, 4),
            "max_prob_on_positive": round(max_prob_pos, 4),
        }

    macro_f1 = float(np.mean(f1_list))
    macro_prec = float(np.mean(prec_list))
    macro_rec = float(np.mean(rec_list))

    # Micro F1
    tot_tp = sum(per_class[c]["tp"] for c in CLASS_NAMES)
    tot_fp = sum(per_class[c]["fp"] for c in CLASS_NAMES)
    tot_fn = sum(per_class[c]["fn"] for c in CLASS_NAMES)
    micro_prec = float(tot_tp / max(tot_tp + tot_fp, 1e-6))
    micro_rec = float(tot_tp / max(tot_tp + tot_fn, 1e-6))
    micro_f1 = float((2 * micro_prec * micro_rec) / max(micro_prec + micro_rec, 1e-6))

    # Weighted F1
    tot_support = sum(supports)
    weighted_f1 = (
        float(sum(f1_list[i] * supports[i] for i in range(len(CLASS_NAMES))) / max(tot_support, 1))
        if tot_support > 0
        else 0.0
    )

    # Calibration
    ece = compute_calibration_ece(y_probs, y_true, n_bins=10)
    brier = compute_brier_score(y_probs, y_true)

    # Multi-label match metrics
    exact_match = float(np.mean(np.all(y_pred == y_true, axis=1)))
    hamming_loss = float(np.mean(y_pred != y_true))

    # Binary road damage
    any_true = (np.sum(y_true, axis=1) > 0).astype(int)
    any_pred = (np.sum(y_pred, axis=1) > 0).astype(int)
    bin_tp = int(np.sum((any_pred == 1) & (any_true == 1)))
    bin_fp = int(np.sum((any_pred == 1) & (any_true == 0)))
    bin_fn = int(np.sum((any_pred == 0) & (any_true == 1)))
    bin_tn = int(np.sum((any_pred == 0) & (any_true == 0)))
    bin_acc = float(np.mean(any_true == any_pred))
    bin_prec = float(bin_tp / max(bin_tp + bin_fp, 1e-6))
    bin_rec = float(bin_tp / max(bin_tp + bin_fn, 1e-6))
    bin_f1 = float((2 * bin_prec * bin_rec) / max(bin_prec + bin_rec, 1e-6))

    # Load historical benchmarks for 3-way comparison
    historical_split_results = {
        "macro_f1": 0.4529,
        "micro_f1": 0.6379,
        "weighted_f1": 0.6288,
        "binary_acc": 0.8199,
        "binary_f1": 0.7433,
        "exact_match": 0.7170,
        "ece": 0.0295,
        "brier": 0.0637,
        "d00_f1": 0.5230,
        "d10_f1": 0.0000,
        "d20_f1": 0.7885,
        "d40_f1": 0.5000,
    }

    historical_on_grouped_v2 = {
        "macro_f1": 0.7640,
        "micro_f1": 0.8883,
        "weighted_f1": 0.8848,
        "binary_acc": 0.9379,
        "binary_f1": 0.9252,
        "exact_match": 0.8997,
        "ece": 0.0109,
        "brier": 0.0235,
        "d00_f1": 0.8797,
        "d10_f1": 0.4000,
        "d20_f1": 0.9295,
        "d40_f1": 0.8467,
        "training_overlap_percentage": 72.7,
        "overlap_samples": 761,
        "total_samples": 1047,
    }

    # Summary payload
    results = {
        "candidate_model": "resnet18_clean_grouped_v2",
        "checkpoint_path": str(checkpoint_path),
        "sha256": ckpt_sha256,
        "evaluated_dataset": "datasets/splits/rdd2022_india_grouped_v2/test",
        "test_sample_count": len(test_samples),
        "training_overlap_with_test": 0,
        "training_overlap_percentage": 0.0,
        "overall_metrics": {
            "macro_f1": round(macro_f1, 4),
            "macro_precision": round(macro_prec, 4),
            "macro_recall": round(macro_rec, 4),
            "micro_f1": round(micro_f1, 4),
            "micro_precision": round(micro_prec, 4),
            "micro_recall": round(micro_rec, 4),
            "weighted_f1": round(weighted_f1, 4),
            "exact_match_ratio": round(exact_match, 4),
            "hamming_loss": round(hamming_loss, 4),
            "expected_calibration_error_ece": round(ece, 4),
            "brier_score": round(brier, 4),
            "avg_latency_ms": round(avg_latency_ms, 2),
            "p95_latency_ms": round(p95_latency_ms, 2),
        },
        "binary_damage_metrics": {
            "accuracy": round(bin_acc, 4),
            "precision": round(bin_prec, 4),
            "recall": round(bin_rec, 4),
            "f1_score": round(bin_f1, 4),
            "confusion_matrix": {
                "tn": bin_tn,
                "fp": bin_fp,
                "fn": bin_fn,
                "tp": bin_tp,
            },
        },
        "per_class_metrics": per_class,
        "comparative_benchmarks": {
            "historical_resnet18_random_split": historical_split_results,
            "historical_resnet18_grouped_split_with_72_7pct_overlap": historical_on_grouped_v2,
            "clean_resnet18_grouped_v2_zero_overlap": {
                "macro_f1": round(macro_f1, 4),
                "micro_f1": round(micro_f1, 4),
                "weighted_f1": round(weighted_f1, 4),
                "binary_acc": round(bin_acc, 4),
                "binary_f1": round(bin_f1, 4),
                "exact_match": round(exact_match, 4),
                "ece": round(ece, 4),
                "brier": round(brier, 4),
                "d00_f1": per_class["D00"]["f1_score"],
                "d10_f1": per_class["D10"]["f1_score"],
                "d20_f1": per_class["D20"]["f1_score"],
                "d40_f1": per_class["D40"]["f1_score"],
            },
        },
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nSaved evaluation results to: {output_path}")

    # Print Pretty Comparative Table
    print("\n" + "=" * 90)
    print("  3-WAY CLASSIFIER GENERALIZATION BENCHMARK SUMMARY")
    print("=" * 90)
    print(f"{'Metric':<28} | {'1. Hist (Random Split)':<22} | {'2. Hist (Grouped, 72.7% Overlap)':<32} | {'3. Clean ResNet-18 (0% Overlap)':<25}")
    print("-" * 90)
    print(f"{'Macro F1 Score':<28} | {historical_split_results['macro_f1']:<22.4f} | {historical_on_grouped_v2['macro_f1']:<32.4f} | {macro_f1:<25.4f}")
    print(f"{'Micro F1 Score':<28} | {historical_split_results['micro_f1']:<22.4f} | {historical_on_grouped_v2['micro_f1']:<32.4f} | {micro_f1:<25.4f}")
    print(f"{'Weighted F1 Score':<28} | {historical_split_results['weighted_f1']:<22.4f} | {historical_on_grouped_v2['weighted_f1']:<32.4f} | {weighted_f1:<25.4f}")
    print(f"{'Exact Match Ratio':<28} | {historical_split_results['exact_match']*100:<21.2f}% | {historical_on_grouped_v2['exact_match']*100:<31.2f}% | {exact_match*100:<24.2f}%")
    print(f"{'Binary Road Accuracy':<28} | {historical_split_results['binary_acc']*100:<21.2f}% | {historical_on_grouped_v2['binary_acc']*100:<31.2f}% | {bin_acc*100:<24.2f}%")
    print(f"{'Binary Damage F1':<28} | {historical_split_results['binary_f1']:<22.4f} | {historical_on_grouped_v2['binary_f1']:<32.4f} | {bin_f1:<25.4f}")
    print(f"{'Expected Calibration (ECE)':<28} | {historical_split_results['ece']:<22.4f} | {historical_on_grouped_v2['ece']:<32.4f} | {ece:<25.4f}")
    print(f"{'Brier Calibration Score':<28} | {historical_split_results['brier']:<22.4f} | {historical_on_grouped_v2['brier']:<32.4f} | {brier:<25.4f}")
    print("-" * 90)
    print(f"{'D00 Longitudinal F1':<28} | {historical_split_results['d00_f1']:<22.4f} | {historical_on_grouped_v2['d00_f1']:<32.4f} | {per_class['D00']['f1_score']:<25.4f}")
    print(f"{'D10 Transverse F1':<28} | {historical_split_results['d10_f1']:<22.4f} | {historical_on_grouped_v2['d10_f1']:<32.4f} | {per_class['D10']['f1_score']:<25.4f}")
    print(f"{'D20 Alligator Crack F1':<28} | {historical_split_results['d20_f1']:<22.4f} | {historical_on_grouped_v2['d20_f1']:<32.4f} | {per_class['D20']['f1_score']:<25.4f}")
    print(f"{'D40 Critical Pothole F1':<28} | {historical_split_results['d40_f1']:<22.4f} | {historical_on_grouped_v2['d40_f1']:<32.4f} | {per_class['D40']['f1_score']:<25.4f}")
    print("=" * 90)

    return results


if __name__ == "__main__":
    evaluate_clean_model()

