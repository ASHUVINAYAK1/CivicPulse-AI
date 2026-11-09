"""
CivicPulse AI — Phase 2.2: ResNet-18 Generalization Evaluation
Evaluates the production ResNet-18 model checkpoint against the newly partitioned,
unseen grouped test split (rdd2022_india_grouped_v2/test).
Strictly frozen evaluation: NO training, fine-tuning, or hyperparameter searching.
"""

import sys
import json
import time
from pathlib import Path
from typing import Dict, Any, List

import numpy as np
import torch
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
        mask = (probs >= bin_lower) & (probs < bin_upper) if i < n_bins - 1 else (probs >= bin_lower) & (probs <= bin_upper)
        bin_size = np.sum(mask)

        if bin_size > 0:
            avg_confidence = np.mean(probs[mask])
            avg_accuracy = np.mean(targets[mask])
            ece += (bin_size / total_elements) * np.abs(avg_accuracy - avg_confidence)

    return float(ece)


def main():
    print("=" * 75)
    print("  CivicPulse AI — Phase 2.2 ResNet-18 Generalization Evaluation  ")
    print("=" * 75)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Evaluation Hardware: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")

    test_dir = Path("datasets/splits/rdd2022_india_grouped_v2/test")
    if not test_dir.exists():
        raise FileNotFoundError(f"Grouped v2 test directory not found: {test_dir}")

    print(f"Loading samples from {test_dir}...")
    samples = RDDAnnotationParser.load_yolo_split(test_dir)
    print(f"Loaded {len(samples)} grouped test samples.")

    dataset = GeneralizationDataset(samples)
    test_loader = DataLoader(dataset, batch_size=64, shuffle=False, num_workers=0, pin_memory=True)

    ckpt_path = Path("services/ml/model_artifacts/production_model.pt")
    if not ckpt_path.exists():
        ckpt_path = Path("services/ml/model_artifacts/resnet18_best.pt")
    print(f"Loading frozen checkpoint: {ckpt_path}...")

    checkpoint = torch.load(ckpt_path, map_location=device)
    state_dict = checkpoint["state_dict"] if "state_dict" in checkpoint else checkpoint

    model = RoadDamageClassifier(backbone_name="resnet18", num_classes=4, pretrained=False).to(device)
    model.load_state_dict(state_dict)
    model.eval()

    all_logits = []
    all_targets = []
    all_image_ids = []

    start_t = time.perf_counter()
    with torch.no_grad():
        for imgs, tgts, ids in test_loader:
            imgs = imgs.to(device)
            logits = model(imgs)
            all_logits.append(logits.cpu().numpy())
            all_targets.append(tgts.numpy())
            all_image_ids.extend(ids)
            
    torch.cuda.synchronize() if device.type == "cuda" else None
    eval_time = time.perf_counter() - start_t
    print(f"Inference completed in {eval_time:.2f}s ({len(samples)/eval_time:.1f} images/sec).")

    y_logits = np.vstack(all_logits)
    y_true = np.vstack(all_targets)
    y_prob = 1.0 / (1.0 + np.exp(-y_logits))
    y_pred = (y_prob >= 0.5).astype(int)

    # 1. Per-Class Metrics
    per_class = {}
    tp_total, fp_total, fn_total, tn_total = 0, 0, 0, 0
    f1_list, prec_list, rec_list, support_list = [], [], [], []

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

    macro_f1 = float(np.mean(f1_list))
    macro_prec = float(np.mean(prec_list))
    macro_rec = float(np.mean(rec_list))

    micro_prec = float(tp_total / max(tp_total + fp_total, 1e-6))
    micro_rec = float(tp_total / max(tp_total + fn_total, 1e-6))
    micro_f1 = float((2 * micro_prec * micro_rec) / max(micro_prec + micro_rec, 1e-6))

    total_support = max(sum(support_list), 1)
    weighted_f1 = float(sum(f * s for f, s in zip(f1_list, support_list)) / total_support)
    exact_match_ratio = float(np.mean(np.all(y_pred == y_true, axis=1)))
    hamming_loss = float(np.mean(y_pred != y_true))

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

    ece = compute_calibration_ece(y_prob, y_true, n_bins=10)
    brier_score = float(np.mean((y_prob - y_true) ** 2))

    # D10 threshold sensitivity
    d10_idx = CLASS_NAMES.index("D10")
    d10_probs = y_prob[:, d10_idx]
    d10_targets = y_true[:, d10_idx]
    threshold_sweeps = []
    for th in [0.50, 0.40, 0.30, 0.25, 0.20, 0.15, 0.10, 0.05, 0.02]:
        p_th = (d10_probs >= th).astype(int)
        tp_th = int(np.sum((p_th == 1) & (d10_targets == 1)))
        fp_th = int(np.sum((p_th == 1) & (d10_targets == 0)))
        fn_th = int(np.sum((p_th == 0) & (d10_targets == 1)))
        tn_th = int(np.sum((p_th == 0) & (d10_targets == 0)))
        p_val = float(tp_th / max(tp_th + fp_th, 1e-6))
        r_val = float(tp_th / max(tp_th + fn_th, 1e-6))
        f1_th = float((2 * p_val * r_val) / max(p_val + r_val, 1e-6))
        threshold_sweeps.append({
            "threshold": th,
            "tp": tp_th,
            "fp": fp_th,
            "fn": fn_th,
            "tn": tn_th,
            "precision": round(p_val, 4),
            "recall": round(r_val, 4),
            "f1_score": round(f1_th, 4),
        })

    # Check provenance against raw splits
    raw_train_names = set(p.name for p in Path("datasets/raw/RDD2022_India/train/images").glob("*.jpg"))
    raw_val_names = set(p.name for p in Path("datasets/raw/RDD2022_India/val/images").glob("*.jpg"))
    raw_test_names = set(p.name for p in Path("datasets/raw/RDD2022_India/test/images").glob("*.jpg"))

    provenance_counts = {
        "from_raw_train": sum(1 for s in samples if Path(s.image_path).name in raw_train_names),
        "from_raw_val": sum(1 for s in samples if Path(s.image_path).name in raw_val_names),
        "from_raw_test": sum(1 for s in samples if Path(s.image_path).name in raw_test_names),
    }

    # Also compute strictly unseen subset metrics (only frames in raw_test)
    strictly_unseen_indices = [i for i, s in enumerate(samples) if Path(s.image_path).name in raw_test_names]
    unseen_y_pred = y_pred[strictly_unseen_indices]
    unseen_y_true = y_true[strictly_unseen_indices]
    
    unseen_f1_list = []
    unseen_per_class = {}
    for i, cname in enumerate(CLASS_NAMES):
        tp_u = int(np.sum((unseen_y_pred[:, i] == 1) & (unseen_y_true[:, i] == 1)))
        fp_u = int(np.sum((unseen_y_pred[:, i] == 1) & (unseen_y_true[:, i] == 0)))
        fn_u = int(np.sum((unseen_y_pred[:, i] == 0) & (unseen_y_true[:, i] == 1)))
        tn_u = int(np.sum((unseen_y_pred[:, i] == 0) & (unseen_y_true[:, i] == 0)))
        supp_u = int(np.sum(unseen_y_true[:, i] == 1))
        p_u = float(tp_u / max(tp_u + fp_u, 1e-6))
        r_u = float(tp_u / max(tp_u + fn_u, 1e-6))
        f_u = float((2 * p_u * r_u) / max(p_u + r_u, 1e-6))
        unseen_f1_list.append(f_u)
        unseen_per_class[cname] = {
            "support": supp_u, "tp": tp_u, "fp": fp_u, "fn": fn_u, "tn": tn_u,
            "precision": round(p_u, 4), "recall": round(r_u, 4), "f1_score": round(f_u, 4)
        }
    unseen_macro_f1 = float(np.mean(unseen_f1_list))
    unseen_bin_true = (np.sum(unseen_y_true, axis=1) > 0).astype(int)
    unseen_bin_pred = (np.sum(unseen_y_pred, axis=1) > 0).astype(int)
    unseen_bin_acc = float(np.mean(unseen_bin_true == unseen_bin_pred))

    # Load baseline Phase 2.1 audit metrics for side-by-side comparison
    orig_audit_path = Path("services/ml/model_artifacts/classification_audit_results.json")
    orig_metrics = {}
    if orig_audit_path.exists():
        orig_data = json.loads(orig_audit_path.read_text())
        orig_metrics = orig_data.get("resnet18_production_model", {})

    results = {
        "model_architecture": "resnet18",
        "checkpoint_path": str(ckpt_path),
        "dataset_split": "rdd2022_india_grouped_v2/test",
        "evaluation_timestamp": "2026-11-07",
        "num_test_samples": len(samples),
        "provenance": provenance_counts,
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
            "expected_calibration_error_ece": round(ece, 4),
            "brier_score": round(brier_score, 4),
        },
        "binary_damage_metrics": {
            "accuracy": round(b_acc, 4),
            "precision": round(b_prec, 4),
            "recall": round(b_rec, 4),
            "f1_score": round(b_f1, 4),
            "confusion_matrix": {"tn": b_tn, "fp": b_fp, "fn": b_fn, "tp": b_tp},
        },
        "per_class_metrics": per_class,
        "strictly_unseen_heldout_subset": {
            "sample_count": len(strictly_unseen_indices),
            "macro_f1": round(unseen_macro_f1, 4),
            "binary_accuracy": round(unseen_bin_acc, 4),
            "per_class": unseen_per_class
        },
        "d10_transverse_threshold_sensitivity": threshold_sweeps,
        "baseline_comparison_phase_2_1": {
            "original_macro_f1": orig_metrics.get("overall_metrics", {}).get("macro_f1"),
            "original_micro_f1": orig_metrics.get("overall_metrics", {}).get("micro_f1"),
            "original_exact_match": orig_metrics.get("overall_metrics", {}).get("exact_match_ratio"),
            "original_binary_accuracy": orig_metrics.get("binary_damage_metrics", {}).get("accuracy"),
            "original_binary_f1": orig_metrics.get("binary_damage_metrics", {}).get("f1_score"),
            "delta_macro_f1_full": round(macro_f1 - (orig_metrics.get("overall_metrics", {}).get("macro_f1") or 0.0), 4),
            "delta_macro_f1_strictly_unseen": round(unseen_macro_f1 - (orig_metrics.get("overall_metrics", {}).get("macro_f1") or 0.0), 4),
            "delta_binary_accuracy_strictly_unseen": round(unseen_bin_acc - (orig_metrics.get("binary_damage_metrics", {}).get("accuracy") or 0.0), 4),
        }
    }

    orig_bin = orig_metrics.get("binary_damage_detection", {})
    results["baseline_comparison_phase_2_1"]["original_binary_accuracy"] = orig_bin.get("accuracy")
    results["baseline_comparison_phase_2_1"]["original_binary_f1"] = orig_bin.get("f1_score")
    results["baseline_comparison_phase_2_1"]["delta_binary_accuracy_strictly_unseen"] = round(
        unseen_bin_acc - (orig_bin.get("accuracy") or 0.0), 4
    )

    out_file = Path("services/ml/model_artifacts/resnet18_generalization_v2_results.json")
    with open(out_file, "w") as f:
        json.dump(results, f, indent=2)

    print("\n" + "=" * 55)
    print("  ResNet-18 Grouped v2 Generalization Benchmark Summary")
    print("=" * 55)
    print(f"Macro F1 Score:        {macro_f1:.4f} (Original Random Split: {orig_metrics.get('overall_metrics', {}).get('macro_f1', 'N/A')})")
    print(f"Micro F1 Score:        {micro_f1:.4f} (Original Random Split: {orig_metrics.get('overall_metrics', {}).get('micro_f1', 'N/A')})")
    print(f"Exact Match Ratio:     {exact_match_ratio*100:.2f}% (Original Random Split: {orig_metrics.get('overall_metrics', {}).get('exact_match_ratio', 0)*100:.2f}%)")
    print(f"Binary Road Accuracy:  {b_acc*100:.2f}% (Original Random Split: {orig_metrics.get('binary_damage_metrics', {}).get('accuracy', 0)*100:.2f}%)")
    print(f"Binary Precision:      {b_prec*100:.2f}%")
    print(f"Binary Recall:         {b_rec*100:.2f}%")
    print(f"Binary F1 Score:       {b_f1:.4f}")
    print(f"Expected Calib (ECE):  {ece:.4f}")
    print(f"Brier Score:           {brier_score:.4f}")

    print("\nPer-Class Breakdown:")
    for cname in CLASS_NAMES:
        m = per_class[cname]
        print(f"  {cname} ({m['name']}): Support={m['support']}, P={m['precision']*100:.1f}%, R={m['recall']*100:.1f}%, F1={m['f1_score']:.4f}")

    print(f"\nResults saved to {out_file}")


if __name__ == "__main__":
    main()
