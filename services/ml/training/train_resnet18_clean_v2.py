"""
CivicPulse AI — Phase 2.3: Clean ResNet-18 Training on Grouped v2 Split
Trains a fresh ResNet-18 multi-label classifier exclusively on:
- Training:   datasets/splits/rdd2022_india_grouped_v2/train
- Validation: datasets/splits/rdd2022_india_grouped_v2/val

Guarantees:
- Never accesses or evaluates test labels during training or checkpoint selection.
- Multi-label target representation with BCEWithLogitsLoss.
- Principled class imbalance handling via square-root damped pos_weight from training set.
- Early stopping based strictly on Validation Macro F1.
- Tracks all parameters, metrics, and checkpoints in MLflow.
- Saves checkpoint to services/ml/model_artifacts/resnet18_clean_grouped_v2_best.pt
  without overwriting production_model.pt.
"""

import os
import sys
import time
import json
import hashlib
from pathlib import Path
from typing import Dict, Any, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import mlflow

# Add services/ml to path
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


class CleanRoadDamageDataset(Dataset):
    """Memory-efficient PyTorch Dataset for clean RDD2022 samples."""

    def __init__(
        self,
        samples: List[RoadDamageSample],
        target_size: Tuple[int, int] = (224, 224),
        augment: bool = False,
    ):
        self.samples = samples
        self.target_size = target_size
        self.augment = augment
        self.preprocessor = ImagePreprocessor(target_size=target_size, normalize=True)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        sample = self.samples[idx]
        img_path = Path(sample.image_path)
        try:
            with Image.open(img_path) as img:
                pil_img = img.convert("RGB")
        except Exception:
            pil_img = Image.new("RGB", self.target_size, color=(128, 128, 128))

        tensor_np = self.preprocessor.preprocess_image(pil_img, augment=self.augment)
        image_tensor = torch.from_numpy(tensor_np)

        target = torch.zeros(4, dtype=torch.float32)
        for dt in sample.damage_types:
            if dt in CLASS_NAMES:
                target[CLASS_NAMES.index(dt)] = 1.0

        return image_tensor, target


def compute_multilabel_metrics(y_true: np.ndarray, y_pred_prob: np.ndarray, threshold: float = 0.5) -> Dict[str, Any]:
    """Computes macro F1, micro F1, weighted F1, per-class metrics, and binary damage accuracy."""
    y_pred = (y_pred_prob >= threshold).astype(int)
    num_samples = len(y_true)

    f1_list = []
    prec_list = []
    rec_list = []
    per_class = {}
    supports = []

    for i, cname in enumerate(CLASS_NAMES):
        tp = int(np.sum((y_pred[:, i] == 1) & (y_true[:, i] == 1)))
        fp = int(np.sum((y_pred[:, i] == 1) & (y_true[:, i] == 0)))
        fn = int(np.sum((y_pred[:, i] == 0) & (y_true[:, i] == 1)))
        tn = int(np.sum((y_pred[:, i] == 0) & (y_true[:, i] == 0)))
        support = int(np.sum(y_true[:, i] == 1))
        supports.append(support)

        prec = float(tp / max(tp + fp, 1e-6))
        rec = float(tp / max(tp + fn, 1e-6))
        f1 = float((2 * prec * rec) / max(prec + rec, 1e-6))

        f1_list.append(f1)
        prec_list.append(prec)
        rec_list.append(rec)

        per_class[cname] = {
            "name": CLASS_LABELS[cname],
            "support": support,
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "tn": tn,
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "f1_score": round(f1, 4),
        }

    macro_f1 = float(np.mean(f1_list))
    macro_prec = float(np.mean(prec_list))
    macro_rec = float(np.mean(rec_list))

    # Micro F1
    total_tp = sum(per_class[c]["tp"] for c in CLASS_NAMES)
    total_fp = sum(per_class[c]["fp"] for c in CLASS_NAMES)
    total_fn = sum(per_class[c]["fn"] for c in CLASS_NAMES)
    micro_prec = float(total_tp / max(total_tp + total_fp, 1e-6))
    micro_rec = float(total_tp / max(total_tp + total_fn, 1e-6))
    micro_f1 = float((2 * micro_prec * micro_rec) / max(micro_prec + micro_rec, 1e-6))

    # Weighted F1
    total_support = sum(supports)
    weighted_f1 = (
        float(sum(f1_list[i] * supports[i] for i in range(len(CLASS_NAMES))) / max(total_support, 1))
        if total_support > 0
        else 0.0
    )

    # Exact match ratio
    exact_acc = float(np.mean(np.all(y_pred == y_true, axis=1)))

    # Binary road damage detection
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

    return {
        "macro_f1": round(macro_f1, 4),
        "macro_precision": round(macro_prec, 4),
        "macro_recall": round(macro_rec, 4),
        "micro_f1": round(micro_f1, 4),
        "micro_precision": round(micro_prec, 4),
        "micro_recall": round(micro_rec, 4),
        "weighted_f1": round(weighted_f1, 4),
        "exact_match_ratio": round(exact_acc, 4),
        "binary_damage": {
            "accuracy": round(bin_acc, 4),
            "precision": round(bin_prec, 4),
            "recall": round(bin_rec, 4),
            "f1_score": round(bin_f1, 4),
            "tp": bin_tp,
            "fp": bin_fp,
            "fn": bin_fn,
            "tn": bin_tn,
        },
        "per_class": per_class,
    }


def train_clean_resnet18(
    epochs: int = 15,
    batch_size: int = 64,
    lr: float = 3e-4,
    patience: int = 4,
    artifact_dir: Path = Path("services/ml/model_artifacts"),
) -> Dict[str, Any]:
    print("\n" + "=" * 70)
    print("  CivicPulse AI — Phase 2.3: Clean ResNet-18 Training (Grouped v2)")
    print("=" * 70)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Compute Hardware: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")

    artifact_dir.mkdir(parents=True, exist_ok=True)
    clean_checkpoint_path = artifact_dir / "resnet18_clean_grouped_v2_best.pt"

    # 1. Load Grouped v2 Train and Val Splits ONLY
    train_dir = Path("datasets/splits/rdd2022_india_grouped_v2/train")
    val_dir = Path("datasets/splits/rdd2022_india_grouped_v2/val")

    print(f"Loading training samples from: {train_dir}")
    train_samples = RDDAnnotationParser.load_yolo_split(train_dir)
    print(f"Loading validation samples from: {val_dir}")
    val_samples = RDDAnnotationParser.load_yolo_split(val_dir)

    print(f"Clean Partition Counts -> Train: {len(train_samples)}, Val: {len(val_samples)}")

    # 2. Compute training class imbalance weights
    num_train = len(train_samples)
    class_pos_counts = {c: 0 for c in CLASS_NAMES}
    for s in train_samples:
        for dt in s.damage_types:
            if dt in class_pos_counts:
                class_pos_counts[dt] += 1

    print("\nTraining Class Distribution:")
    raw_pos_weights = []
    damped_pos_weights = []
    for c in CLASS_NAMES:
        pos = class_pos_counts[c]
        neg = num_train - pos
        raw_pw = neg / max(pos, 1)
        # Apply square-root damping with upper bound to stabilize D10 gradients
        damped_pw = float(min(np.sqrt(raw_pw) * 1.5, 12.0))
        raw_pos_weights.append(raw_pw)
        damped_pos_weights.append(damped_pw)
        print(f"  {c} ({CLASS_LABELS[c]}): pos={pos} ({pos/num_train*100:.2f}%), raw_pw={raw_pw:.2f}, damped_pw={damped_pw:.2f}")

    pos_weight_tensor = torch.tensor(damped_pos_weights, dtype=torch.float32, device=device)

    # 3. Create DataLoaders
    train_loader = DataLoader(
        CleanRoadDamageDataset(train_samples, augment=True),
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        CleanRoadDamageDataset(val_samples, augment=False),
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )

    # 4. Instantiate Model & Optimizer
    # ResNet18 initialized from ImageNet pretrained weights
    model = RoadDamageClassifier(backbone_name="resnet18", num_classes=4, pretrained=True).to(device)
    param_count = sum(p.numel() for p in model.parameters())

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

    # 5. MLflow Tracking Setup
    os.environ["MLFLOW_DISABLE_AGENT_HINT"] = "1"
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("civicpulse-road-damage-intelligence")

    best_val_macro_f1 = -1.0
    best_weights = None
    best_epoch = 0
    epochs_no_improve = 0
    history = []

    start_train_time = time.time()

    with mlflow.start_run(run_name="clean_resnet18_grouped_v2_training") as run:
        run_id = run.info.run_id
        print(f"\nMLflow Run Initiated: {run_id}")

        # Log parameters
        mlflow.log_params({
            "model_architecture": "resnet18",
            "dataset_split": "rdd2022_india_grouped_v2",
            "train_samples": len(train_samples),
            "val_samples": len(val_samples),
            "epochs": epochs,
            "batch_size": batch_size,
            "initial_lr": lr,
            "weight_decay": 1e-4,
            "early_stopping_patience": patience,
            "param_count": param_count,
            "pos_weight_d00": damped_pos_weights[0],
            "pos_weight_d10": damped_pos_weights[1],
            "pos_weight_d20": damped_pos_weights[2],
            "pos_weight_d40": damped_pos_weights[3],
            "criterion": "BCEWithLogitsLoss_SquareRootDampedPosWeight",
            "checkpoint_target": str(clean_checkpoint_path),
        })

        for epoch in range(1, epochs + 1):
            t_epoch_start = time.time()
            model.train()
            running_train_loss = 0.0
            total_train_items = 0

            for imgs, tgts in train_loader:
                imgs = imgs.to(device, non_blocking=True)
                tgts = tgts.to(device, non_blocking=True)

                optimizer.zero_grad()
                logits = model(imgs)
                loss = criterion(logits, tgts)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

                running_train_loss += loss.item() * imgs.size(0)
                total_train_items += imgs.size(0)

            scheduler.step()
            train_loss = running_train_loss / max(total_train_items, 1)

            # Validation Loop
            model.eval()
            running_val_loss = 0.0
            total_val_items = 0
            val_preds, val_targets = [], []

            with torch.no_grad():
                for imgs, tgts in val_loader:
                    imgs = imgs.to(device, non_blocking=True)
                    tgts = tgts.to(device, non_blocking=True)
                    logits = model(imgs)
                    vloss = criterion(logits, tgts)

                    running_val_loss += vloss.item() * imgs.size(0)
                    total_val_items += imgs.size(0)

                    probs = torch.sigmoid(logits).cpu().numpy()
                    val_preds.append(probs)
                    val_targets.append(tgts.cpu().numpy())

            val_loss = running_val_loss / max(total_val_items, 1)
            val_probs_all = np.vstack(val_preds)
            val_tgts_all = np.vstack(val_targets)
            val_metrics = compute_multilabel_metrics(val_tgts_all, val_probs_all)

            epoch_dur = time.time() - t_epoch_start
            val_macro_f1 = val_metrics["macro_f1"]

            # Log MLflow metrics per epoch
            mlflow.log_metrics({
                "train_loss": train_loss,
                "val_loss": val_loss,
                "val_macro_f1": val_macro_f1,
                "val_micro_f1": val_metrics["micro_f1"],
                "val_weighted_f1": val_metrics["weighted_f1"],
                "val_binary_acc": val_metrics["binary_damage"]["accuracy"],
                "val_binary_f1": val_metrics["binary_damage"]["f1_score"],
                "val_d00_f1": val_metrics["per_class"]["D00"]["f1_score"],
                "val_d10_f1": val_metrics["per_class"]["D10"]["f1_score"],
                "val_d20_f1": val_metrics["per_class"]["D20"]["f1_score"],
                "val_d40_f1": val_metrics["per_class"]["D40"]["f1_score"],
            }, step=epoch)

            epoch_info = {
                "epoch": epoch,
                "train_loss": round(train_loss, 4),
                "val_loss": round(val_loss, 4),
                "val_macro_f1": round(val_macro_f1, 4),
                "duration_sec": round(epoch_dur, 1),
            }
            history.append(epoch_info)

            print(
                f"Epoch [{epoch:02d}/{epochs:02d}] ({epoch_dur:.1f}s) | "
                f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
                f"Val Macro F1: {val_macro_f1:.4f} | D40: {val_metrics['per_class']['D40']['f1_score']:.4f} | "
                f"D20: {val_metrics['per_class']['D20']['f1_score']:.4f} | D00: {val_metrics['per_class']['D00']['f1_score']:.4f} | "
                f"D10: {val_metrics['per_class']['D10']['f1_score']:.4f}"
            )

            # Checkpoint Selection based on Validation Macro F1
            if val_macro_f1 > best_val_macro_f1:
                improvement = val_macro_f1 - best_val_macro_f1
                best_val_macro_f1 = val_macro_f1
                best_epoch = epoch
                best_weights = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                epochs_no_improve = 0
                print(f"  >>> Checkpoint updated at epoch {epoch} (Val Macro F1: {val_macro_f1:.4f}, +{improvement:.4f})")
            else:
                epochs_no_improve += 1
                print(f"  --- No improvement for {epochs_no_improve}/{patience} epochs.")

            if epochs_no_improve >= patience:
                print(f"\n[Early Stopping Triggered] Stopping at epoch {epoch}. Best epoch was {best_epoch} with Val Macro F1 = {best_val_macro_f1:.4f}")
                break

        total_train_sec = time.time() - start_train_time

        # Save the best model artifact
        assert best_weights is not None, "Error: No checkpoint weights captured."
        checkpoint_payload = {
            "model_architecture": "resnet18",
            "backbone": "resnet18",
            "num_classes": 4,
            "classes": CLASS_NAMES,
            "state_dict": best_weights,
            "best_epoch": best_epoch,
            "best_val_macro_f1": best_val_macro_f1,
            "training_dataset": "rdd2022_india_grouped_v2/train",
            "validation_dataset": "rdd2022_india_grouped_v2/val",
            "pos_weight": damped_pos_weights,
            "training_time_sec": round(total_train_sec, 1),
            "mlflow_run_id": run_id,
        }
        torch.save(checkpoint_payload, clean_checkpoint_path)

        ckpt_sha256 = get_file_sha256(clean_checkpoint_path)
        print(f"\nSaved clean checkpoint: {clean_checkpoint_path}")
        print(f"Checksum SHA-256: {ckpt_sha256}")
        print(f"Size: {clean_checkpoint_path.stat().st_size / (1024*1024):.2f} MB")

        # Log artifact and final summary to MLflow
        mlflow.log_artifact(str(clean_checkpoint_path))
        mlflow.log_metric("best_epoch", best_epoch)
        mlflow.log_metric("best_val_macro_f1", best_val_macro_f1)
        mlflow.log_metric("total_train_time_sec", round(total_train_sec, 1))

    return {
        "status": "success",
        "best_epoch": best_epoch,
        "best_val_macro_f1": best_val_macro_f1,
        "checkpoint_path": str(clean_checkpoint_path),
        "sha256": ckpt_sha256,
        "history": history,
        "mlflow_run_id": run_id,
    }


if __name__ == "__main__":
    train_clean_resnet18()

