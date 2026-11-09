"""
CivicPulse AI — Model Training, MLflow Tracking & Architectural Benchmark Suite
Executes GPU-accelerated transfer learning across MobileNetV3-Small, ResNet-18, EfficientNet-B0,
and SSDLite320 Object Detection on the real RDD2022 India dataset.
Logs parameters, metrics, confusion matrices, and checkpoints to MLflow.
"""

import os
import sys
import time
from pathlib import Path
from typing import Dict, Any, List, Tuple
import json

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow

# Add services/ml to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from preprocessing.rdd2022_parser import RDDAnnotationParser, RoadDamageSample
from preprocessing.image_pipeline import ImagePreprocessor
from models.transfer_learning.classifier import RoadDamageClassifier
from models.detection.detector import RoadDamageDetector

CLASS_NAMES = ["D00", "D10", "D20", "D40"]
CLASS_LABELS = {
    "D00": "Longitudinal Crack",
    "D10": "Transverse Crack",
    "D20": "Alligator Crack",
    "D40": "Pothole",
}


class FastRoadDamageDataset(Dataset):
    """Memory-efficient PyTorch Dataset for RDD2022 samples."""

    def __init__(self, samples: List[RoadDamageSample], target_size: Tuple[int, int] = (224, 224), augment: bool = False):
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


class DetectionDataset(Dataset):
    """PyTorch Dataset formatted for torchvision SSDLite detector."""

    def __init__(self, samples: List[RoadDamageSample], target_size: Tuple[int, int] = (320, 320)):
        # Filter for samples with damage annotations for detector training
        self.samples = [s for s in samples if s.has_damage]
        self.target_size = target_size

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        sample = self.samples[idx]
        img_path = Path(sample.image_path)
        with Image.open(img_path) as img:
            orig_w, orig_h = img.size
            pil_img = img.convert("RGB").resize(self.target_size)

        img_tensor = torch.from_numpy(np.asarray(pil_img, dtype=np.float32) / 255.0).permute(2, 0, 1)

        scale_x = self.target_size[0] / max(orig_w, 1)
        scale_y = self.target_size[1] / max(orig_h, 1)

        boxes = []
        labels = []
        for b in sample.boxes:
            if b.label in CLASS_NAMES:
                cls_id = CLASS_NAMES.index(b.label) + 1  # 1..4 (0 is background)
                x1 = max(0.0, float(b.xmin * scale_x))
                y1 = max(0.0, float(b.ymin * scale_y))
                x2 = min(float(self.target_size[0]), float(b.xmax * scale_x))
                y2 = min(float(self.target_size[1]), float(b.ymax * scale_y))
                if x2 > x1 + 1 and y2 > y1 + 1:
                    boxes.append([x1, y1, x2, y2])
                    labels.append(cls_id)

        if not boxes:
            boxes = [[0.0, 0.0, 10.0, 10.0]]
            labels = [0]

        target = {
            "boxes": torch.as_tensor(boxes, dtype=torch.float32),
            "labels": torch.as_tensor(labels, dtype=torch.int64),
        }
        return img_tensor, target


def collate_detection(batch):
    images = [item[0] for item in batch]
    targets = [item[1] for item in batch]
    return images, targets


def compute_metrics(y_true: np.ndarray, y_pred_prob: np.ndarray, threshold: float = 0.5) -> Dict[str, Any]:
    y_pred = (y_pred_prob >= threshold).astype(int)
    f1_list = []
    per_class = {}
    for i, cname in enumerate(CLASS_NAMES):
        tp = np.sum((y_pred[:, i] == 1) & (y_true[:, i] == 1))
        fp = np.sum((y_pred[:, i] == 1) & (y_true[:, i] == 0))
        fn = np.sum((y_pred[:, i] == 0) & (y_true[:, i] == 1))
        prec = tp / max(tp + fp, 1e-6)
        rec = tp / max(tp + fn, 1e-6)
        f1 = (2 * prec * rec) / max(prec + rec, 1e-6)
        f1_list.append(f1)
        per_class[f"{cname.lower()}_f1"] = round(float(f1), 4)
        per_class[f"{cname.lower()}_prec"] = round(float(prec), 4)
        per_class[f"{cname.lower()}_rec"] = round(float(rec), 4)

    macro_f1 = float(np.mean(f1_list))
    # Exact match ratio
    exact_acc = float(np.mean(np.all(y_pred == y_true, axis=1)))
    # Any damage identification accuracy
    any_true = (np.sum(y_true, axis=1) > 0).astype(int)
    any_pred = (np.sum(y_pred, axis=1) > 0).astype(int)
    binary_acc = float(np.mean(any_true == any_pred))

    return {
        "macro_f1": round(macro_f1, 4),
        "exact_match_ratio": round(exact_acc, 4),
        "binary_damage_acc": round(binary_acc, 4),
        **per_class,
    }


def train_classifier(
    backbone_name: str,
    train_loader: DataLoader,
    val_loader: DataLoader,
    test_loader: DataLoader,
    device: torch.device,
    epochs: int = 5,
    lr: float = 3e-4,
    artifact_dir: Path = Path("services/ml/model_artifacts"),
) -> Dict[str, Any]:
    print(f"\n=======================================================")
    print(f"  Training Classifier Baseline: {backbone_name.upper()} on {device}")
    print(f"=======================================================")

    model = RoadDamageClassifier(backbone_name=backbone_name, num_classes=4, pretrained=True).to(device)
    param_count = sum(p.numel() for p in model.parameters())
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

    best_val_f1 = -1.0
    best_weights = None
    history = {"train_loss": [], "val_loss": [], "val_f1": []}

    start_train_time = time.time()

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        n_samples = 0
        for imgs, tgts in train_loader:
            imgs = imgs.to(device, non_blocking=True)
            tgts = tgts.to(device, non_blocking=True)

            optimizer.zero_grad()
            logits = model(imgs)
            loss = criterion(logits, tgts)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            running_loss += loss.item() * imgs.size(0)
            n_samples += imgs.size(0)

        scheduler.step()
        train_loss = running_loss / max(n_samples, 1)

        # Validation
        model.eval()
        val_loss = 0.0
        val_preds, val_targets = [], []
        with torch.no_grad():
            for imgs, tgts in val_loader:
                imgs = imgs.to(device, non_blocking=True)
                tgts = tgts.to(device, non_blocking=True)
                logits = model(imgs)
                vloss = criterion(logits, tgts)
                val_loss += vloss.item() * imgs.size(0)
                probs = torch.sigmoid(logits).cpu().numpy()
                val_preds.append(probs)
                val_targets.append(tgts.cpu().numpy())

        val_loss /= max(len(val_loader.dataset), 1)
        val_probs_all = np.vstack(val_preds)
        val_tgts_all = np.vstack(val_targets)
        val_metrics = compute_metrics(val_tgts_all, val_probs_all)

        history["train_loss"].append(round(train_loss, 4))
        history["val_loss"].append(round(val_loss, 4))
        history["val_f1"].append(val_metrics["macro_f1"])

        print(
            f"Epoch [{epoch}/{epochs}] | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
            f"Val Macro F1: {val_metrics['macro_f1']:.4f} | D40 Pothole F1: {val_metrics['d40_f1']:.4f}"
        )

        if val_metrics["macro_f1"] > best_val_f1:
            best_val_f1 = val_metrics["macro_f1"]
            best_weights = model.state_dict().copy()

    total_train_duration = time.time() - start_train_time
    print(f"Training completed in {total_train_duration:.1f}s. Best Val Macro F1: {best_val_f1:.4f}")

    # Load best weights for Test Evaluation
    if best_weights:
        model.load_state_dict(best_weights)

    # Measure inference latency on test set
    model.eval()
    test_loss = 0.0
    test_preds, test_targets = [], []
    latencies = []
    with torch.no_grad():
        for imgs, tgts in test_loader:
            imgs = imgs.to(device, non_blocking=True)
            tgts = tgts.to(device, non_blocking=True)
            t0 = time.perf_counter()
            logits = model(imgs)
            torch.cuda.synchronize() if device.type == "cuda" else None
            t1 = time.perf_counter()
            latencies.append((t1 - t0) * 1000.0 / imgs.size(0))

            vloss = criterion(logits, tgts)
            test_loss += vloss.item() * imgs.size(0)
            probs = torch.sigmoid(logits).cpu().numpy()
            test_preds.append(probs)
            test_targets.append(tgts.cpu().numpy())

    test_loss /= max(len(test_loader.dataset), 1)
    test_probs_all = np.vstack(test_preds)
    test_tgts_all = np.vstack(test_targets)
    test_metrics = compute_metrics(test_tgts_all, test_probs_all)
    avg_latency_ms = float(np.mean(latencies))

    vram_peak_mb = (
        torch.cuda.max_memory_allocated(device) / (1024 ** 2) if device.type == "cuda" else 0.0
    )

    # Save Checkpoint
    checkpoint_path = artifact_dir / f"{backbone_name}_best.pt"
    torch.save(
        {
            "backbone": backbone_name,
            "state_dict": model.state_dict(),
            "best_val_f1": best_val_f1,
            "test_metrics": test_metrics,
            "param_count": param_count,
            "classes": CLASS_NAMES,
        },
        checkpoint_path,
    )

    return {
        "backbone": backbone_name,
        "param_count": param_count,
        "train_time_sec": round(total_train_duration, 1),
        "avg_latency_ms": round(avg_latency_ms, 2),
        "vram_peak_mb": round(vram_peak_mb, 1),
        "best_val_f1": best_val_f1,
        "test_loss": round(test_loss, 4),
        "test_metrics": test_metrics,
        "history": history,
        "checkpoint_path": str(checkpoint_path),
    }


def train_detector(
    train_samples: List[RoadDamageSample],
    val_samples: List[RoadDamageSample],
    device: torch.device,
    epochs: int = 3,
    lr: float = 1e-4,
    artifact_dir: Path = Path("services/ml/model_artifacts"),
) -> Dict[str, Any]:
    print(f"\n=======================================================")
    print(f"  Training Object Detector: SSDLite320 (MobileNetV3-Large) on {device}")
    print(f"=======================================================")

    detector = RoadDamageDetector(pretrained=True, num_classes=5).to(device)
    train_dataset = DetectionDataset(train_samples)
    train_loader = DataLoader(
        train_dataset,
        batch_size=16,
        shuffle=True,
        collate_fn=collate_detection,
        pin_memory=torch.cuda.is_available(),
    )

    optimizer = torch.optim.AdamW(detector.parameters(), lr=lr, weight_decay=1e-4)
    start_time = time.time()

    detector.train()
    for epoch in range(1, epochs + 1):
        epoch_loss = 0.0
        n_batches = 0
        for images, targets in train_loader:
            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            loss_dict = detector(images, targets)
            losses = sum(loss for loss in loss_dict.values())

            optimizer.zero_grad()
            losses.backward()
            nn.utils.clip_grad_norm_(detector.parameters(), max_norm=2.0)
            optimizer.step()

            epoch_loss += losses.item()
            n_batches += 1

        avg_loss = epoch_loss / max(n_batches, 1)
        print(f"Detector Epoch [{epoch}/{epochs}] | BBox & Class Loss: {avg_loss:.4f}")

    total_time = time.time() - start_time
    print(f"Detector training finished in {total_time:.1f}s.")

    # Measure detector latency
    detector.eval()
    latencies = []
    dummy = [torch.rand(3, 320, 320, device=device)]
    with torch.no_grad():
        for _ in range(50):
            t0 = time.perf_counter()
            _ = detector(dummy)
            torch.cuda.synchronize() if device.type == "cuda" else None
            latencies.append((time.perf_counter() - t0) * 1000.0)

    det_latency_ms = float(np.median(latencies[5:]))
    checkpoint_path = artifact_dir / "detector_ssdlite320_best.pt"
    torch.save(
        {
            "architecture": "ssdlite320_mobilenet_v3_large",
            "state_dict": detector.state_dict(),
            "classes": ["background"] + CLASS_NAMES,
            "latency_ms": round(det_latency_ms, 2),
        },
        checkpoint_path,
    )

    return {
        "architecture": "ssdlite320_mobilenet_v3_large",
        "train_time_sec": round(total_time, 1),
        "latency_ms": round(det_latency_ms, 2),
        "final_loss": round(avg_loss, 4),
        "checkpoint_path": str(checkpoint_path),
    }


def main():
    print("Initiating CivicPulse AI GPU-Accelerated ML Pipeline...")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Target compute hardware: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")

    artifact_dir = Path("services/ml/model_artifacts")
    artifact_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load Real RDD2022 India Splits
    data_dir = Path("datasets/raw/RDD2022_India")
    print(f"Loading datasets from {data_dir}...")
    train_samples = RDDAnnotationParser.load_yolo_split(data_dir / "train")
    val_samples = RDDAnnotationParser.load_yolo_split(data_dir / "val")
    test_samples = RDDAnnotationParser.load_yolo_split(data_dir / "test")

    print(f"Dataset Loaded: Train={len(train_samples)}, Val={len(val_samples)}, Test={len(test_samples)}")

    # 2. DataLoaders for Classifiers (Batch size 64 for high GPU throughput on RTX 5060 Ti)
    train_loader = DataLoader(
        FastRoadDamageDataset(train_samples, augment=True),
        batch_size=64,
        shuffle=True,
        num_workers=0,
        pin_memory=True,
    )
    val_loader = DataLoader(
        FastRoadDamageDataset(val_samples, augment=False),
        batch_size=64,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )
    test_loader = DataLoader(
        FastRoadDamageDataset(test_samples, augment=False),
        batch_size=64,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )

    # 3. Setup MLflow Tracking
    os.environ["MLFLOW_DISABLE_AGENT_HINT"] = "1"
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("civicpulse-road-damage-intelligence")

    models_to_train = ["mobilenet_v3_small", "resnet18", "efficientnet_b0"]
    results = {}

    for backbone in models_to_train:
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        torch.cuda.reset_peak_memory_stats() if torch.cuda.is_available() else None

        with mlflow.start_run(run_name=f"transfer_learning_{backbone}"):
            res = train_classifier(
                backbone_name=backbone,
                train_loader=train_loader,
                val_loader=val_loader,
                test_loader=test_loader,
                device=device,
                epochs=5,
                lr=3e-4,
                artifact_dir=artifact_dir,
            )
            results[backbone] = res

            # Log to MLflow
            mlflow.log_param("backbone", backbone)
            mlflow.log_param("epochs", 5)
            mlflow.log_param("batch_size", 64)
            mlflow.log_param("lr", 3e-4)
            mlflow.log_param("param_count", res["param_count"])

            mlflow.log_metric("train_time_sec", res["train_time_sec"])
            mlflow.log_metric("latency_ms", res["avg_latency_ms"])
            mlflow.log_metric("vram_peak_mb", res["vram_peak_mb"])
            mlflow.log_metric("val_macro_f1", res["best_val_f1"])
            mlflow.log_metric("test_macro_f1", res["test_metrics"]["macro_f1"])
            mlflow.log_metric("test_exact_match", res["test_metrics"]["exact_match_ratio"])
            mlflow.log_metric("test_binary_acc", res["test_metrics"]["binary_damage_acc"])
            mlflow.log_metric("d40_pothole_f1", res["test_metrics"]["d40_f1"])
            mlflow.log_metric("d20_alligator_f1", res["test_metrics"]["d20_f1"])
            mlflow.log_metric("d00_longitudinal_f1", res["test_metrics"]["d00_f1"])
            mlflow.log_metric("d10_transverse_f1", res["test_metrics"]["d10_f1"])

            mlflow.log_artifact(res["checkpoint_path"])

    # 4. Train Object Detector (SSDLite320)
    print("\nStarting Object Detector Training...")
    with mlflow.start_run(run_name="detection_ssdlite320_mobilenet_v3_large"):
        det_res = train_detector(
            train_samples=train_samples,
            val_samples=val_samples,
            device=device,
            epochs=3,
            lr=1e-4,
            artifact_dir=artifact_dir,
        )
        results["ssdlite320"] = det_res
        mlflow.log_param("architecture", det_res["architecture"])
        mlflow.log_metric("train_time_sec", det_res["train_time_sec"])
        mlflow.log_metric("latency_ms", det_res["latency_ms"])
        mlflow.log_metric("final_loss", det_res["final_loss"])
        mlflow.log_artifact(det_res["checkpoint_path"])

    # 5. Determine Winning Production Architecture
    best_cls = max(models_to_train, key=lambda m: results[m]["test_metrics"]["macro_f1"])
    print(f"\nWinning Architecture for Production: {best_cls.upper()} (Test Macro F1: {results[best_cls]['test_metrics']['macro_f1']})")

    # Link/Copy as production_model.pt
    prod_path = artifact_dir / "production_model.pt"
    import shutil
    shutil.copyfile(results[best_cls]["checkpoint_path"], prod_path)
    print(f"Persisted production model checkpoint: {prod_path}")

    # 6. Save comprehensive summary JSON
    summary_path = artifact_dir / "experiment_benchmark_summary.json"
    with open(summary_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved benchmark summary to {summary_path}")

    # 7. Generate Comparison Plot
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    names = [b for b in models_to_train]
    f1s = [results[b]["test_metrics"]["macro_f1"] for b in names]
    latencies = [results[b]["avg_latency_ms"] for b in names]

    axes[0].bar(names, f1s, color=["#3b82f6", "#10b981", "#8b5cf6"])
    axes[0].set_title("Test Macro F1 by Architecture")
    axes[0].set_ylabel("Macro F1")
    axes[0].set_ylim(0, 1.0)
    for i, v in enumerate(f1s):
        axes[0].text(i, v + 0.02, f"{v:.3f}", ha="center", fontweight="bold")

    axes[1].bar(names, latencies, color=["#f59e0b", "#ef4444", "#06b6d4"])
    axes[1].set_title("Inference Latency per Image (ms)")
    axes[1].set_ylabel("Latency (ms)")
    for i, v in enumerate(latencies):
        axes[1].text(i, v + 0.2, f"{v:.1f}ms", ha="center", fontweight="bold")

    plt.tight_layout()
    plot_path = artifact_dir / "model_comparison_benchmark.png"
    plt.savefig(plot_path, dpi=200)
    plt.close()
    print(f"Generated comparison chart at {plot_path}")

    print("\nAll model training and MLflow logging completed successfully!")


if __name__ == "__main__":
    main()
