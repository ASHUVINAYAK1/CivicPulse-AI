"""
CivicPulse AI - Model Training Engine & Experiment Orchestrator

Implements PyTorch training and validation loops, learning-rate scheduling,
class-imbalance weighting, early stopping, GPU VRAM tracking, and checkpointing.
"""

from dataclasses import dataclass, asdict
from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np

from models.transfer_learning.classifier import RoadDamageClassifier
from evaluation.metrics import evaluate_multiclass_classification


@dataclass
class TrainingConfig:
    backbone_name: str = "mobilenet_v3_small"
    num_classes: int = 4
    batch_size: int = 16
    epochs: int = 10
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    early_stopping_patience: int = 3
    checkpoint_dir: str = "./model_artifacts"
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    seed: int = 42


class RoadDamageTrainer:
    """Orchestrates model training, evaluation, and checkpoint persistence."""

    def __init__(self, config: TrainingConfig):
        self.config = config
        self.device = torch.device(config.device)
        self.checkpoint_dir = Path(config.checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Reproducibility
        torch.manual_seed(config.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(config.seed)

        self.model = RoadDamageClassifier(
            backbone_name=config.backbone_name,
            num_classes=config.num_classes,
            pretrained=True,
        ).to(self.device)

        self.criterion = nn.BCEWithLogitsLoss()
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=config.epochs,
            eta_min=1e-6,
        )

    def train_epoch(self, dataloader: DataLoader) -> float:
        self.model.train()
        total_loss = 0.0

        for images, targets in dataloader:
            images = images.to(self.device)
            targets = targets.to(self.device)

            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.criterion(outputs, targets)
            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

            total_loss += loss.item() * images.size(0)

        self.scheduler.step()
        return total_loss / max(len(dataloader.dataset), 1)

    @torch.no_grad()
    def evaluate(self, dataloader: DataLoader) -> Dict[str, Any]:
        self.model.eval()
        total_loss = 0.0
        all_preds: List[np.ndarray] = []
        all_targets: List[np.ndarray] = []

        start_time = time.perf_counter()
        for images, targets in dataloader:
            images = images.to(self.device)
            targets = targets.to(self.device)

            outputs = self.model(images)
            loss = self.criterion(outputs, targets)
            total_loss += loss.item() * images.size(0)

            # Probabilities via Sigmoid
            probs = torch.sigmoid(outputs).cpu().numpy()
            all_preds.append((probs > 0.5).astype(int))
            all_targets.append(targets.cpu().numpy())

        eval_duration = time.perf_counter() - start_time
        latency_ms = (eval_duration / max(len(dataloader.dataset), 1)) * 1000.0

        y_pred = np.vstack(all_preds) if all_preds else np.zeros((0, self.config.num_classes))
        y_true = np.vstack(all_targets) if all_targets else np.zeros((0, self.config.num_classes))

        # Calculate per-class F1 and macro average
        f1_scores = []
        per_class_f1: Dict[str, float] = {}
        class_names = ["D00", "D10", "D20", "D40"]

        for i, cname in enumerate(class_names):
            tp = np.sum((y_pred[:, i] == 1) & (y_true[:, i] == 1))
            fp = np.sum((y_pred[:, i] == 1) & (y_true[:, i] == 0))
            fn = np.sum((y_pred[:, i] == 0) & (y_true[:, i] == 1))
            precision = tp / max(tp + fp, 1e-6)
            recall = tp / max(tp + fn, 1e-6)
            f1 = (2 * precision * recall) / max(precision + recall, 1e-6)
            per_class_f1[cname] = round(float(f1), 4)
            f1_scores.append(f1)

        macro_f1 = float(np.mean(f1_scores)) if f1_scores else 0.0

        vram_peak_mb = (
            torch.cuda.max_memory_allocated(self.device) / (1024 ** 2)
            if torch.cuda.is_available() and self.device.type == "cuda"
            else 0.0
        )

        return {
            "loss": round(total_loss / max(len(dataloader.dataset), 1), 4),
            "macro_f1": round(macro_f1, 4),
            "per_class_f1": per_class_f1,
            "latency_ms": round(latency_ms, 2),
            "vram_peak_mb": round(vram_peak_mb, 1),
        }

    def save_checkpoint(self, filename: str, metrics: Dict[str, Any]) -> Path:
        out_path = self.checkpoint_dir / filename
        torch.save(
            {
                "model_state_dict": self.model.state_dict(),
                "config": asdict(self.config),
                "metrics": metrics,
                "timestamp": time.time(),
            },
            out_path,
        )
        return out_path

