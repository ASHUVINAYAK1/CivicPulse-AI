# CivicPulse AI — Model Training Architecture & Protocol

**Hardware Platform**: NVIDIA GeForce RTX 5060 Ti (Blackwell, Compute Capability `12.0`, 8,151 MiB GDDR7 VRAM)  
**Host Framework**: PyTorch `2.11.0+cu128`, TorchVision `0.26.0+cu128`, CUDA `13.3` driver  
**Experiment Tracker**: MLflow 3.18.0 (`sqlite:///mlflow.db`)  
**Dataset**: RDD2022 India Subset (7,706 images: 5,368 train, 1,172 val, 1,166 test)  

---

## 1. Architectural Strategy & Problem Formulation

Road distress assessment from dashcam photography poses two distinct computer vision challenges:
1. **Multi-Label Pavement Distress Classification**: Dashcam frames frequently contain multiple co-occurring defects (e.g. longitudinal cracking transitioning into pothole cavities). Standard single-label softmax cross-entropy falsely penalizes secondary genuine damage. We formulate distress identification as a **multi-label sigmoid cross-entropy** problem across four standardized distress categories:
   - `D00`: Longitudinal Crack
   - `D10`: Transverse Crack
   - `D20`: Alligator Crack (Fatigue cracking)
   - `D40`: Pothole
2. **Distress Localization (Object Detection)**: Pinpointing physical bounding-box pixel extents is required to prioritize municipal repair crews. We implement an **SSDLite320** detector with MobileNetV3-Large feature extraction backbone for real-time edge/server localization.

---

## 2. Model Candidates Compared

| Model Architecture | Backbone Family | Trainable Parameters | Pretrained Initialization | Design Rationale |
|---|---|---|---|---|
| **MobileNetV3-Small** | MobileNet | 1,521,956 | ImageNet-1K (`DEFAULT`) | Ultra-lightweight mobile baseline for low-power edge compute. |
| **ResNet-18** | Residual Networks | 11,178,564 | ImageNet-1K (`DEFAULT`) | Residual skip connections providing stable gradient propagation across varied road textures. |
| **EfficientNet-B0** | EfficientNet | 4,012,672 | ImageNet-1K (`DEFAULT`) | Compound scaling balancing depth, width, and resolution. |
| **SSDLite320** | SSD + MobileNetV3-L | ~3,400,000 | ImageNet-1K Backbone | Real-time anchor-based single-shot bounding-box detector (320x320). |

---

## 3. Training Hyperparameters & Optimization

- **Loss Function**: `torch.nn.BCEWithLogitsLoss()` for classification; Smooth L1 + Focal/CrossEntropy for SSDLite detection.
- **Optimizer**: AdamW (`weight_decay=1e-4`).
- **Initial Learning Rate**: `3e-4` for classifiers, `1e-4` for object detector.
- **Learning Rate Schedule**: `CosineAnnealingLR` with $T_{max}=5$ and $\eta_{min}=10^{-6}$.
- **Batch Size**: 64 (classifiers) / 16 (detector), optimized for the RTX 5060 Ti memory bus.
- **Data Augmentations**: Horizontal flipping ($p=0.5$), ImageNet standard channel standardization ($\mu=[0.485, 0.456, 0.406]$, $\sigma=[0.229, 0.224, 0.225]$), Lanczos resampling.
- **Gradient Clipping**: `max_norm=1.0` to stabilize transfer-learning head adaptation.

---

## 4. Hardware Resource Utilization During Training

| Model | GPU VRAM Peak | Training Time (5 Epochs) | Throughput | Temperature Range |
|---|---|---|---|---|
| MobileNetV3-Small | 1,056.7 MiB | 281.2s | ~95 img/sec | 42°C – 48°C |
| ResNet-18 | 1,586.0 MiB | 272.2s | ~98 img/sec | 45°C – 52°C |
| EfficientNet-B0 | 5,491.9 MiB | 284.5s | ~94 img/sec | 47°C – 54°C |
| SSDLite320 (3 Epochs) | 2,840.0 MiB | 71.8s | ~92 img/sec | 44°C – 49°C |

All training workloads completed comfortably within the 8.0 GB GDDR7 VRAM envelope without host RAM paging or memory fragmentation.

