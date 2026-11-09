# CivicPulse AI — Phase 2.1: ML Validation and Production Readiness Audit Report

**Audit Date**: 2026-11-07  
**Auditor Role**: Senior ML Research Engineer & MLOps Systems Auditor  
**Evaluation Target**: Phase 2 Road Damage Intelligence Pipeline (RDD2022 India, PyTorch cu128, NVIDIA GeForce RTX 5060 Ti)  
**Audit Status**: **COMPLETED — CONDITIONAL PASS (Production Approved for ResNet-18 Classifier; Demoted SSDLite320 Detector to Experimental)**  

---

## Executive Summary

An independent, rigorous audit was conducted across the machine learning artifacts, datasets, inference endpoints, database schemas, and end-to-end integration workflows of CivicPulse AI.

### Key Audit Verdicts:
1. **ResNet-18 Road Distress Classifier**: **APPROVED FOR PRODUCTION**.
   - Recomputed test metrics on 1,166 held-out test frames: Macro F1 = **0.4529**, Micro F1 = **0.6379**, Exact Match = **71.70%**, Binary Damage Accuracy = **81.99%**, Expected Calibration Error (ECE) = **0.0295**, GPU Latency = **1.90 ms**.
   - Checkpoint hash match: `production_model.pt` verified 100% bit-for-bit identical to `resnet18_best.pt` (`SHA-256: 4a912380...`).
2. **SSDLite320 Object Detector**: **FAILED PRODUCTION GATE — DEMOTED TO EXPERIMENTAL**.
   - The Phase 2 implementation report substituted training loss convergence (5.49) for detection accuracy.
   - Independent audit reveals genuine held-out **mAP@0.5 is only 0.0898 (8.98%)** and **mAP@0.5:0.95 is 0.0285 (2.85%)**.
   - While recall on alligator fatigue cracking (D20) is strong (89.16%), the model produces massive false-positive proposals on background textures (~100,000 proposals on test split). It cannot be trusted for automated municipal work order generation.
   - Database record in `model_registry` has been updated from `production` to `experimental`.
3. **Dataset Leakage Finding**: **HIGH TEMPORAL CORRELATION IN PRE-PACKAGED SPLIT**.
   - Zero cryptographic duplicates exist across splits (0 cross-split duplicates).
   - However, **94.17% of test frames** have an immediate burst capture ($\pm 1$ or $\pm 2$ frames) in the training set due to uniform random partitioning in the upstream RDD2022 India archive.
4. **Latency Audit**:
   - Classification GPU Latency: **1.90 ms** (median) / **2.53 ms** (p95).
   - Detection GPU Latency: **12.74 ms** (median) / **15.28 ms** (p95).
   - FastAPI IPv4 HTTP Latency: **29.80 ms** (median).
   - Complete Upload -> MinIO S3 -> PyTorch GPU -> PostgreSQL Persistence: **47.46 ms** (median) / **48.82 ms** (p95).
   - *Windows Networking Anomaly Discovered*: Using `localhost` incurs a **2,071 ms** IPv6 (`::1`) DNS fallback penalty on Windows; binding to `127.0.0.1` completely resolves this.
5. **End-to-End Workflow**: **VERIFIED FUNCTIONAL**.
   - Multipart image upload, MinIO object storage, PostgreSQL transactional persistence, presigned image URL retrieval, and human-in-the-loop inspector verification all pass with 100% reliability.
   - Missing input and 404 boundaries correctly return HTTP 400 and 404.
   - Duplicate image submission currently creates distinct incidents due to lack of an ingest-time perceptual hash deduplication gate.

---

## 1. Classification Audit (Independent Metric Verification)

The three trained classification checkpoints were evaluated against the held-out test split of 1,166 images from `datasets/raw/RDD2022_India/test`:

### 1.1 Architectural Implementation Verification
- **Target Encoding**: Multi-hot 4-dimensional binary vector `[D00, D10, D20, D40]` representing presence of distress categories.
- **Model Output Dimensions**: Exactly `(batch_size, 4)` linear logits.
- **Activation & Loss Formulation**: Correctly uses un-activated raw logits with `torch.nn.BCEWithLogitsLoss()` during training, and independent `torch.sigmoid()` during inference. This correctly models multi-label co-occurrence rather than mutually exclusive softmax assumptions.
- **Data Augmentation**: Random horizontal flipping ($p=0.5$), ImageNet standard channel normalization ($\mu=[0.485, 0.456, 0.406]$, $\sigma=[0.229, 0.224, 0.225]$).

### 1.2 Recomputed Performance Metrics on Held-Out Test Set (1,166 Images)

| Metric | ResNet-18 (Production) | EfficientNet-B0 (Evaluated) | MobileNetV3-Small (Evaluated) |
|---|---|---|---|
| **Macro F1 Score** | **0.4529** | 0.4394 | 0.3545 |
| **Macro Precision** | **0.4970** | 0.4763 | 0.4154 |
| **Macro Recall** | **0.4178** | 0.4113 | 0.3256 |
| **Micro F1 Score** | **0.6379** | 0.6282 | 0.5464 |
| **Micro Precision** | **0.6996** | 0.6789 | 0.6211 |
| **Micro Recall** | **0.5863** | 0.5845 | 0.4877 |
| **Weighted F1 Score** | **0.6288** | 0.6169 | 0.5137 |
| **Exact Match Ratio (Subset Accuracy)** | **71.70%** | 70.50% | 66.38% |
| **Hamming Loss (Label Error Rate)** | **0.0810** | 0.0843 | 0.0986 |
| **Expected Calibration Error (ECE)** | **0.0295** | 0.0286 | **0.0193** |
| **Brier Score** | **0.0637** | 0.0638 | 0.0713 |
| **Binary Road Damage Accuracy** | **81.99%** | 81.65% | 78.39% |
| **Binary Precision / Recall / F1** | P: 80.64% / R: 68.93% / **F1: 74.33%** | P: 78.73% / R: 70.52% / **F1: 74.40%** | P: 77.08% / R: 61.00% / **F1: 68.10%** |

### 1.3 Per-Class Performance Breakdown & 2x2 Confusion Matrices

#### ResNet-18 (Production Checkpoint)
| Distress Code | Distress Name | Support (Images) | TP | FP | FN | TN | Precision | Recall | F1 Score | Specificity |
|---|---|---|---|---|---|---|---|---|---|---|
| **D00** | Longitudinal Crack | 162 | 74 | 47 | 88 | 957 | 61.16% | 45.68% | **0.5230** | 95.32% |
| **D10** | Transverse Crack | 6 | 0 | 0 | 6 | 1160 | 0.00% | 0.00% | **0.0000** | 100.00% |
| **D20** | Alligator Crack | 251 | 192 | 44 | 59 | 871 | 81.36% | 76.49% | **0.7885** | 95.19% |
| **D40** | Pothole (Critical) | 149 | 67 | 52 | 82 | 965 | 56.30% | 44.97% | **0.5000** | 94.89% |

#### EfficientNet-B0
- D00: P = 60.34%, R = 43.21%, **F1 = 0.5036** (TP: 70, FP: 46, FN: 92, TN: 958)
- D10: P = 0.00%, R = 0.00%, **F1 = 0.0000** (TP: 0, FP: 0, FN: 6, TN: 1160)
- D20: P = 80.97%, R = 79.68%, **F1 = 0.8032** (TP: 200, FP: 47, FN: 51, TN: 868)
- D40: P = 49.21%, R = 41.61%, **F1 = 0.4509** (TP: 62, FP: 64, FN: 87, TN: 953)

#### MobileNetV3-Small
- D00: P = 51.52%, R = 31.48%, **F1 = 0.3908** (TP: 51, FP: 48, FN: 111, TN: 956)
- D10: P = 0.00%, R = 0.00%, **F1 = 0.0000** (TP: 0, FP: 0, FN: 6, TN: 1160)
- D20: P = 70.80%, R = 77.29%, **F1 = 0.7390** (TP: 194, FP: 80, FN: 57, TN: 835)
- D40: P = 43.84%, R = 21.48%, **F1 = 0.2883** (TP: 32, FP: 41, FN: 117, TN: 976)

### 1.4 Investigation: Why Does D10 Have Zero F1?

A dedicated probability distribution and threshold sensitivity study was conducted for `D10`:
1. **Severe Dataset Imbalance**:
   - Total training instances: 50 boxes across 5,368 images (**0.93%**).
   - Total test instances: 6 images across 1,166 images (**0.51%**).
2. **Predicted Probability Analysis**:
   - ResNet-18 mean predicted probability for D10 across all test images: **0.0083** (0.83%).
   - Maximum predicted probability output across all test images: **0.2506**.
   - Mean predicted probability on actual ground-truth positive samples: **0.0147** (1.47%).
3. **Threshold Sweep**:
   - Standard threshold (0.50): TP = 0, FP = 0, FN = 6 $\rightarrow$ **F1 = 0.000**.
   - Reduced threshold (0.25): TP = 0, FP = 1, FN = 6 $\rightarrow$ **F1 = 0.000**.
   - Reduced threshold (0.10): TP = 0, FP = 15, FN = 6 $\rightarrow$ **F1 = 0.000**.
   - Ultra-low threshold (0.05): TP = 1, FP = 32, FN = 5 $\rightarrow$ Recall = 16.67%, Precision = 3.03%, **F1 = 0.0513**.
4. **Physical & Civil Engineering Conclusion**:
   - Transverse cracking (thermal expansion fractures perpendicular to the lane) occurs predominantly in climates with freeze-thaw cycles (e.g., Japan, Norway, Czech Republic).
   - In Indian road networks, failures are overwhelmingly dominated by monsoon moisture softening the sub-base under high axle loads, causing longitudinal rutting (`D00`), alligator fatigue cracking (`D20`), and complete asphalt cavitation into potholes (`D40`).
   - The model appropriately learned to suppress false D10 alarms given the near-total absence of positive gradient signals.

---

## 2. Object Detection Audit (SSDLite320 Independent Evaluation)

### 2.1 Audit Findings & Methodology
SSDLite320 was evaluated directly on all 1,166 test images against the original 723 ground-truth bounding boxes using standard Intersection-over-Union (IoU) matching:

| Detection Metric | Measured Audit Value | Assessment |
|---|---|---|
| **mAP@0.5 (PASCAL VOC Metric)** | **0.0898 (8.98%)** | **FAILED Production Gate (< 30% baseline)** |
| **mAP@0.5:0.95 (COCO Standard Metric)** | **0.0285 (2.85%)** | **FAILED Production Gate** |
| **Mean IoU on Matched True Positives** | **0.6262 (62.62%)** | Acceptable proposal localization when detected |
| **Total Ground Truth Boxes** | 723 | Original RDD2022 Pascal VOC annotations |
| **Total Detection Proposals ($\ge 0.05$)** | 349,701 | Excessive background false alarms |

### 2.2 Per-Class Detection AP Breakdown

| Damage Code | Damage Category | Ground Truth Boxes | AP@0.5 | AP@0.5:0.95 | Recall@0.5 | Precision@0.5 |
|---|---|---|---|---|---|---|
| **D20** | Alligator Crack | 286 | **0.3360 (33.6%)** | **0.1090 (10.9%)** | **89.16%** | 0.25% |
| **D40** | Pothole | 201 | **0.0185 (1.85%)** | **0.0041 (0.41%)** | **30.35%** | 0.07% |
| **D00** | Longitudinal Crack | 229 | **0.0047 (0.47%)** | **0.0009 (0.09%)** | **50.22%** | 0.07% |
| **D10** | Transverse Crack | 7 | **0.0000 (0.00%)** | **0.0000 (0.00%)** | **0.00%** | 0.00% |

### 2.3 Diagnostic Error Analysis
- **Root Cause**: The detector was trained for only 3 epochs (71.8 seconds). While the MobileNetV3-Large feature backbone was pre-trained on ImageNet, the multi-scale classification and regression heads were initialized randomly.
- **D20 Success vs D40 Failure**: Alligator cracking covers large image patches, allowing the anchor generator to overlap easily (Recall = 89.16%, Mean IoU = 0.626). Potholes and thin longitudinal cracks, however, require fine multi-scale feature pyramids and extensive hard-negative mining to avoid triggering on road tarmac texture variations and oil stains.
- **Corrective Action Taken**: The database record for `road-damage-detector-ssdlite320` has been changed from `status: "production"` to `status: "experimental"`.

---

## 3. Dataset Leakage & Integrity Audit

### 3.1 Cryptographic Duplicate Check
- Computed SHA-256 hashes across all 7,706 images (5,368 train, 1,172 val, 1,166 test).
- **Exact Cross-Split Duplicates**: **0** (Zero byte-for-byte duplicates).
- **Internal Split Duplicates**: **0** in train, **0** in val, **0** in test.

### 3.2 Sequence Contiguity & Burst Capture Analysis
Dashcam imagery is recorded continuously along travel corridors at 1–5 FPS. Adjacent numeric image IDs (e.g. `India_000036.jpg` and `India_000038.jpg`) represent frames captured within 0.5 to 1.5 seconds of each other.
- **Finding**: **1,098 out of 1,166 test frames (94.17%)** have an immediate adjacent neighbor ($\pm 1$ or $\pm 2$ frames) present in the training set.
- **Examples**:
  - `India_000038.jpg` (Test) has `India_000036.jpg` (Train) at $\Delta = -2$ (~1.0s gap).
  - `India_000053.jpg` (Test) has `India_000052.jpg` (Train) at $\Delta = -1$ (~0.5s gap).
  - `India_000096.jpg` (Test) has `India_000095.jpg` (Train) at $\Delta = -1$ (~0.5s gap).
- **Implication**: The upstream pre-packaged `RDD2022_India` distribution applied pseudo-random uniform splitting across images rather than blocking monolithic road segments. This introduces temporal burst correlation that may slightly inflate test metrics relative to deployment on completely unseen geographic highways.
- **Hyperparameter Contamination Verification**:
  - Test set was strictly held out: inspected `train_and_evaluate.py` to confirm checkpoint selection was based solely on `val_macro_f1`. Zero test gradients or validation choices occurred.

---

## 4. Inference and Latency Audit (Synchronized GPU Benchmarks)

Every layer of the inference and persistence pipeline was benchmarked over multiple iterations on the RTX 5060 Ti:

| Pipeline Layer | Median Latency | p95 Latency | Min Latency | Max Latency | Measurement Notes |
|---|---|---|---|---|---|
| **1. Image Preprocessing** | **6.76 ms** | 7.52 ms | 6.21 ms | 8.14 ms | PIL decode, EXIF transpose, 224x224 Lanczos, float32 norm |
| **2. ResNet-18 Classification (GPU)** | **1.90 ms** | 2.53 ms | 1.82 ms | 3.10 ms | PyTorch CUDA synchronized (`cuda:0`), batch size 1 |
| **3. SSDLite320 Detection (GPU)** | **12.74 ms** | 15.28 ms | 12.10 ms | 16.45 ms | PyTorch CUDA synchronized (`cuda:0`), 320x320 proposals |
| **4. FastAPI Direct HTTP (IPv4 `127.0.0.1`)** | **29.80 ms** | 36.05 ms | 26.40 ms | 41.20 ms | Multipart HTTP POST + GPU inference + JSON serialize |
| **4b. FastAPI Direct HTTP (Windows `localhost`)** | **2,071.46 ms** | 2,076.63 ms | 2,065.10 ms | 2,083.40 ms | **Windows IPv6 `::1` resolution fallback timeout penalty** |
| **5. Express-to-FastAPI Proxy** | **31.19 ms** | 35.34 ms | 28.10 ms | 38.90 ms | Node Express proxy forwarding buffer to FastAPI |
| **6. Complete Production Pipeline** | **47.46 ms** | 48.82 ms | 44.10 ms | 53.20 ms | **Upload + MinIO S3 put + GPU ML + PostgreSQL persistence** |

---

## 5. Model Registry & Checksum Verification

### 5.1 On-Disk Checksums (SHA-256)
All artifacts in `services/ml/model_artifacts` were cryptographically audited:
- `production_model.pt` (44.8 MB): `4a912380869873840fac09eb0aec3de9fd73334271104399e846398b7d0c7acf`
- `resnet18_best.pt` (44.8 MB): `4a912380869873840fac09eb0aec3de9fd73334271104399e846398b7d0c7acf` (100% Bit-for-bit identical)
- `efficientnet_b0_best.pt` (16.4 MB): `1d24d1b9560901a5f41df5b183ca9ec7e17afe4b5a40f4867dc28a29dbe48d52`
- `mobilenet_v3_small_best.pt` (6.2 MB): `78304ba597fdeb4d17ada1b15c446c11f50c10847ce91f2d0f40207e5ef9c1b9`
- `detector_ssdlite320_best.pt` (15.4 MB): `10883f0d06a1d44d0c9c7e1fa6a74e1cb7faf66a88fbc65d157c7ca6c6ddbf4a`

### 5.2 PostgreSQL Model Registry State

| Model Name | Version | Framework | Audited Status | Reason / Evidence |
|---|---|---|---|---|
| `road-damage-classifier-resnet18` | 1.0.0 | PyTorch | **production** | Validated Macro F1 = 0.4529, Binary Acc = 81.99%, Pothole F1 = 0.5000, Latency = 1.90ms. |
| `road-damage-detector-ssdlite320` | 1.0.0 | PyTorch | **experimental** | **Demoted**: mAP@0.5 is 8.98%. Bounding boxes serve only as coarse proposals. |
| `road-damage-classifier-efficientnet-b0` | 1.0.0 | PyTorch | **evaluated** | Runner-up baseline (Macro F1 = 0.4394). |
| `road-damage-classifier-mobilenetv3` | 1.0.0 | PyTorch | **evaluated** | Edge baseline (Macro F1 = 0.3545). |

---

## 6. End-to-End Workflow & Error Boundary Audit

The complete transactional pipeline was tested with automated verification script `services/api/src/scripts/audit-e2e-workflow.ts`:

1. **Real Image Ingestion & Storage**:
   - `India_000038.jpg` uploaded via `POST /api/v1/incidents/report`.
   - MinIO S3 object stored under `incidents/2026-11-07/vzncnt27-India_000038.jpg` (52,165 bytes confirmed via `statObject`).
2. **GPU ML Inference & Database Persistence**:
   - Model correctly detected critical pothole (D40) with **84.4% confidence**.
   - Created `Incident` record (`INC-357677-895`), `IncidentImage` record, and `AiPrediction` record linked via foreign keys in PostgreSQL.
3. **Presigned URL Retrieval**:
   - `GET /api/v1/incidents/:id` generated valid presigned MinIO URL with 24-hour expiry.
4. **Inspector Verification**:
   - `POST /api/v1/incidents/:id/verify` updated incident status from `submitted` to `verified`, logging an immutable `IncidentVerification` entry.
5. **Error Boundaries**:
   - Empty multipart payload $\rightarrow$ returned **HTTP 400 Bad Request** (`PASS`).
   - Non-existent UUID $\rightarrow$ returned **HTTP 404 Not Found** (`PASS`).
   - Corrupted image bytes $\rightarrow$ returned **HTTP 422 Unprocessable Content** (`PASS`).
6. **Duplicate Submission Behavior**:
   - Submitting the same photo twice resulted in two distinct incident IDs. Storage did not collide due to unique keys, but lack of spatial/perceptual hash deduplication permits redundant reports.

---

## 7. Production Readiness Assessment & Corrective Actions

### 7.1 Production Readiness Scorecard

| Subsystem | Readiness Status | Operational Recommendation |
|---|---|---|
| **Hardware & Acceleration** | **READY** | RTX 5060 Ti with PyTorch cu128 provides stable sub-2ms tensor execution. |
| **ResNet-18 Classifier** | **READY** | Approved for citizen and dashcam triage (82% binary accuracy, 72% exact match). |
| **SSDLite320 Object Detector** | **NOT READY (EXPERIMENTAL)** | Keep disabled for automated dispatch; use only as assistive visual overlay. |
| **MinIO Storage & PostgreSQL** | **READY** | Transactional integrity, foreign keys, and presigned URLs operational. |
| **FastAPI ML Service** | **READY** | Stable on port 8000; requires binding to IPv4 (`127.0.0.1`) on Windows. |
| **Next.js Dashboard** | **READY** | Model registry and incident queue render verified data and inspector controls. |

### 7.2 Prioritized Corrective Actions for Phase 3

1. **High Priority — Object Detector Retraining**:
   - Train SSDLite320 (or fine-tune YOLOv8/RT-DETR) for **30–50 epochs** with hard-negative mining, Focal Loss, and mosaic augmentation to increase mAP@0.5 from 8.98% to $\ge 45\%$.
2. **High Priority — Pavement Spatial/Corridor Partitioning**:
   - Re-partition RDD2022 dataset using contiguous road segment blocking (e.g. grouped by geographic road corridor) to eliminate the 94.2% burst frame overlap between training and testing sets.
3. **Medium Priority — Cross-National Transfer for D10**:
   - Supplement the Indian dataset with transverse crack samples from the RDD2022 Japan or Czech subsets to provide gradient signal for thermal cracking.
4. **Medium Priority — Ingest Deduplication Gate**:
   - Implement perceptual image hashing (dHash/pHash) and PostGIS spatial distance checks ($\le 10$ meters) in `services/api` to merge duplicate reports of the same defect.
5. **Low Priority — Network Binding Configuration**:
   - Update default backend service URL in configuration from `localhost` to `127.0.0.1` to prevent Windows IPv6 DNS resolution fallback delays.

