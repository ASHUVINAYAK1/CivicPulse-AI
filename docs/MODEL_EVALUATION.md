# CivicPulse AI — Model Evaluation & Architectural Benchmark Report

**Benchmark Date**: 2026-11-07  
**Evaluation Dataset**: RDD2022 India Test Split (1,166 unseen images, 723 damage bounding boxes, 121 multi-class frames)  
**Tracking URI**: `sqlite:///mlflow.db` (`civicpulse-road-damage-intelligence`)  

---

## 1. Overall Architectural Performance Comparison

| Metric | MobileNetV3-Small | ResNet-18 (Selected Winner) | EfficientNet-B0 | SSDLite320 (Detector) |
|---|---|---|---|---|
| **Parameters** | 1,521,956 | 11,178,564 | 4,012,672 | ~3,400,000 |
| **Test Macro F1** | 0.3545 | **0.4529** | 0.4394 | N/A (Localization) |
| **Exact Match Ratio** | 66.38% | **71.70%** | 70.50% | N/A |
| **Binary Road Damage Accuracy** | 78.39% | **81.99%** | 81.65% | N/A |
| **Test Loss (BCE)** | 0.2368 | **0.2243** | **0.2238** | 5.4974 (SSD loss) |
| **Inference Latency (GPU)** | **0.48 ms** | **0.66 ms** | 0.94 ms | 10.01 ms |
| **Inference Latency (FPS equivalent)** | ~2,080 FPS | ~1,515 FPS | ~1,063 FPS | ~100 FPS |
| **Peak VRAM During Training** | 1,056.7 MB | 1,586.0 MB | 5,491.9 MB | 2,840.0 MB |
| **Artifact Checkpoint** | `mobilenet_v3_small_best.pt` | `resnet18_best.pt` | `efficientnet_b0_best.pt` | `detector_ssdlite320_best.pt` |

---

## 2. Per-Class Performance Breakdown on Unseen Road Images

Distress classification performance across the standardized RDD2022 distress taxonomy:

### ResNet-18 (Production Model)
| Damage Code | Pavement Distress | Precision | Recall | F1 Score | Support (Test Set) |
|---|---|---|---|---|---|
| **D00** | Longitudinal Crack | 61.16% | 45.68% | **0.5230** | 229 |
| **D10** | Transverse Crack | 0.00% | 0.00% | **0.0000** | 7 (Severe minority) |
| **D20** | Alligator Crack | 81.36% | 76.49% | **0.7885** | 286 |
| **D40** | Pothole (Critical) | 56.30% | 44.97% | **0.5000** | 201 |
| **Macro Average** | — | **49.71%** | **41.79%** | **0.4529** | 723 boxes |

### EfficientNet-B0 (Runner-up)
- **D00 Longitudinal**: F1 = `0.5036` (Prec: 60.34%, Rec: 43.21%)
- **D10 Transverse**: F1 = `0.0000` (Prec: 0.00%, Rec: 0.00%)
- **D20 Alligator**: F1 = `0.8032` (Prec: 80.97%, Rec: 79.68%)
- **D40 Pothole**: F1 = `0.4509` (Prec: 49.21%, Rec: 41.61%)

### MobileNetV3-Small (Edge Baseline)
- **D00 Longitudinal**: F1 = `0.3908` (Prec: 51.52%, Rec: 31.48%)
- **D10 Transverse**: F1 = `0.0000` (Prec: 0.00%, Rec: 0.00%)
- **D20 Alligator**: F1 = `0.7390` (Prec: 70.80%, Rec: 77.29%)
- **D40 Pothole**: F1 = `0.2883` (Prec: 43.84%, Rec: 21.48%)

---

## 3. Analysis & Key Insights

1. **Alligator Cracking (`D20`) Performance**: All architectures scored highest on alligator cracks (up to 80.3% F1). The distinctive interconnected network texture allows CNN feature extractors to detect pavement fatigue reliably across varying lighting conditions.
2. **Critical Pothole (`D40`) Detection**: ResNet-18 achieved the highest pothole F1 score (**0.5000** vs 0.4509 for EfficientNet and 0.2883 for MobileNetV3). Because potholes pose the greatest road safety risk to motorists and cyclists, ResNet-18 is selected for production deployment.
3. **Severe Imbalance on Transverse Cracks (`D10`)**: Transverse cracking occurred in only 50 training images and 7 test images out of 7,706 total frames. In tropical and sub-tropical road networks (such as India), asphalt thermal contraction transverse fractures are physically rare compared to structural fatigue cracking. In Phase 3, targeted cross-dataset sampling from the RDD2022 Japan or Czech subsets will be added to reinforce D10.
4. **Latency vs Throughput**: ResNet-18 provides sub-millisecond inference (**0.66 ms** per frame), consuming only 1.5 GB VRAM on the RTX 5060 Ti. Combining ResNet-18 classification with SSDLite320 detection yields a total processing time of **~10.7 ms per image**, enabling CivicPulse AI to comfortably process over 90 road inspection images per second in real time.

