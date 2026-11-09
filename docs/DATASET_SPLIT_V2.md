# CivicPulse AI — Dataset Partitioning & Road Corridor Integrity Report (v2)

**Document Version**: 2.0.0  
**Status**: Validated & Active  
**Storage Location**: `datasets/splits/rdd2022_india_grouped_v2/`  
**Dataset**: Road Damage Dataset 2022 (RDD2022 India Subset, 7,706 Images)  
**Author**: Senior Computer Vision Research & MLOps Engineering Team  

---

## 1. Executive Summary & Audit Background

During the Phase 2.1 Independent ML Audit, a critical data integrity flaw was identified in the pre-packaged upstream `RDD2022_India` distribution:

1. **Temporal & Spatial Burst Leakage**: **94.17% (1,098 of 1,166)** of test frames had an immediate adjacent burst capture ($\pm 1$ or $\pm 2$ frames) present in the training set.
2. **Root Cause**: The upstream archive applied uniform pseudo-random partitioning across individual frames rather than blocking continuous road stretches. Dashcam videos recorded at 1–5 frames per second along highway corridors were scattered arbitrarily across splits, causing near-identical pavement distress instances to appear simultaneously in both training and test sets.
3. **Implication**: Any held-out model evaluation on the original random split risked measuring visual sequence memorization rather than true geographic generalization to unseen highway corridors.

To resolve this issue permanently without corrupting historical evaluation baselines, a new versioned grouped split was engineered: **`rdd2022_india_grouped_v2`**.

---

## 2. Sequence Structure & Grouping Methodology

### 2.1 Sequence Identification in RDD2022 India
The RDD2022 India dataset consists of 7,706 dashcam frames sequentially indexed from `India_000000.jpg` to `India_009891.jpg`.
- Images were sampled at fixed temporal intervals during mobile survey driving.
- Analysis of index continuity reveals that 6,011 frame pairs have $\Delta = 1$ and 1,321 pairs have $\Delta = 2$.
- No official corridor name, route number, or GPS geofence metadata was provided in the stripped public dataset release.
- Consecutive frame indices represent continuous travel along specific pavement stretches.

### 2.2 Monolithic Road Corridor Partitioning
To prevent spatial and temporal contamination:
1. **Corridor Definition**: The dataset index space is divided into 99 discrete road corridors of 100 consecutive frame indices (`corridor_id = idx // 100`), spanning approximately 1.5 to 2.5 minutes (~1 to 2.5 km) of driving per corridor.
2. **Monolithic Group Assignment**: Every corridor is assigned strictly as a single indivisible block to either `train`, `val`, or `test`. No within-corridor splitting is permitted.
3. **Macro-Block Grouping**: Corridors are scheduled in cyclic macro-blocks (e.g. 14 contiguous corridors to train, 3 to val, 3 to test) to minimize split transitions along the recording timeline.
4. **Boundary Guard Buffering**: At every boundary between two corridors of different split assignments, a **guard band of $G = 5$ frames** is applied on each side of the transition. The transitional frames (190 total images) are segregated into a `buffer/` partition and quarantined from both validation and test sets.
5. **Class Representation Balancing**: Macro-corridor allocations were chosen to ensure adequate representation of all four distress categories across every split, with special preservation of the severe minority class `D10` (transverse cracks).

---

## 3. Quantitative Split Summary

The partitioned dataset strictly preserves all 7,706 original images and Pascal VOC / YOLO annotations without loss or modification:

| Split Partition | Images | Image % | Damaged Images | Clean Images | D00 (Longitudinal) | D10 (Transverse) | D20 (Alligator) | D40 (Pothole) | Total Damage Boxes |
|---|---|---|---|---|---|---|---|---|---|
| **`train`** | **5,309** | 68.89% | 2,127 (40.06%) | 3,182 | 1,072 | 46 | 1,392 | 941 | 3,451 |
| **`val`** | **1,160** | 15.05% | 463 (39.91%) | 697 | 218 | 10 | 315 | 187 | 730 |
| **`test`** | **1,047** | 13.59% | 444 (42.41%) | 603 | 218 | 9 | 278 | 213 | 718 |
| **`buffer` (Quarantine)**| **190** | 2.47% | 74 (38.95%) | 116 | 47 | 3 | 36 | 31 | 117 |
| **Total Dataset** | **7,706** | 100.00% | 3,108 (40.33%) | 4,598 | 1,555 | 68 | 2,021 | 1,372 | 5,016 |

### Class Proportionality Verification:
- **D00 (Longitudinal)**: 68.9% Train / 14.0% Val / 14.0% Test / 3.0% Buffer
- **D10 (Transverse)**: 67.6% Train / 14.7% Val / 13.2% Test / 4.4% Buffer
- **D20 (Alligator)**: 68.9% Train / 15.6% Val / 13.8% Test / 1.8% Buffer
- **D40 (Pothole)**: 68.6% Train / 13.6% Val / 15.5% Test / 2.3% Buffer
- **Overall Damaged Images**: 40.1% in Train vs 39.9% in Val vs 42.4% in Test.

---

## 4. Integrity and Leakage Verification

Independent verification tests were run across all 7,706 images to confirm zero contamination:

| Metric | Original Random Split (v1) | Grouped Corridor Split (v2) | Improvement / Verdict |
|---|---|---|---|
| **Test frames with train neighbor ($\Delta \le 2$)** | **1,098 (94.17%)** | **0 (0.00%)** | **Eliminated (100% burst leakage removed)** |
| **Val frames with train neighbor ($\Delta \le 2$)** | 1,092 (93.17%) | **0 (0.00%)** | **Eliminated (100% burst leakage removed)** |
| **Minimum index distance: Test $\rightarrow$ Train** | **1 frame** | **12 frames** | **$\ge 12$ frames separation guaranteed** |
| **Minimum index distance: Val $\rightarrow$ Train** | **1 frame** | **11 frames** | **$\ge 11$ frames separation guaranteed** |
| **Exact Cryptographic Duplicates (SHA-256)** | 0 | **0** | Verified zero exact duplicates |
| **Perceptual Near-Duplicates (dHash dist $\le 0$)** | 0 | **0** | Zero cross-split perceptual collision |
| **Corridor Monolithic Integrity** | FAILED (Random) | **100% Monolithic** | Corridors never split across partitions |

---

## 5. Storage Structure & Reproducibility

The versioned dataset split is persisted at `datasets/splits/rdd2022_india_grouped_v2/`:

```
datasets/splits/rdd2022_india_grouped_v2/
├── data.yaml                    # YOLO/TorchVision dataset descriptor
├── split_manifest.json          # Machine-readable corridor & split manifest
├── train/
│   ├── images/                  # 5,309 JPEG images
│   └── labels/                  # 5,309 YOLO format text annotations
├── val/
│   ├── images/                  # 1,160 JPEG images
│   └── labels/                  # 1,160 YOLO format text annotations
├── test/
│   ├── images/                  # 1,047 JPEG images
│   └── labels/                  # 1,047 YOLO format text annotations
└── buffer/
    ├── images/                  # 190 boundary transition JPEG images
    └── labels/                  # 190 boundary transition annotations
```

The split generation is 100% deterministic and reproducible via:
```bash
python services/ml/preprocessing/partition_dataset_v2.py
```
The original `datasets/raw/RDD2022_India/` archive remains unmodified and preserved.

