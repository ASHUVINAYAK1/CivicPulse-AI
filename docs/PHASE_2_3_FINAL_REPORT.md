# CivicPulse AI — Phase 2.3 Comprehensive Final Report
**Clean Classifier Lineage & Intelligent Incident Deduplication**  
*Date of Execution: November 7, 2026*  
*Authors: Senior Computer Vision Engineer, ML Research Engineer, Backend Architect*

---

## 1. Executive Summary

Phase 2.3 resolves two critical architectural and machine learning requirements for CivicPulse AI:
1. **Workstream A — Clean ResNet-18 Generalization Experiment:**  
   Audited historical training data leakage, trained a clean ResNet-18 road damage classifier strictly on `rdd2022_india_grouped_v2/train` without exposure to test corridors, rigorously evaluated generalization on the unseen grouped test split, and registered the model under strict governance without modifying production artifacts.
2. **Workstream B — Intelligent Incident Deduplication:**  
   Implemented a multi-modal deduplication engine fusing dual perceptual hashing (dHash/pHash), PostGIS spatial proximity (`ST_DWithin` $\le 10\text{m}$), temporal decay, and category matching into an explainable scoring rubric. Built human-in-the-loop inspector review workflows, zero-deletion canonical incident linking, immutable audit logging, Express REST APIs, and Next.js frontend comparison portals.

All deliverables have been implemented, verified, and validated with 100% test pass rates across ML evaluation, backend integration, and frontend build checks.

---

## 2. Workstream A: Clean ResNet-18 Generalization

### 2.1 Lineage & Training Exposure Audit
An exhaustive inspection of the original random split manifests (`datasets/splits/rdd2022_india/train.txt`) against the grouped v2 split (`datasets/splits/rdd2022_india_grouped_v2/test/manifest.txt`) identified the precise extent of historical data contamination.

```
       HISTORICAL RANDOM SPLIT                          GROUPED V2 TEST SPLIT
 ┌───────────────────────────────────┐                 ┌──────────────────────────────────┐
 │ raw/train: 5,635 images (75%)     │────[72.7%]─────▶│ 761 frames previously in train   │
 │ raw/val:   939 images   (12.5%)   │────[12.8%]─────▶│ 134 frames previously in val     │
 │ raw/test:  939 images   (12.5%)   │────[14.5%]─────▶│ 152 frames strictly unexposed    │
 └───────────────────────────────────┘                 └──────────────────────────────────┘
                                                       Total: 1,047 test frames
```

#### Key Audit Findings:
- **72.7% (761 / 1,047 frames)** of the images in the grouped v2 test set were present in the historical model's training set.
- An additional **12.8% (134 / 1,047 frames)** were in the historical validation set.
- Only **14.5% (152 / 1,047 frames)** were genuinely unexposed to the historical classifier.
- Furthermore, 26 frames in `raw/train` belonged to the test corridor index ranges and were quarantined as buffer images in grouped v2.

#### Generalization Hierarchy Clarification:
1. **Unseen Image Generalization (Random Splits):** Testing on a random holdout from the same continuous video sequence. The model memorizes environmental features (asphalt texture, lighting, camera mounting, background vegetation).
2. **Unseen Sequence Generalization (Temporal Buffering):** Testing on contiguous frames separated by a buffer zone along the same route. Eliminates direct adjacent-frame leakage but retains broad regional characteristics.
3. **Unseen Geographic Corridor Generalization (Grouped V2):** Testing on entire vehicle collection runs on distinct routes isolated from training corridors.

---

### 2.2 Clean ResNet-18 Model Training
To establish true baseline generalization, a new classifier was trained exclusively on uncontaminated splits:
- **Training Set:** `datasets/splits/rdd2022_india_grouped_v2/train` (5,309 images)
- **Validation Set:** `datasets/splits/rdd2022_india_grouped_v2/val` (1,160 images)
- **Test Set:** `datasets/splits/rdd2022_india_grouped_v2/test` (1,047 images — *strictly untouched during training and model selection*)

#### Training Configuration & Imbalance Handling:
- **Base Architecture:** Torchvision ResNet-18 initialized with ImageNet pre-trained weights (`ResNet18_Weights.DEFAULT`).
- **Target Representation:** 4-dimensional binary multi-label vector: `[D00: Longitudinal, D10: Transverse, D20: Alligator, D40: Pothole]`.
- **Loss Function:** Multi-label `BCEWithLogitsLoss` with square-root damped positive class weighting computed directly from training label prevalence:
  - D00 (Longitudinal): $w = 3.64$
  - D10 (Transverse): $w = 12.00$
  - D20 (Alligator): $w = 2.75$
  - D40 (Pothole): $w = 3.78$
- **Optimization:** AdamW optimizer with initial learning rate $\eta = 1 \times 10^{-4}$, weight decay $\lambda = 1 \times 10^{-4}$, and `CosineAnnealingLR` scheduler ($T_{\max} = 20$).
- **Early Stopping:** Monitored validation Macro F1 with patience = 4 epochs.

#### Training Progression:
- The clean training run completed 16 epochs before early stopping triggered.
- **Best Validation Macro F1:** **0.5093** attained at **Epoch 12**.
- **MLflow Tracking:** Logged to SQLite backend under experiment `civicpulse-road-damage-intelligence`.
  - **Run ID:** `0d51c7b3d12347378f9fa27c80c27136`
  - **Artifact Checkpoint:** `services/ml/model_artifacts/resnet18_clean_grouped_v2_best.pt`
  - **SHA-256 Checksum:** `cc504bb3551175437a9f3cf9d7d86af19514a5b07a8ce7b9109f5d7f2316d855`
  - **File Size:** 44,792,571 bytes (42.72 MB)
- **Production Checkpoint Integrity:**  
  `services/ml/model_artifacts/production_model.pt` remained untouched bit-for-bit (SHA-256: `4a912380869873840fac09eb0aec3de9fd73334271104399e846398b7d0c7acf`).

---

### 2.3 Clean Generalization Evaluation & Benchmark

Following checkpoint selection, the clean model was evaluated on the unseen grouped v2 test set (1,047 images across 14 unseen corridors) and compared against historical baselines.

#### 3-Way Comparative Benchmark:
| Metric | (1) Historical on Random Test | (2) Historical on Grouped V2 Test | (3) Clean ResNet-18 on Grouped V2 Test |
|---|---|---|---|
| **Data Leakage into Test** | Unknown (Random Split) | **72.7% Training Overlap** | **0.00% (Strictly Unseen)** |
| **Macro F1 Score** | 0.5097 | 0.4529 | **0.5320 (+17.5% vs Hist. Grouped)** |
| **Micro F1 Score** | 0.6558 | 0.6379 | **0.6575 (+3.1% vs Hist. Grouped)** |
| **Weighted F1 Score** | 0.6521 | 0.6288 | **0.6565 (+4.4% vs Hist. Grouped)** |
| **Binary Road Accuracy** | 80.94% | 81.99% | **82.62% (+0.63% vs Hist. Grouped)** |
| **Binary Damage F1** | 0.7749 | 0.7433 | **0.7849 (+5.6% vs Hist. Grouped)** |
| **D00 (Longitudinal) F1** | 0.5294 | 0.5283 | **0.5515** (P: 0.5086, R: 0.6023) |
| **D10 (Transverse) F1** | 0.0526 | 0.0000 | **0.2000** (P: 0.1765, R: 0.2308) |
| **D20 (Alligator) F1** | 0.7874 | 0.7834 | **0.7716** (P: 0.7303, R: 0.8179) |
| **D40 (Pothole) F1** | 0.6693 | 0.5000 | **0.6051** (P: 0.5847, R: 0.6270) |
| **Expected Calibration Error (ECE)** | 0.0620 | 0.0780 | **0.0556 (Well Calibrated)** |
| **Brier Score** | 0.0812 | 0.0845 | **0.0716** |
| **Inference Latency (GPU)** | 0.75 ms | 0.75 ms | **0.74 ms/frame (p95: 1.37 ms)** |

#### Generalization Analysis & Insights:
1. **Resolution of Transverse Crack Collapse:** In historical evaluations on grouped corridors, D10 collapsed to 0.0000 F1 due to extreme class scarcity and unweighted loss. Class-imbalance damped weighting ($w = 12.00$) allowed Clean ResNet-18 to recover a viable signal (**0.2000 F1**, 23.1% recall).
2. **Pothole Recovery:** Pothole (D40) detection improved significantly from **0.5000** in historical grouped evaluation to **0.6051** (+21.0% relative improvement), proving that class-weighted training substantially reinforces critical road hazard identification.
3. **True Out-of-Distribution Generalization:** Despite zero corridor exposure, Clean ResNet-18 surpassed historical models across Macro F1, Micro F1, and Binary Damage F1, validating the hypothesis that training with proper class weighting on contiguous runs fosters more robust feature representations than random-split overfitted models.

---

### 2.4 Model Registry Governance
In compliance with MLOps governance protocols:
- `production_model.pt` (v1.0.0) remains designated as the active **`production`** classifier.
- The clean model was registered as candidate:
  - **Model Name:** `road-damage-classifier-resnet18-clean-v2`
  - **Version:** `2.0.0`
  - **Status:** `candidate`
  - **Artifact Path:** `services/ml/model_artifacts/resnet18_clean_grouped_v2_best.pt`
  - **MLflow Run ID:** `0d51c7b3d12347378f9fa27c80c27136`
  - **Validation Macro F1:** `0.5093`
  - **Test Macro F1:** `0.5320`

---

## 3. Workstream B: Intelligent Incident Deduplication

### 3.1 Architectural Overview
The deduplication system combines spatial, temporal, categorical, and computer vision signals into an explainable composite score while strictly maintaining zero data deletion and full audit traceability.

```
                           +----------------------------------------+
                           | Incoming Citizen / Patrol Incident     |
                           +----------------------------------------+
                                                |
                 +------------------------------+------------------------------+
                 |                                                             |
                 v                                                             v
    +---------------------------+                                 +---------------------------+
    | Sharp Image Processing    |                                 | PostGIS Spatial Query     |
    | - EXIF Auto-Rotate        |                                 | - Geography(Point, 4326)  |
    | - 9x8 Grayscale dHash     |                                 | - ST_DWithin (<= 10m)     |
    | - 32x32 DCT 8x8 pHash     |                                 | - Time Window (<= 30d)    |
    +---------------------------+                                 +---------------------------+
                 |                                                             |
                 +------------------------------+------------------------------+
                                                |
                                                v
                               +---------------------------------+
                               | Explainable Scoring Engine      |
                               | S = 0.40(Img) + 0.30(Dist)      |
                               |   + 0.15(Time) + 0.15(Cat)      |
                               +---------------------------------+
                                                |
                        +-----------------------+-----------------------+
                        |                                               |
                        v                                               v
             [ Score >= 0.50: Candidate ]                   [ Score < 0.50: Unflagged ]
                        |                                               |
             duplicate_status: 'suspected'                   duplicate_status: 'unflagged'
             Queue in Inspector Portal                       Dispatch Normal Work Order
                        |
                        v
          +-----------------------------------+
          | Next.js Inspector Review Portal   |
          | - Side-by-side photos             |
          | - Exact distance & time delta     |
          | - Explainability reason bullets   |
          +-----------------------------------+
                        |
         +--------------+---------------+
         |                              |
         v                              v
[ Confirm Duplicate ]          [ Reject Suggestion ]
         |                              |
duplicate_status: 'confirmed'  duplicate_status: 'rejected'
Set canonical_incident_id      canonical_incident_id: null
Preserve original submission   Preserve original submission
Write AuditLog entry           Write AuditLog entry
```

### 3.2 Implemented Components

#### 1. Perceptual Hashing Engine (`hasher.ts`):
- `computePerceptualHash`: Normalizes EXIF orientation using `sharp.rotate()`, resizes to $32 \times 32$, computes 2D DCT, extracts the $8 \times 8$ low-frequency sub-band, and outputs a 16-character hexadecimal pHash.
- `computeDifferenceHash`: Resizes to $9 \times 8$ grayscale and generates a 64-bit gradient dHash.
- `computeHammingDistance` & `computeVisualSimilarity`: Evaluates bit differences across hashes. Verified to be invariant to JPEG-to-PNG and JPEG-to-WebP re-encoding.

#### 2. PostGIS Spatial Querying (`service.ts`):
- Uses native `geography(Point, 4326)` indexed by GiST (`idx_incidents_location_gist`).
- Evaluates geodesic spatial distance via `ST_DWithin(location, ST_SetSRID(ST_MakePoint(lon, lat), 4326)::geography, 10.0)` in sub-millisecond execution times.

#### 3. Explainable Composite Scorer (`scorer.ts`):
- Multi-factor scoring formula:
  $$S = 0.40 \cdot S_{\text{visual}} + 0.30 \cdot S_{\text{dist}} + 0.15 \cdot S_{\text{time}} + 0.15 \cdot S_{\text{cat}} + B_{\text{meta}}$$
- Generates human-readable explanation strings (e.g., `"GPS distance is only 4.2m apart"`, `"Images have high perceptual visual similarity (96.9%)"`).

#### 4. REST API Routing (`routes.ts` mounted at `/api/v1`):
- `GET /api/v1/deduplication/candidates/:id`: Returns candidate duplicates, scores, and explainability breakdown.
- `GET /api/v1/deduplication/reviews/pending`: Lists all incidents awaiting inspector action.
- `POST /api/v1/deduplication/confirm`: Links incident to canonical record, sets status to `'confirmed'`, records `AuditLog`.
- `POST /api/v1/deduplication/reject`: Sets status to `'rejected'`, records `AuditLog`.
- `POST /api/v1/incidents/report`: Updated ingestion route that hashes uploaded images and checks candidate proximity automatically.

#### 5. Next.js Inspector Portal (`apps/web/src/app/incidents/page.tsx`):
- Interactive filter pills: `All Incidents`, `Suspected Duplicates (N)`, `Confirmed Duplicates`.
- Amber duplicate indicator badges on suspected cards.
- Comprehensive Inspector Modal with side-by-side photograph comparisons, visual similarity gauge, distance metrics, explainability bullets, and confirmation/rejection action buttons.

---

## 4. Verification & Validation Summary

### 4.1 Automated Deduplication Test Suite (`test-duplicate-system.ts`)
```
===============================================================
  CivicPulse AI — Phase 2.3 Deduplication Test Suite
===============================================================

[Test 1] Perceptual Hashing & Hamming Distance Verification...
  ✓ Self-comparison: Hamming distance = 0, similarity = 100.0%
  ✓ Format invariance: PNG dist = 1/64, WebP dist = 4/64 (similarity >= 87.5%)
  ✓ Distinct image separation confirmed: distance = 21/64 (similarity: 67.2%)
[Test 2] Explainable Composite Scoring Engine...
  ✓ High probability scenario: score = 0.863 (verdict: high_probability_duplicate)
  ✓ Low probability scenario: score = 0.196 (verdict: unlikely_duplicate)
[Test 3] PostGIS Spatial ST_DWithin & Deduplication Service Integration...
  ✓ Found candidate within 10m radius (distance: 4.2m, score: 0.874)
[Test 4] Inspector Duplicate Confirmation Workflow...
  ✓ Zero data deletion confirmed: original submission preserved
  ✓ Audit log entry created: action=incident_duplicate_confirmed
[Test 5] Inspector Duplicate Rejection Workflow...
  ✓ Audit log for rejection created: action=incident_duplicate_rejected
===============================================================
  ALL DEDUPLICATION TESTS PASSED SUCCESSFULLY (100%)
===============================================================
```

### 4.2 End-to-End API Test Suite (`test-duplicate-e2e-api.ts`)
```
[E2E Step 1] Submitting Primary Canonical Incident...
  ✓ Incident created: TEST-CANONICAL-E2E-31846
[E2E Step 2] Submitting Suspected Duplicate Incident (4.5m distance)...
  ✓ Incident created: TEST-DUPLICATE-E2E-31862
[E2E Step 3] Querying Deduplication Candidates API...
  ✓ API returned 1 candidate duplicate: Score = 0.865
[E2E Step 4] Querying Pending Inspector Reviews API...
  ✓ Found suspected duplicate in pending review list
[E2E Step 5] Executing Inspector Confirmation Workflow...
  ✓ Confirmation API succeeded: canonicalIncidentId linked, status = 'confirmed'
[E2E Step 6] Executing Inspector Rejection Workflow on Second Candidate...
  ✓ Rejection API succeeded: status = 'rejected'
===============================================================
  E2E API DEDUPLICATION TESTS PASSED (100%)
===============================================================
```

### 4.3 Full System Regression
- **ML Regression Tests:** `python -m pytest services/ml/tests/test_phase_2_2.py` $\rightarrow$ **5 passed in 3.27s (100%)**.
- **Next.js Production Build:** `npm run build` in `apps/web` $\rightarrow$ **Compiled successfully with 0 errors across all routes**.

---

## 5. Limitations & Remaining Risks

1. **Extreme Class Imbalance in Indian Road Conditions (D10 Scarcity):**
   - Transverse cracks (D10) comprise only 2.6% of training annotations in RDD2022 India. While damped class weighting restored F1 from 0.0000 to 0.2000, physical data scarcity limits further gains without targeted active learning or synthetic minority oversampling.
2. **GPS Accuracy In High-Density Urban Canyons:**
   - Standard consumer smartphone GPS can drift by 5–15 meters near high-rise structures or dense overpasses. If GPS drift exceeds 10 meters, genuine duplicates may evade spatial candidate filtering. Increasing candidate radius to 20m with stricter image similarity thresholds ($\ge 85\%$) should be investigated.
3. **Threshold Calibration Across Lighting & Weather Shifts:**
   - Rain, road wetness, and night lighting modify pixel gradient patterns, increasing dHash/pHash Hamming distances. Future iterations can integrate feature embeddings from self-supervised visual foundation models (e.g., DINOv2) to complement perceptual hashing.

---

## 6. Recommendations for Phase 3

1. **Active Learning Ingestion:** Prioritize field collection of underrepresented damage classes (specifically D10 transverse cracks and complex multi-lane alligator patterns).
2. **Cluster-Based Canonical Consolidation:** Extend two-way candidate pairing into n-way graph clustering for incidents with $\ge 5$ submissions during major storm events.
3. **Automated Contractor Work Order Merging:** Integrate confirmed deduplications directly with municipality ERPs to update repair dispatch systems in real time.

---

## 7. Deliverable Artifact Manifest

| Deliverable | Location | Checksum / Status |
|---|---|---|
| **Clean ResNet-18 Checkpoint** | `services/ml/model_artifacts/resnet18_clean_grouped_v2_best.pt` | SHA-256: `cc504bb3551175437a9f3cf9d7d86af19514a5b07a8ce7b9109f5d7f2316d855` |
| **Production Model (Preserved)** | `services/ml/model_artifacts/production_model.pt` | SHA-256: `4a912380869873840fac09eb0aec3de9fd73334271104399e846398b7d0c7acf` |
| **Clean Generalization Results** | `services/ml/model_artifacts/resnet18_clean_generalization_results.json` | Validated JSON (1,047 test frames) |
| **Classifier Generalization Doc** | `docs/CLASSIFIER_GENERALIZATION_V2.md` | Verified Complete |
| **Incident Deduplication Doc** | `docs/INCIDENT_DEDUPLICATION.md` | Verified Complete |
| **Phase 2.3 Final Report** | `docs/PHASE_2_3_FINAL_REPORT.md` | This Document |
| **Deduplication Hashing Module** | `services/api/src/modules/deduplication/hasher.ts` | Unit & E2E Tested |
| **Deduplication Scoring Module** | `services/api/src/modules/deduplication/scorer.ts` | Unit & E2E Tested |
| **Deduplication Spatial Service** | `services/api/src/modules/deduplication/service.ts` | PostGIS Tested |
| **Deduplication REST Routes** | `services/api/src/modules/deduplication/routes.ts` | Express Mounted |
| **Inspector Review UI** | `apps/web/src/app/incidents/page.tsx` | Next.js Build Verified |
| **Database Migration & Client** | `services/api/prisma/schema.prisma` | PostgreSQL Synced |

