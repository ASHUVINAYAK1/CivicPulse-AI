# CivicPulse AI — Phase 2.4 Comprehensive Final Report
**Final ML Benchmark Reconciliation, Deduplication Validation & System Safety Audit**  
*Date of Audit: November 7, 2026*  
*Authors: Senior ML Research Engineer, MLOps Auditor, Backend Quality Engineer*

---

## 1. Executive Summary

Phase 2.4 serves as the rigorous, independent verification gate between Phase 2 (Computer Vision & Intelligence) and Phase 3 (Production Scale & Field Deployment). 

This audit completed four exhaustive workstreams:
1. **Workstream A — ResNet-18 Metric Reconciliation:**  
   Resolved historical reporting discrepancies between Phase 2.1 (0.4529), Phase 2.2 (0.7640 / 0.4621), and Phase 2.3 (0.5320). Empirically reproduced every metric from saved checkpoints and dataset manifests. Formally verified that the clean ResNet-18 checkpoint was trained exclusively on `grouped_v2/train` without test leakage.
2. **Workstream B — Deduplication Validation & Ablation:**  
   Constructed an independent, curated benchmark across 8 realistic distress conditions (transcoding, compression, cropping, rotation, multi-angle photos, distant similar defects, same-road distinct defects, nearby unrelated defects) split into Calibration (12 pairs) and Held-out Evaluation (20 pairs) sets. Quantified the isolated and composite performance of dHash, pHash, GPS, time delta, and damage category.
3. **Workstream C — Score and Database Safety:**  
   Verified mathematical normalization of the composite score ($[0.0, 1.0]$). Validated circular relationship prevention, transitive hierarchy flattening, Prisma transaction consistency, audit log immutability, and zero-deletion citizen data guarantees.
4. **Workstream D — Full System Regression Suite:**  
   Executed the ML test suite (22/22 passed), Deduplication unit suite (5/5 passed), Deduplication E2E API suite (6/6 passed), Database safety suite (6/6 passed), and Next.js frontend production build (0 errors).

---

## 2. Workstream A: ResNet-18 Benchmark Reconciliation

### 2.1 Reconciliation of Discrepant Values
The investigation confirmed that the divergent Macro F1 numbers recorded in previous phase reports reflect different dataset splits and training exposure levels:

```
┌────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│                                   RESNET-18 METRIC RECONCILIATION SUMMARY                              │
├──────────────────────────┬──────────────────────┬─────────────┬──────────────┬────────────┬────────────┤
│ Evaluation Scenario      │ Checkpoint           │ Dataset     │ Samples ($N$)│ Overlap    │ Macro F1   │
├──────────────────────────┼──────────────────────┼─────────────┼──────────────┼────────────┼────────────┤
│ (1) Hist. on Random Test │ production_model.pt  │ raw/test    │ 1,166        │ 0.0%       │ **0.4529** │
│ (2) Hist. on Grouped (All│ production_model.pt  │ grouped/test│ 1,047        │ **72.7%**  │ **0.7640** │
│ (3) Hist. on Unexposed   │ production_model.pt  │ grouped/test│ 152          │ 0.0%       │ **0.4621** │
│ (4) Clean on Grouped Test│ clean_grouped_v2.pt  │ grouped/test│ 1,047        │ **0.0%**   │ **0.5320** │
└──────────────────────────┴──────────────────────┴─────────────┴──────────────┴────────────┴────────────┘
```

#### Detailed Root Cause Analysis:
1. **Scenario 1 (Phase 2.1 — 0.4529):**  
   Evaluated `production_model.pt` on the historical random test split (`datasets/raw/RDD2022_India/test`). Scored **0.4529 Macro F1**, with severe collapse on transverse cracks (D10 F1 = 0.0000).
2. **Scenario 2 (Phase 2.2 — 0.7640):**  
   Evaluated `production_model.pt` on the newly partitioned corridor test split (`rdd2022_india_grouped_v2/test`). The high score of **0.7640 Macro F1** was caused by **72.7% training data contamination** (761 of the 1,047 test frames had been in the model's training set).
3. **Scenario 3 (Phase 2.2 — 0.4621):**  
   Evaluated `production_model.pt` strictly on the **152 unexposed frames** in those test corridors. Performance dropped to **0.4621 Macro F1**, confirming that true generalization was near 0.45–0.46.
4. **Scenario 4 (Phase 2.3 — 0.5320):**  
   Evaluated the new **Clean ResNet-18** (`resnet18_clean_grouped_v2_best.pt`), trained exclusively on `grouped_v2/train` (5,309 images) with zero exposure to test corridors. With square-root damped positive class weighting, the clean model achieved **0.5320 Macro F1** (+15.1% relative improvement over unexposed baseline), recovering D10 transverse cracks (0.2000 F1) and significantly improving pothole detection (0.6051 F1).
5. **Phase 2.3 Report Transcription Swap:**  
   In `PHASE_2_3_FINAL_REPORT.md` (Section 2.3, Line 86), the author mistakenly placed the 0.4529 random test metric into the "Historical on Grouped V2 Test" column (which was 0.7640), creating the perception of a numerical contradiction. In `CLASSIFIER_GENERALIZATION_V2.md`, the numbers were documented accurately.

### 2.2 Clean Checkpoint Audit
- **Artifact:** `services/ml/model_artifacts/resnet18_clean_grouped_v2_best.pt`
- **Computed SHA-256:** `cc504bb3551175437a9f3cf9d7d86af19514a5b07a8ce7b9109f5d7f2316d855`
- **MLflow Tracking:** SQLite run `0d51c7b3d12347378f9fa27c80c27136` confirms training exclusively on 5,309 images and checkpoint selection at Epoch 12 based on Validation Macro F1 = 0.5093.
- **Production Baseline Integrity:** `services/ml/model_artifacts/production_model.pt` remains unchanged bit-for-bit (SHA-256: `4a912380869873840fac09eb0aec3de9fd73334271104399e846398b7d0c7acf`).

---

## 3. Workstream B: Deduplication Validation & Ablation

### 3.1 Curated Benchmark Evaluation
Evaluated across 20 held-out pairs (10 positive duplicates, 10 negative non-duplicates) unexposed during calibration:

| Feature / Signal Evaluated | Decision Rule | Precision | Recall | F1 Score | FPR | FNR | Accuracy |
|---|---|---|---|---|---|---|---|
| **dHash (Difference Hash)** | Hamming dist $\le 10$ ($S \ge 0.69$) | 71.4% | 50.0% | 0.5882 | 20.0% | 50.0% | 65.0% |
| **pHash (DCT Perceptual)** | Hamming dist $\le 10$ ($S \ge 0.69$) | 75.0% | 60.0% | 0.6667 | 20.0% | 40.0% | 70.0% |
| **GPS Distance Only** | Distance $\le 10.0\text{ meters}$ | 76.9% | **100.0%** | **0.8696** | 30.0% | **0.0%** | **85.0%** |
| **Time Delta Only** | Reported within 24 hours | 71.4% | **100.0%** | 0.8333 | 40.0% | **0.0%** | 80.0% |
| **Damage Category Only** | Defect categories match | 76.9% | **100.0%** | **0.8696** | 30.0% | **0.0%** | **85.0%** |
| **Composite Multi-Modal Scorer** | $S_{\text{composite}} \ge 0.50$ | 71.4% | **100.0%** | 0.8333 | 40.0% | **0.0%** | 80.0% |

### 3.2 Two-Stage Processing Pipeline Performance
```
  [ Stage 1: PostGIS ST_DWithin <= 10m ] ──▶ Filters 100% of distant false matches in 3.09 ms (p95: 4.07 ms)
  [ Stage 2: Sharp Image Hashing       ] ──▶ Computes dHash & pHash in 8.58 ms
  [ Stage 3: Explainable Scorer        ] ──▶ Evaluates composite score in 0.02 ms
  ─────────────────────────────────────────────────────────────────────────────────────────────
  Total Processing Latency per Pair      ──▶ Mean: 11.69 ms (p95: 13.31 ms)
```

---

## 4. Workstream C: Score & Database Safety Audit

All safety mechanisms were validated against PostgreSQL via `services/api/src/scripts/test-database-safety.ts`:

1. **Normalized Score Range:** The composite score formula is mathematically bounded in $[0.0, 1.0]$.
2. **Circular Relationship Prevention:**
   - Self-referencing duplicate attempts are blocked with HTTP 400.
   - Mutual circular loops ($A \rightarrow B$ when $B \rightarrow A$ exists) are strictly rejected with an exception.
3. **Transitive Hierarchy Flattening:**
   - Multi-hop duplicate chains ($C \rightarrow B \rightarrow A$) are automatically traversed and flattened so that child incident $C$ is linked directly to root canonical incident $A$.
4. **Transaction Atomicity:**
   - Incident linkage, status updates, child re-routing, and `AuditLog` generation are wrapped in atomic `prisma.$transaction(async (tx) => { ... })` blocks.
5. **Zero Data Deletion Guarantee:**
   - Confirmed that citizen submissions, image binaries, GPS coordinates, and reporter identities are **never deleted**. All data is preserved with full audit traceability.

---

## 5. Workstream D: Full System Regression Suite

| Test Suite | Scope | Target | Result | Status |
|---|---|---|---|---|
| **ML Pytest Suite** | Inference, metrics, pipelines, Phase 2.2 regression | `services/ml/tests/` (22 tests) | 22 passed in 10.39s | **PASSED (100%)** |
| **Deduplication Unit Suite** | Hashing, scoring, PostGIS, confirm/reject | `test-duplicate-system.ts` (5 tests) | 5 passed in 3.8s | **PASSED (100%)** |
| **Deduplication E2E API** | Full HTTP REST reporting, candidate review, audit | `test-duplicate-e2e-api.ts` (6 steps) | 6 steps passed in 4.5s | **PASSED (100%)** |
| **Database Safety Suite** | Circular loops, flattening, transactions, zero deletion | `test-database-safety.ts` (6 checks) | 6 checks passed in 3.5s | **PASSED (100%)** |
| **Frontend Production Build**| Turbopack compilation & TypeScript validation | `apps/web` (Next.js 16.4.5) | Compiled successfully (0 errors) | **PASSED (100%)** |

---

## 6. Model Registry Governance & Status

In accordance with strict MLOps release protocols:
- `production_model.pt` (v1.0.0) remains designated as the **`production`** classifier.
- `resnet18_clean_grouped_v2_best.pt` (v2.0.0) remains registered as **`candidate`**. No automatic promotion occurred.

---

## 7. Phase 3 Recommendations & Limitations

1. **D10 Data Scarcity:**  
   Transverse cracking (D10) represents only 2.6% of annotations in RDD2022 India. While class weighting restored F1 from 0.0000 to 0.2000, physical data scarcity in Indian road conditions remains a constraint. Phase 3 should deploy targeted active learning collection for transverse defects.
2. **Dense Defect Clusters:**  
   In dual-carriageway roads where defects occur within 6–9 meters across opposite traffic lanes, inspectors should utilize the side-by-side photographic review portal to verify road bearing before confirming duplicate links.
3. **Foundation Model Embeddings:**  
   For Phase 3 mobile patrol units, perceptual hashing can be augmented with lightweight self-supervised visual embeddings (e.g., MobileNetV4 / DINOv2-small) for enhanced angle-invariant visual matching.

---

## 8. Artifact and Deliverable Manifest

| Deliverable | Location | SHA-256 / Checksum |
|---|---|---|
| **Clean ResNet-18 Checkpoint** | `services/ml/model_artifacts/resnet18_clean_grouped_v2_best.pt` | `cc504bb3551175437a9f3cf9d7d86af19514a5b07a8ce7b9109f5d7f2316d855` |
| **Production ResNet-18 (Unchanged)**| `services/ml/model_artifacts/production_model.pt` | `4a912380869873840fac09eb0aec3de9fd73334271104399e846398b7d0c7acf` |
| **Benchmark Reconciliation Document** | `docs/PHASE_2_4_BENCHMARK_RECONCILIATION.md` | Verified Complete |
| **Deduplication Validation Document** | `docs/DEDUPLICATION_VALIDATION.md` | Verified Complete |
| **Phase 2.4 Final Report** | `docs/PHASE_2_4_FINAL_REPORT.md` | This Document |
| **Deduplication Benchmark Script** | `services/api/src/scripts/validate-deduplication-benchmark.ts` | Complete |
| **Database Safety Test Script** | `services/api/src/scripts/test-database-safety.ts` | Complete |
| **PostGIS Latency Test Script** | `services/api/src/scripts/measure-retrieval-latency.ts` | Complete |

