# CivicPulse AI — Phase 3: Data Readiness and Governance Audit

**Document Version:** 1.0.0  
**Audit Date:** November 7, 2026  
**Auditor:** Senior ML Engineer, Data Scientist & MLOps Architect  
**Scope:** PostgreSQL/PostGIS database entities, RDD2022 India CV dataset, maintenance logs, historical forecasts, and model registry artifacts.

---

## 1. Executive Summary

This audit assesses the data availability, label integrity, and statistical viability for the five proposed Phase 3 Machine Learning and Predictive Intelligence workstreams:
1. **Severity Prediction**
2. **Maintenance Priority Prediction**
3. **Repair Cost and Duration Estimation**
4. **Geographical Hotspot Detection**
5. **Incident Volume Forecasting**

In strict adherence to the CivicPulse AI engineering guidelines, **no synthetic or seed data will be disguised as real operational data**, and **no models will be claimed as validated on unrepresentative or arbitrary ground truth**.

---

## 2. Audit Findings by Database Entity

A full audit of the production PostgreSQL schema (`services/api/prisma/schema.prisma`) and live database records was conducted on November 7, 2026.

### 2.1 Entity Record Census

| Database Entity | Total Rows | Verified Rows | Key Missing Attributes / Limitations |
|---|---|---|---|
| `users` | 5 | 5 | All 5 PRD roles represented (citizen, inspector, admin, maintenance, ml_admin). |
| `incidents` | 66 | 4 status='verified', 3 verifiedAt | All 66 reported within a 5.8-hour window on 2026-11-07. No multi-month operational history. |
| `incident_images` | 63 | N/A | Linked to MinIO and local storage; perceptual hash and dHash populated. |
| `ai_predictions` | 63 | N/A | 58 object detection / identification, 5 damage classification. |
| `incident_verifications` | 6 | 6 | Only 6 human inspector ground-truth verifications exist. |
| `maintenance_tasks` | 2 | 0 completed | 1 assigned, 1 in-progress. 0 completed repairs. Actual cost and actual duration are 100% NULL. |
| `maintenance_updates` | 2 | N/A | 2 dispatch progress notes. No materials or labor records. |
| `model_registry` | 9 | N/A | Includes ResNet-18, MobileNetV3, EfficientNet-B0, SSDLite320, clean ResNet-18 v2, plus legacy seed stubs. |
| `incident_forecasts` | 6 | N/A | 6 future projection rows generated on 2026-11-07 for `delhi-central-zone` (Dec 2026 – Feb 2027). |

---

### 2.2 Deep Dive: Incident Records & Verifications

```
Total Incidents: 66
Status Distribution:
  - submitted:            56 (84.8%)
  - verified:              4 (6.1%)
  - maintenance_assigned:  1 (1.5%)
  - repair_in_progress:    1 (1.5%)
  - duplicate:             2 (3.0%)
  - under_review:          1 (1.5%)
  - reported:              1 (1.5%)

Damage Type Distribution:
  - pothole:              60 (90.9%)
  - D40 (Pothole code):    2 (3.0%)
  - D20 (Alligator crack): 1 (1.5%)
  - D00 (Longitudinal):    1 (1.5%)
  - D10 (Transverse):      1 (1.5%)
  - alligator_crack:       1 (1.5%)

Reported Severity Distribution:
  - critical:             59 (89.4%)
  - low:                   2 (3.0%)
  - high:                  1 (1.5%)
  - medium:                1 (1.5%)
  - NULL:                  3 (4.5%)
```

#### Inspector Ground Truth (`incident_verifications`):
- **Total Verification Records:** **6**
- **Verified Classes:** `pothole`: 3, `D40`: 1, `D20`: 1, `D00`: 1.
- **Verified Severities:** `critical`: 3, `high`: 1, `medium`: 1, `low`: 1.
- **Remarks Present:** 6 / 6.

> [!WARNING]
> **Sample Size Alert (Severity Ground Truth):**  
> With only $N=6$ inspector-verified severity labels in the operational database, training a complex multi-class or ordinal supervised classifier (e.g., Random Forest or Gradient Boosting) directly on `incident_verifications` would yield catastrophic overfitting and meaningless validation metrics. The remaining 60 unverified records in `incidents` have uncurated citizen or seed severity entries that cannot be treated as ground truth.

---

### 2.3 Deep Dive: Maintenance & Repair Outcomes

```
Total Maintenance Tasks: 2
Status Distribution:
  - assigned:             1
  - in_progress:          1
  - completed:            0 (0.0%)

Cost & Duration Tracking:
  - estimated_cost:       2 present (100.0%) [both recorded as ₹15,400]
  - actual_cost:          0 present (0.0% populated, 2 NULL)
  - estimated_duration:   2 present (100.0%) [both recorded as 6 hours]
  - actual_duration:      0 present (0.0% populated, 2 NULL)
  - started_at:           1 present
  - completed_at:         0 present
```

> [!CAUTION]
> **Data Absence Alert (Cost & Duration):**  
> The database contains **zero completed maintenance records** and **zero actual cost or actual duration entries**. Furthermore, granular fields required for cost estimation (e.g., asphalt volume, labor hours, equipment dispatch, material cost) do not currently exist in the `maintenance_tasks` schema. Training regression models on $N=0$ real targets is impossible without fabricating labels.

---

### 2.4 Deep Dive: Geospatial Coordinates

```
Total Incidents: 66
Valid Latitude & Longitude: 66 / 66 (100.0%)
Geographic Classification:
  - Region 1 (Bengaluru Metropolitan Area): 58 incidents (87.9%)
      * Dense Outer Ring Road Corridor: 57 incidents (12.9716° N, 77.5946° E)
      * South Corridor Outlier (INC-357677-895): 1 incident (12.9260° N, 77.6762° E, 10,017m southeast)
  - Region 2 (Delhi NCR): 8 incidents (12.1%)
      * Connaught Place / Barakhamba / University Road: 8 incidents (28.6139° N - 28.6412° N, 77.2045° E - 77.2250° E)
  - Ambiguous / Unassigned: 0 incidents (0.0%)
Address Populated: 63 / 66 (95.5%)
PostGIS Extension: Enabled (PostgreSQL 16 + PostGIS 3.4)
```

> [!NOTE]
> **Spatial Readiness & Geographic Governance Reconciliation:**  
> Stage 1 originally cited Delhi NCR based on seed records. Exhaustive coordinate auditing in Stage 5 and Stage 3 proved that the 66 records actually span two distinct urban centers: Bengaluru ($N=58$) and Delhi ($N=8$), separated by 1,740 km. Spatial algorithms (K-Means, DBSCAN, PostGIS `ST_DWithin`) operate with spherical Haversine / PostGIS geography calculations across both regions. Bounding-box based regional assignment is validated and documented in `docs/PHASE_3_STAGE_3_MAINTENANCE_PRIORITY.md`.

---

### 2.5 Deep Dive: Longitudinal Timestamps & Time-Series

```
Total Incidents: 66
Earliest Reported Timestamp: 2026-11-07T11:27:04.544Z
Latest Reported Timestamp:   2026-11-07T17:14:59.852Z
Observation Window:          5 hours, 47 minutes, 55 seconds (Single day)
Historical Longitudinal Span: 0 months, 0 weeks, 0 multi-day series
```

> [!WARNING]
> **Temporal Readiness Alert (Time-Series Forecasting):**  
> All 66 incident rows in the operational database were created on November 7, 2026 within a 5.8-hour span. There is **zero historical longitudinal data** (no multi-month or multi-year daily incident counts).  
> The 6 rows in `incident_forecasts` are synthetic seed artifacts created during initial database bootstrapping.  
> Training an LSTM or auto-regressive statistical model (ARIMA) on a 5.8-hour single-day cluster would violate fundamental temporal modeling requirements and generate spurious projections.

---

### 2.6 Deep Dive: Computer Vision Dataset (RDD2022 India)

```
Location: datasets/raw/RDD2022_India/ and datasets/splits/rdd2022_india_grouped_v2/
Image Modality: Smartphone dashcam / ground photography
Classes: D00 (Longitudinal), D10 (Transverse), D20 (Alligator), D40 (Pothole)
Annotations: XML/PASCAL-VOC bounding boxes
Bounding Box Features Extractable:
  - Defect count per image
  - Bounding box areas and relative surface coverage (%)
  - Defect aspect ratios and spatial distribution
Ground Truth Severity Labels: None (RDD2022 provides damage classes and geometry, not severity grades)
```

---

## 3. Model Data Availability Matrix

| Proposed ML Capability | Required Features | Available Features | Ground Truth Target | Target Sample Size ($N$) | Data Readiness Classification |
|---|---|---|---|---|---|
| **1. Severity Prediction** | Damage category, defect count, bounding box area/ratio, classification confidence, road type | Damage category (66), CV bounding box metrics from RDD2022, confidence | Inspector-verified severity (`low`, `medium`, `high`, `critical`) | **$N = 6$** verified in DB | **Class D**: Suitable only for rule-based & experimental baseline with CV proxy features. |
| **2. Maintenance Priority** | Verified severity, damage type, defect extent, aging/dwell time, nearby density, safety hazard | Damage type (66), age, spatial density, verified severity (6) | Administrative Priority decision (`low`, `medium`, `high`, `critical`) | **$N = 2$** tasks, 63 heuristic priority tags | **Class D**: Suitable for transparent multi-factor rule engine + exploratory baseline. |
| **3. Repair Cost Estimation** | Damage type, surface area ($m^2$), depth/cavity, asphalt/patch material, labor hours | Damage type (66), bounding box area proxy | Actual repair cost (`actual_cost`) | **$N = 0$** completed repairs | **Class C**: Requires additional real-world data. Schema collection extension needed. |
| **4. Repair Duration Estimation** | Repair category, severity, crew size, weather conditions, equipment type | Damage type (66), severity (6 verified) | Actual repair duration (`actual_duration`) | **$N = 0$** completed repairs | **Class C**: Requires additional real-world data. Schema collection extension needed. |
| **5. Geographical Hotspot Detection** | Latitude, longitude, spatial density, cluster radius | Latitude & Longitude (66 / 66 valid), PostGIS spatial index | Unsupervised clusters & density anomalies | **$N = 66$** real GPS points | **Class B**: Ready for exploratory analysis (K-Means scratch/sklearn, DBSCAN, PostGIS). |
| **6. Incident Volume Forecasting** | Daily/weekly incident count sequences over 12–36 months, seasonality | 66 timestamps within a 5.8-hour window on 2026-11-07 | Future incident count per zone | **$N = 0$** multi-day historical observations | **Class C / D**: Pipeline implementation with moving average/statistical baseline, explicit readiness limitations, no unvalidated LSTM. |

---

## 4. Readiness Classification Definitions

- **Class A — Ready for Supervised Training:**  
  Adequate verified ground-truth labels ($N \ge 500$), balanced target distribution, and validated independent features.
- **Class B — Ready for Exploratory Analysis:**  
  Features are complete and geometrically/statistically sound for unsupervised algorithms or spatial clustering, though operational sample size may be moderate.
- **Class C — Requires Additional Real-World Data:**  
  The target labels or essential explanatory covariates are absent from historical operations. Requires production collection instrumentation before any statistical validation can occur.
- **Class D — Suitable Only for Experimental or Rule-Based Implementation:**  
  Labels are insufficient for generalizable supervised learning. The system must employ deterministic, transparent domain heuristics (e.g., IRC / municipal civil engineering rules) and establish baseline models strictly for pipeline verification.

---

## 5. Identified Data Governance & Label Gaps

1. **Lack of Closed Maintenance History:**  
   The system has recorded 2 maintenance dispatches, but zero work order completions. Actual costs, actual durations, labor hours, and material consumption are completely unrecorded.
2. **Cold-Start Verification Queue:**  
   Only 6 incidents have been reviewed and verified by an inspector. 60 incidents remain unverified or have citizen-reported severity that cannot be used as training labels.
3. **Temporal Truncation:**  
   The entire operational database reflects a single day of test and benchmark reporting. Time-series models that assume multi-scale periodicity (day-of-week, seasonal rain cycles, monsoon effects) cannot learn from 5.8 hours of records.
4. **Schema Deficiencies in Maintenance Tracking:**  
   The existing `maintenance_tasks` schema lacks fields for repair material (cold mix vs hot mix asphalt), crew size, machinery deployed, and surface area repaired ($m^2$).

---

## 6. Recommended Phase 3 Execution Strategy & Governance Protocols

Based on this audit, the implementation plan will be executed strictly in accordance with empirical reality:

### Protocol 1: Geographical Hotspot Detection (Stage 5) — Priority 1
- **Approach:** Fully operational on real GPS coordinates.
- Implement K-Means from scratch with Haversine distance, Scikit-learn spherical K-Means, and DBSCAN for density-based noise separation.
- Integrate PostGIS spatial clustering queries and render dynamic cluster polygons on the Next.js Leaflet map.

### Protocol 2: Severity Prediction (Stage 2) — Priority 2
- **Approach:** Rule-based + Computer Vision Proxy Pipeline.
- Use RDD2022 India defect bounding boxes (area, count, aspect ratio) combined with engineering standards (e.g., Pothole / D40 with area $> 5\%$ of image frame $\rightarrow$ Critical/High) to construct a reproducible severity pipeline.
- Implement Logistic Regression, Decision Tree, Random Forest, and Gradient Boosting on the structured CV-derived features.
- Explicitly document that models are evaluated on CV geometric damage extent proxies and are flagged as candidate baselines pending inspector verification volume.

### Protocol 3: Maintenance Priority Engine (Stage 3) — Priority 3
- **Approach:** Transparent Multi-Factor Decision Engine + Classical ML Baseline.
- Combine verified severity, damage category, spatial incident density (from PostGIS/Haversine), and incident aging.
- Compare transparent rule-based priority matrices against Logistic Regression, Random Forest, and Gradient Boosting.
- Provide detailed rationale explanations for every priority score. Prevent automatic dispatch without human approval.

### Protocol 4: Repair Cost & Duration Estimation (Stage 4) — Priority 4
- **Approach:** Data Collection Schema Migration + Parametric Civil Engineering Baseline.
- Update Prisma schema to add maintenance tracking attributes (material type, repair square meters, crew size, actual labor cost, actual equipment cost).
- Implement parametric engineering estimation algorithms (standard municipal Schedule of Rates / CPWD norms) as an explicitly experimental baseline.
- Provide Linear Regression and Random Forest Regressor scaffolding that automatically becomes trainable as completed maintenance records accumulate.

### Protocol 5: Incident Volume Forecasting (Stage 6) — Priority 5
- **Approach:** Pipeline Implementation & Statistical Baselines with Explicit Readiness Limitations.
- Build the aggregation and forecasting pipeline supporting daily, weekly, and monthly resampled buckets.
- Implement Moving Average, Naive Seasonal, and Linear Trend baselines.
- Implement the LSTM architecture comparison, but strictly enforce a data-readiness gate: if historical observations $N < 30$ time steps, the pipeline refuses to fit the LSTM and reports a `DATA_INSUFFICIENT_FOR_DEEP_LEARNING` status code.

### Protocol 6: Database & Backend Integration (Stage 7 & 8)
- Enhance Prisma schema with models for:
  - `SeverityAssessment`
  - `MaintenancePriorityRecommendation`
  - `RepairEstimate`
  - `HotspotCluster`
  - `IncidentForecastRecord`
- Maintain audit logs, model provenance, and version tracking.
- Track all models in MLflow and PostgreSQL `model_registry`.

### Protocol 7: Frontend Predictive Maintenance Dashboard (Stage 9)
- Build out `apps/web/src/app/maintenance/page.tsx`, `apps/web/src/app/analytics/page.tsx`, and `apps/web/src/app/map/page.tsx` into an integrated predictive intelligence suite displaying predictions, confidence intervals, provenance tags, and human verification badges.

