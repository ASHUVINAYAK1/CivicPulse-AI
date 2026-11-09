# CivicPulse AI — Phase 3, Stage 4: Repair Cost & Duration Estimation

**Status:** Completed & Verified  
**Date:** November 2026  
**Module:** Predictive Maintenance Intelligence — Repair Estimation  
**Services:** `services/api` (Express / TypeScript / PostGIS), `services/ml` (Python / Scikit-Learn Scaffolding), `apps/web` (Next.js 16)

---

## Executive Summary

Phase 3 Stage 4 delivers a **transparent, explainable parametric repair cost and duration estimation pipeline** for road distress incidents across municipal road networks.

Because the operational database contains **zero completed maintenance tasks and no actual cost or duration ground truth**, this stage adheres strictly to core MLOps governance:
1. **Supervised Regression Locked:** Machine-learning regression models (Linear Regression, Random Forest, Gradient Boosting) are strictly gated with status `DATA_INSUFFICIENT` ($N=0/50$ completed repairs). Training predictive regression models on synthetic data or parametric formula outputs is prohibited to prevent circular self-prediction.
2. **Physical Area Separation:** The system enforces an absolute boundary between 2D optical bounding-box pixel ratios and physical ground measurements ($m^2$). When physical dimensions are unmeasured, the pipeline returns `MEASUREMENT_PENDING` and refuses to invent physical dimensions.
3. **Regional Isolation:** Separate, non-interchangeable rate schedules are maintained for Bengaluru (`BENGALURU_METRO`) and Delhi (`DELHI_NCR`). Unconfigured territories explicitly return `RATE_SOURCE_UNAVAILABLE`.
4. **Estimated vs Actual Separation:** Parametric estimates are explicitly distinguished from actual municipal expenditures.

---

## 1. Regional Rate-Source Audit

An exhaustive repository search was conducted to inspect whether official municipal or state government Schedule of Rates (SoR) documents were bundled in the codebase:

```bash
Audit Command:
Get-ChildItem -Recurse -File | Where-Object { $_.Name -match "sor|cpwd|bbmp|kpwd|schedule_of_rates" }
```

### Audit Findings:
- **Bengaluru (Karnataka PWD / BBMP):** **No official Schedule of Rates documents exist in the repository.**
- **Delhi (CPWD / Delhi PWD):** **No official Schedule of Rates documents exist in the repository.**
- **Governance Action:** In accordance with instructions, CivicPulse AI **does not invent official rates or fabricate government gazette citations**.
- **Implementation:** Built a configurable regional rate-source interface (`services/api/src/modules/repair-estimation/rate-schedules.ts`). Active schedules (`KPWD-BBMP-PROVISIONAL-2026.1` and `CPWD-DELHI-PROVISIONAL-2026.1`) are marked `isOfficialDocument = false` and accompanied by explicit disclosures that they represent transparent municipal engineering parametric baselines for decision support.
- **Cross-Regional Prohibition:** The system strictly forbids applying Bengaluru rates to Delhi incidents or vice versa. Unconfigured regions return `RATE_SOURCE_UNAVAILABLE`.

---

## 2. Feature Pipeline & Physical Measurement Boundary

### 2.1 The Optical vs Physical Boundary
Computer vision object detectors output bounding boxes as image-plane normalized coordinates:
$$a_{\text{norm}} = \frac{w_{\text{pixel}} \times h_{\text{pixel}}}{W_{\text{image}} \times H_{\text{image}}} \in [0.0, 1.0]$$

> [!CAUTION]
> **Fundamental Optical Principle:** Normalized bounding-box area $a_{\text{norm}}$ is an optical perspective ratio, **NOT physical road surface area in square meters ($m^2$)**. Optical pixel extent varies dramatically based on drone altitude, smartphone camera focal length, tilt angle, and distance to the pavement.

### 2.2 The `MEASUREMENT_PENDING` Gate
If physical repair dimensions are unrecorded in the field, the estimator returns:
```json
{
  "estimationStatus": "MEASUREMENT_PENDING",
  "totalEstimatedCostINR": null,
  "totalEstimatedDurationHours": null,
  "measurementCaveat": "Physical repair dimensions are unmeasured. 2D bounding-box pixel area ratio on camera frame cannot be converted into physical ground area without field measurement or photogrammetric ground calibration."
}
```
Only when empirical field measurements ($m^2$, depth) or calibrated photogrammetric ground references are submitted does the engine calculate parametric financial and duration values.

---

## 3. Parametric Cost Estimation Formulation

When physical measurements are available, the parametric engine computes four distinct components in Indian Rupees (`INR ₹`):

$$\text{Total Estimated Cost} = \text{Materials} + \text{Labor} + \text{Equipment} + \text{Mobilization}$$

### 3.1 Material Cost
- **Pothole Cavity Patching (Cold Mix Bituminous Macadam):**
  $$V = \text{Area } (m^2) \times \text{Depth } (m)$$
  $$\text{Cost}_{\text{asphalt}} = V \times \text{Unit Rate}_{\text{cold mix}}$$
  $$\text{Cost}_{\text{tack coat}} = \text{Area } (m^2) \times \text{Unit Rate}_{\text{emulsion}}$$
- **Alligator Structural Base Fatigue (Dense Bituminous Macadam):**
  $$V = \text{Area } (m^2) \times \text{Depth } (m)$$
  $$\text{Cost}_{\text{hot mix}} = V \times \text{Unit Rate}_{\text{hot mix}}$$
  $$\text{Cost}_{\text{tack coat}} = \text{Area } (m^2) \times \text{Unit Rate}_{\text{emulsion}}$$
- **Linear Crack Sealing (Polymer Modified Bitumen Hot-Pour):**
  $$L = \max(2.0\text{ m}, \text{Area } (m^2) \times 3.0\text{ m/m}^2)$$
  $$\text{Cost}_{\text{sealant}} = L \times \text{Unit Rate}_{\text{crack sealant}}$$

### 3.2 Labor Cost
Based on a standard 4-person municipal road maintenance squad:
1. **1 Skilled Road Mason / Asphalt Lead:** Excavation trimming, screeding, leveling.
2. **2 Unskilled Maintenance Laborers:** Debris removal, wheelbarrow transit, compaction assist.
3. **1 Dedicated Traffic Safety Marshall:** Arterial lane cordoning, safety signage, traffic flow control.

$$\text{Active Hours} = \text{Operational Hours} + \text{Traffic Setup Hours}$$
$$\text{Cost}_{\text{labor}} = \text{Active Hours} \times (R_{\text{mason}} + 2 \cdot R_{\text{laborer}} + R_{\text{marshall}})$$

### 3.3 Equipment Cost
Includes dedicated plant machinery:
1. **Vibratory Tandem Roller / Plate Compactor:** Layer compaction for the full active duration.
2. **Asphalt Cutter / Jackhammer:** Straight vertical key-cut edging ($50\%$ of active duration).
3. **Air Compressor & High-Pressure Dust Blower:** Cavity air-jet dusting ($0.5$ hours).

$$\text{Cost}_{\text{equipment}} = \sum (\text{Machine Hours} \times \text{Hourly Machine Rate})$$

### 3.4 Mobilization & Site Management Overhead
Applied as a transparent regional percentage over direct costs ($10\%$ in Bengaluru, $12\%$ in Delhi due to arterial congestion constraints):
$$\text{Cost}_{\text{mobilization}} = (\text{Materials} + \text{Labor} + \text{Equipment}) \times \frac{\text{Overhead \%}}{100}$$

---

## 4. Duration Estimation Formulation

Operational durations are estimated based on civil engineering production-rate baselines:

$$\text{Total Hours} = \text{Operational Hours} + \text{Setup Hours} + \text{Curing Wait Hours}$$
$$\text{Total Working Days} = \max\left(0.25, \left\lceil \frac{\text{Total Hours}}{8.0} \right\rceil\right)$$

### 4.1 Documented Productivity Rates
- **Pothole Cavity Patching:**
  - Bengaluru: $1.5\text{ m}^2 / \text{crew-hour}$
  - Delhi: $1.25\text{ m}^2 / \text{crew-hour}$ (restricted work windows)
- **Alligator Structural Resurfacing:** $3.5 - 4.0\text{ m}^2 / \text{crew-hour}$
- **Crack Sealing:** $12.0 - 15.0\text{ linear meters} / \text{crew-hour}$
- **Traffic Setup & Cordoning:** $1.0\text{h}$ (Bengaluru) / $1.5\text{h}$ (Delhi arterial corridors)
- **Asphalt Curing & Compaction Cooling Wait:** $1.5\text{h}$ (Cold Mix) / $2.0\text{h}$ (Hot Mix)

---

## 5. Regional Schedule Specifications

| Specification | Bengaluru Metropolitan Area (`BENGALURU_METRO`) | Delhi NCT (`DELHI_NCR`) |
| :--- | :--- | :--- |
| **Schedule Version** | `KPWD-BBMP-PROVISIONAL-2026.1` | `CPWD-DELHI-PROVISIONAL-2026.1` |
| **Currency** | `INR` (`₹`) | `INR` (`₹`) |
| **Effective Date** | April 1, 2026 | April 1, 2026 |
| **Official Document Status** | `false` (Parametric Decision Support Benchmark) | `false` (Parametric Decision Support Benchmark) |
| **Cold Mix Asphalt Rate** | ₹ 4,200.00 / $m^3$ | ₹ 4,500.00 / $m^3$ |
| **Hot Mix Bituminous Rate** | ₹ 5,100.00 / $m^3$ | ₹ 5,400.00 / $m^3$ |
| **Tack Coat Primer Rate** | ₹ 45.00 / $m^2$ | ₹ 50.00 / $m^2$ |
| **Crack Sealant Rate** | ₹ 95.00 / linear meter | ₹ 105.00 / linear meter |
| **Skilled Mason Hourly Rate** | ₹ 120.00 / hr | ₹ 135.00 / hr |
| **Unskilled Laborer Hourly** | ₹ 85.00 / hr | ₹ 95.00 / hr |
| **Traffic Marshall Hourly** | ₹ 80.00 / hr | ₹ 90.00 / hr |
| **Roller Compactor Hourly** | ₹ 450.00 / hr | ₹ 500.00 / hr |
| **Asphalt Joint Cutter** | ₹ 250.00 / hr | ₹ 280.00 / hr |
| **Pneumatic Blower** | ₹ 200.00 / hr | ₹ 220.00 / hr |
| **Mobilization Overhead** | $10.0\%$ | $12.0\%$ |

---

## 6. Supervised ML Regression Readiness Assessment

### 6.1 Database Census
- **Completed Maintenance Tasks:** **0**
- **Recorded Actual Costs:** **0**
- **Recorded Actual Durations:** **0**
- **Total Maintenance Work Orders:** **2** (`INC-2024-001` pending, `INC-2024-002` in progress)

### 6.2 Data Readiness Gate (`train_repair_regression_experimental.py`)
- **Status:** **`DATA_INSUFFICIENT`**
- **Threshold Required:** $N \ge 50$ completed work orders with verified actual completion expenditures.
- **Strict Anti-Hallucination Policy:** Training Linear Regression, Random Forest, or Gradient Boosting on parametric formula outputs or synthetic data would constitute circular hallucination and false precision.
- **Scaffolding:** [`services/ml/training/train_repair_regression_experimental.py`](file:///c:/Users/Admin/Desktop/my-projects/CivicPulse-AI/services/ml/training/train_repair_regression_experimental.py) implements the immutable gate and emits an audit report at [`services/ml/model_artifacts/repair_regression_readiness_audit.json`](file:///c:/Users/Admin/Desktop/my-projects/CivicPulse-AI/services/ml/model_artifacts/repair_regression_readiness_audit.json).

---

## 7. Database Integration & Non-Destructive Migrations

To store repair estimates while preserving existing schema models:

```sql
CREATE TABLE IF NOT EXISTS repair_cost_estimates (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  incident_id UUID NOT NULL REFERENCES incidents(id) ON DELETE CASCADE,
  task_id UUID REFERENCES maintenance_tasks(id) ON DELETE SET NULL,
  region VARCHAR(50) NOT NULL,
  rate_schedule_version VARCHAR(100) NOT NULL,
  estimationStatus VARCHAR(50) NOT NULL,
  measured_area_sq_meters DOUBLE PRECISION,
  measurement_source VARCHAR(50) NOT NULL DEFAULT 'unmeasured_missing',
  depth_meters DOUBLE PRECISION,
  material_type VARCHAR(100),
  material_quantity DOUBLE PRECISION,
  material_unit VARCHAR(30),
  material_cost DOUBLE PRECISION,
  labor_hours DOUBLE PRECISION,
  labor_cost DOUBLE PRECISION,
  equipment_hours DOUBLE PRECISION,
  equipment_cost DOUBLE PRECISION,
  mobilization_cost DOUBLE PRECISION,
  total_estimated_cost DOUBLE PRECISION,
  currency VARCHAR(10) NOT NULL DEFAULT 'INR',
  estimated_duration_hours DOUBLE PRECISION,
  estimated_duration_days DOUBLE PRECISION,
  crew_size INTEGER,
  assumptions JSONB,
  breakdown JSONB,
  missing_information JSONB,
  computed_at TIMESTAMP(6) NOT NULL DEFAULT CURRENT_TIMESTAMP,
  is_actual_cost BOOLEAN NOT NULL DEFAULT FALSE
);
```

**Zero Data Loss:** All 66 original incident records, PostGIS `location (geography)` columns, and AI predictions remain untouched.

---

## 8. Backend API Contracts

Mounted under `/api/v1/repair-estimates` (`services/api/src/modules/repair-estimation/routes.ts`):

| Endpoint | Method | Role Check | Description |
| :--- | :--- | :--- | :--- |
| `/summary` | `GET` | Public | System-wide census of measured vs unmeasured incidents, average cost, regional breakdown, and regression readiness (`DATA_INSUFFICIENT`). |
| `/rates` | `GET` | Public | Returns active regional rate schedules with explicit governance disclosures. |
| `/model-status` | `GET` | Public | Supervised regression gate state (`DATA_INSUFFICIENT`, 0/50 completed repairs). |
| `/incidents/:id` | `GET` | Public | Retrieves latest repair estimate for an incident. Returns `MEASUREMENT_PENDING` if unmeasured. |
| `/calculate` | `POST` | Public / Inspector | Computes or updates an estimate using physical dimensions (`measuredAreaSqMeters`, `depthMeters`, `measurementSource`). |

---

## 9. Frontend Maintenance Portal Integration

Upgraded [`apps/web/src/app/maintenance/page.tsx`](file:///c:/Users/Admin/Desktop/my-projects/CivicPulse-AI/apps/web/src/app/maintenance/page.tsx) with a dual-tab modal:
1. **Tab 1: Priority Decision Support:** 6-factor triage radar, explainability narrative, and administrative review actions (Accept, Override, Reject, Require Inspection).
2. **Tab 2: Repair Cost & Duration Estimation:**
   - Displays estimated cost (INR `₹`) and duration (hours and days).
   - Shows itemized materials, labor, equipment, and mobilization subtotal breakdown.
   - Shows active regional rate schedule ID and governance disclaimers.
   - **Interactive Measurement Input Box:** Allows inspectors to input physical area ($m^2$), cavity depth ($m$), and measurement source (`field_measurement`), recalculating estimates in real time.
   - **Expenditure Accounting Box:** Clearly isolates parametric estimates from actual recorded expenditures (`Pending Work Order Completion`).

---

## 10. Verification & Test Suite Summary

### 10.1 Backend Repair Estimation Suite (`test-repair-system.ts`)
- **Total Tests:** **53**
- **Passed:** **53** ($100\%$)
- **Failed:** **0**
- **Coverage:** Rate configuration, regional isolation, unit conversions, volume calculations, cost arithmetic, `MEASUREMENT_PENDING` missing measurement gate, duration breakdown, actual vs estimated separation, database persistence, and Express REST APIs.

### 10.2 Python ML Scaffolding Suite (`test_repair_pipeline.py`)
- **Total Tests:** **3**
- **Passed:** **3** ($100\%$)
- **Failed:** **0**
- **Coverage:** Regression readiness gate, threshold constants ($N_{\text{min}}=50$), circular proxy prohibition, and insufficient data reporting.

### 10.3 Full Regression Suite Verification
- `test-priority-system.ts`: **65 Passed, 0 Failed**
- `test-severity-system.ts`: **47 Passed, 0 Failed**
- `test-hotspot-system.ts`: **67 Passed, 0 Failed**
- `pytest services/ml/tests`: **39 Passed, 0 Failed** (all clustering, severity, priority, and repair tests)
- `services/api` build: `tsc` succeeded with **0 errors**.
- `apps/web` build: `next build` succeeded with **0 errors**.

---

## 11. Completion Boundary & Commitments

- **No Invented Government Rates:** Transparently disclosed that no official state SoR documents exist in the repository; parametric baselines are clearly flagged as provisional.
- **No Fabricated Physical Areas:** Pixel bounding boxes are strictly barred from being converted into physical square meters without field measurement.
- **No Automatic Dispatch:** Field maintenance crews are never dispatched autonomously.
- **No Synthetic Actual Expenditures:** Zero fake completed repair records were generated.
- **Phase 2 Artifacts Intact:** `production_model.pt`, `model.pth`, and MLflow model registry remain unmodified.
- **Original Data Preserved:** All 66 original incident records, GPS coordinates, and inspector verifications remain intact.
- **Boundary Honored:** Phase 3 Stage 4 is complete. Stage 6 (Incident Volume Forecasting) has **NOT** been started.

