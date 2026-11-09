# CivicPulse AI — Road Damage ML Inference Integration Guide

**Framework**: PyTorch `2.11.0+cu128` on NVIDIA GeForce RTX 5060 Ti  
**ML Server**: FastAPI (`services/ml`, port `8000`)  
**Backend API**: Node.js + Express (`services/api`, port `5000`)  
**Database**: PostgreSQL 16 + PostGIS (`civicpulse-db`, port `5432`)  
**Object Store**: MinIO S3 (`civicpulse-storage`, port `9000`)  
**Frontend**: Next.js 16 + TypeScript (`apps/web`, port `3000`)  

---

## 1. End-to-End Architectural Flow

```
[Mobile/Dashcam Camera]
        │
        ▼ (Multipart Photo Upload)
[Express Backend API:5000]
        │
        ├──▶ 1. Upload Original Raw Image ─────────▶ [MinIO S3:9000]
        │                                           (bucket: civicpulse-images)
        │
        ├──▶ 2. Forward Image for GPU Inference ───▶ [FastAPI ML Service:8000]
        │                                           │
        │                                           ├─▶ ResNet-18 (Classification: 0.66ms)
        │                                           └─▶ SSDLite320 (Localization: 10.01ms)
        │
        ├──▶ 3. Persist Records ───────────────────▶ [PostgreSQL 16 + PostGIS:5432]
        │                                           ├─▶ incidents
        │                                           ├─▶ incident_images
        │                                           ├─▶ ai_predictions
        │                                           └─▶ model_registry
        │
        ▼
[Next.js Dashboard:3000]
  - Visualizes road distress overlays and confidence scores
  - Supports Human-in-the-Loop inspector verification & dispatch
```

---

## 2. API Endpoint Specifications

### A. ML Service: Road Damage Inference
- **Method**: `POST`
- **Path**: `/api/v1/ml/predict/road-damage`
- **Content-Type**: `multipart/form-data`
- **Payload**:
  - `image`: binary JPEG, PNG, or WebP file (up to 15MB)
  - `confidence_threshold`: float (default `0.35`)
- **Sample Response**:
```json
{
  "success": true,
  "model": {
    "name": "road-damage-classifier-resnet18",
    "version": "1.0.0",
    "framework": "PyTorch 2.11.0+cu128",
    "device": "cuda:0 (NVIDIA GeForce RTX 5060 Ti)"
  },
  "predictions": [
    {
      "damage_code": "D00",
      "damage_name": "Longitudinal Crack",
      "confidence": 0.4281,
      "severity": "medium",
      "detected": true,
      "description": "Pavement joint separation along travel direction"
    },
    {
      "damage_code": "D10",
      "damage_name": "Transverse Crack",
      "confidence": 0.0120,
      "severity": "low",
      "detected": false,
      "description": "Thermal expansion fracture across lane"
    },
    {
      "damage_code": "D20",
      "damage_name": "Alligator Crack",
      "confidence": 0.1850,
      "severity": "high",
      "detected": false,
      "description": "Structural fatigue cracking indicating base failure"
    },
    {
      "damage_code": "D40",
      "damage_name": "Pothole",
      "confidence": 0.8440,
      "severity": "critical",
      "detected": true,
      "description": "Asphalt cavity creating immediate vehicular hazard"
    }
  ],
  "detections": [],
  "summary": {
    "has_damage": true,
    "primary_distress": "D40 (Pothole)",
    "overall_severity": "critical",
    "max_confidence": 0.8440
  },
  "metrics": {
    "image_width": 720,
    "image_height": 720,
    "classification_latency_ms": 0.65,
    "detection_latency_ms": 10.12,
    "total_latency_ms": 11.24,
    "detection_count": 0
  }
}
```

---

### B. Express API: Incident Reporting with AI Analysis
- **Method**: `POST`
- **Path**: `/api/v1/incidents/report`
- **Content-Type**: `multipart/form-data`
- **Payload**:
  - `image`: binary photo file
  - `roadName`: string
  - `address`: string
  - `latitude`: numeric string (e.g., `12.9715987`)
  - `longitude`: numeric string (e.g., `77.5945627`)
- **Action**:
  1. Stores image in MinIO S3 object store.
  2. Queries PyTorch ML service over internal network.
  3. Records incident with PostGIS coordinates, links `IncidentImage`, and stores full AI inference payload in `AiPrediction`.

---

### C. Express API: Inspector Verification (Human-in-the-Loop)
- **Method**: `POST`
- **Path**: `/api/v1/incidents/:id/verify`
- **Content-Type**: `application/json`
- **Payload**:
```json
{
  "verifiedClass": "pothole",
  "verifiedSeverity": "critical",
  "remarks": "Confirmed in field inspection. Dispatched repair crew."
}
```
- **Action**: Creates `IncidentVerification` audit record and transitions `Incident.status` from `submitted` to `verified`.

---

## 3. Performance & Latency Benchmarks

| Component | Target Latency | Measured GPU Latency | Throughput |
|---|---|---|---|
| **ResNet-18 Classifier** | < 25 ms | **0.66 ms** | ~1,515 img/sec |
| **SSDLite320 Detector** | < 50 ms | **10.01 ms** | ~100 img/sec |
| **End-to-End API Roundtrip** | < 200 ms | **86.43 ms** (inc. MinIO upload & DB write) | ~11 incidents/sec |
| **Memory Footprint** | < 6.0 GB | **1.58 GB peak VRAM** | 77% VRAM headroom remaining |

