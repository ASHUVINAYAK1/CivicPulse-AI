# CivicPulse AI

> **AI-Powered Road Infrastructure Monitoring & Predictive Maintenance Platform**

CivicPulse AI is an enterprise-grade platform transforming citizen road damage reports, ground imagery, geospatial data, and historical maintenance records into actionable municipal infrastructure intelligence.

---

## 🏛 System Architecture Overview

```
                                 [ CITIZENS & INSPECTORS ]
                                             │
                                             ▼
                             ┌───────────────────────────────┐
                             │    Next.js Web Application    │ (App Router, Tailwind CSS,
                             │       [apps/web :3000]        │  Lucide, Recharts, Leaflet)
                             └───────────────┬───────────────┘
                                             │ HTTP / REST
                                             ▼
                             ┌───────────────────────────────┐
                             │      Express Backend API      │ (Node.js, TypeScript, Helmet,
                             │     [services/api :5000]      │  CORS, Zod, pg Pool)
                             └───────┬───────────────┬───────┘
                                     │               │
                     ┌───────────────┘               └───────────────┐
                     ▼                                               ▼
     ┌───────────────────────────────┐               ┌───────────────────────────────┐
     │   PostgreSQL 16 + PostGIS     │               │     FastAPI Python ML API     │
     │     [infra/postgres :5432]    │               │      [services/ml :8000]      │
     ├───────────────────────────────┤               ├───────────────────────────────┤
     │ • Spatial indexing (GiST)     │               │ • Computer Vision (CNN/Trans) │
     │ • Incident & User records     │               │ • Classical ML (Severity/Pri) │
     │ • Maintenance work orders     │               │ • Time-Series (LSTM Forecast) │
     └───────────────┬───────────────┘               │ • K-Means & PCA Hotspots      │
                     │                               └───────────────┬───────────────┘
             ┌───────┴───────┐                                       │
             ▼               ▼                                       ▼
     ┌───────────────┐┌───────────────┐                      ┌───────────────┐
     │ Redis Queue   ││ MinIO Storage │                      │ MLflow Server │
     │ [infra :6379] ││ [infra :9000] │                      │    [:5001]    │
     └───────────────┘└───────────────┘                      └───────────────┘
```

---

## 📂 Monorepo Structure

```
civicpulse-ai/
├── apps/
│   └── web/                     # Next.js 16 frontend (TypeScript, Tailwind, Lucide, Recharts)
├── services/
│   ├── api/                     # Node.js + Express backend (TypeScript, Zod, pg pool)
│   └── ml/                      # Python + FastAPI ML inference & model registry service
├── packages/
│   ├── shared-types/            # Shared TypeScript types across frontend and backend
│   └── api-contracts/           # Shared REST API contract definitions & constants
├── infrastructure/
│   ├── postgres/                # PostgreSQL 16 + PostGIS initialization scripts (init.sql)
│   ├── redis/                   # Redis cache & background task configuration
│   ├── minio/                   # MinIO S3-compatible image storage configuration
│   └── monitoring/              # Observability and metrics configurations
├── datasets/
│   ├── raw/                     # Original road damage datasets (e.g. RDD2022)
│   ├── processed/               # Preprocessed image tensors and normalized tabular data
│   ├── annotations/             # Ground-truth annotations (bounding boxes, severity)
│   └── synthetic/               # Clearly-demarcated synthetic benchmark data
├── notebooks/
│   ├── eda/                     # Exploratory Data Analysis
│   ├── classical_ml/            # Logistic Regression, Random Forest, AdaBoost experiments
│   ├── clustering/              # K-Means from scratch and geospatial clustering
│   ├── computer_vision/         # CNN baseline and transfer learning (MobileNet, EfficientNet)
│   ├── deep_learning/           # Artificial Neural Network (ANN) tabular models
│   └── forecasting/             # LSTM / RNN time-series incident volume forecasting
├── docs/                        # Engineering blueprints, schemas, and specifications
│   ├── ARCHITECTURE.md          # In-depth architectural design
│   ├── LOCAL_SETUP.md           # Local environment setup and execution instructions
│   ├── DATABASE_SCHEMA.md       # Relational and spatial database schemas
│   └── API_SPECIFICATION.md     # Express API and FastAPI ML route specifications
├── scripts/
│   └── setup.sh                 # Environment setup and prerequisite validation script
├── docker-compose.yml           # Local infrastructure stack (PostGIS, Redis, MinIO)
├── .env.example                 # Template for local environment variables
├── .gitignore                   # Multi-stack Git ignore rules
└── README.md                    # Project documentation
```

---

## 🚀 Quick Start Guide

### Prerequisites
- **Node.js**: v22 LTS or higher (tested on Node v24.11 LTS)
- **Python**: 3.11 or 3.12 (with pip)
- **Docker & Docker Compose**: (Required for containerized PostGIS, Redis, and MinIO)

### 1. Environment Configuration
Copy `.env.example` to `.env`:
```bash
cp .env.example .env
```

### 2. Infrastructure Startup (via Docker Compose)
```bash
docker compose up -d db redis minio
```
Services available:
- **PostgreSQL 16 + PostGIS**: `127.0.0.1:5432` (`civicpulse`/`civicpulse_dev_password`)
- **Redis 7**: `127.0.0.1:6379`
- **MinIO S3 API**: `127.0.0.1:9000` | **MinIO Console**: `127.0.0.1:9001`

### 3. Start Backend API
```bash
cd services/api
npm install
npm run dev
```
- Base API URL: `http://localhost:5000/api/v1`
- Health Probe: `http://localhost:5000/health`
- Readiness Probe: `http://localhost:5000/ready`

### 4. Start Python ML Service
```bash
cd services/ml
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000
```
- Interactive Swagger UI: `http://localhost:8000/docs`
- Health Check: `http://localhost:8000/health`
- Readiness Check: `http://localhost:8000/ready`
- Model Registry: `http://localhost:8000/api/v1/ml/models`

### 5. Start Frontend Dashboard
```bash
cd apps/web
npm install
npm run dev
```
Open **`http://localhost:3000`** in your browser.

---

## 🧪 Verification & Testing

- **Backend Typecheck**:
  ```bash
  cd services/api && npm run typecheck
  ```
- **Backend Build**:
  ```bash
  cd services/api && npm run build
  ```
- **Frontend Build**:
  ```bash
  cd apps/web && npm run build
  ```
- **ML Service Test Suite**:
  ```bash
  cd services/ml && python -m pytest
  ```

---

## 🔐 Security & Engineering Standards

- **Environment Validation**: Strict runtime schema validation using Zod (Express) and Pydantic Settings (FastAPI).
- **Network Isolation**: All infrastructure containers bind explicitly to `127.0.0.1`.
- **Decoupled Architecture**: Stateless microservice communication over REST contracts.
- **Traceability**: All model inferences capture model version IDs, execution latencies, and human verification stamps.
- **Data Protection**: Strict `.gitignore` rules prevent accidental commits of secrets, datasets, or trained model artifacts.
