# CivicPulse AI — System Architecture & Engineering Blueprint

## 1. Architectural Principles

1. **Decoupled Workloads**:
   - Web application and transactional business logic reside in Node.js / Express.
   - Machine learning inference, preprocessing, and model training pipelines reside in Python / FastAPI.
   - User interfaces reside in Next.js (React 19, App Router).
2. **Local-First & Production-Ready**:
   - Fully runnable locally via Docker Compose and native dev servers.
   - Clean microservice boundary allowing production Kubernetes or container deployment without rewriting logic.
3. **Traceable ML Lifecycle**:
   - Model versions, inference confidence, execution time, and inspector validation stamps are persistently recorded in PostgreSQL.
4. **Spatial-First Relational Model**:
   - Uses PostGIS spatial types (`GEOGRAPHY(Point, 4326)`) and GiST indexing for fast nearby incident search and hotspot radius queries.

---

## 2. Component Directory Overview

- **`apps/web`**: Next.js 16 App Router application. Provides responsive interfaces for Citizens, Field Inspectors, and Municipal Administrators.
- **`services/api`**: Express TypeScript service handling Authentication (RBAC), Incidents, File Uploads, Maintenance Work Orders, and Analytics.
- **`services/ml`**: FastAPI service handling Computer Vision (damage classification), Classical ML (severity/priority/cost estimation), Geospatial Clustering (K-Means/PCA), and Time-Series Forecasting (LSTM).
- **`packages/shared-types`**: Shared TypeScript definitions (`UserRole`, `DamageType`, `Severity`, `IncidentStatus`, `ApiResponse`).
- **`packages/api-contracts`**: API payload definitions and constants.
- **`infrastructure/`**: Declarative database schemas, Redis configs, and MinIO storage initialization.
- **`datasets/`**: Normalized directory structure for raw, processed, annotated, and synthetic data (git-ignored for privacy/storage).
- **`notebooks/`**: Six specialized research tracks: EDA, Classical ML, Clustering, Computer Vision, Deep Learning, and Forecasting.
