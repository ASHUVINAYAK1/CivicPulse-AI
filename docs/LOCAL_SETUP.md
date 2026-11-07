# CivicPulse AI — Local Development & Setup Guide

This guide details setting up and running CivicPulse AI on Windows with or without WSL2 / Docker.

---

## 1. Prerequisites Check

Verify required toolchains:

```powershell
node --version       # Expected: v22+ (tested v24.11 LTS)
npm --version        # Expected: v10+ (tested v11.6)
python --version     # Expected: 3.11+ (tested 3.11.9)
git --version        # Expected: 2.x
```

### Infrastructure Containers (Docker)
To run PostgreSQL, PostGIS, Redis, and MinIO:
- Install **Docker Desktop for Windows** with WSL2 backend enabled.
- Alternatively, if Docker is not installed natively, each service can be installed locally or run in a cloud database.

---

## 2. Environment Variables

Generate `.env`:
```powershell
Copy-Item .env.example .env
```
Default ports:
- Frontend: `3000`
- Express API: `5000`
- FastAPI ML: `8000`
- PostgreSQL / PostGIS: `5432`
- Redis: `6379`
- MinIO S3 API: `9000` (Console: `9001`)
- MLflow: `5001`

---

## 3. Launching Services

### Terminal 1: Infrastructure
```bash
docker compose up -d db redis minio
```

### Terminal 2: Node.js Express API
```bash
cd services/api
npm install
npm run dev
```

### Terminal 3: FastAPI ML Service
```bash
cd services/ml
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000
```

### Terminal 4: Next.js Frontend
```bash
cd apps/web
npm install
npm run dev
```

---

## 4. Health Check Verification

- Express API Health: `http://localhost:5000/health`
- Express API Readiness: `http://localhost:5000/ready`
- FastAPI ML Health: `http://localhost:8000/health`
- FastAPI ML Readiness: `http://localhost:8000/ready`
- FastAPI Swagger UI: `http://localhost:8000/docs`
- Next.js Web Console: `http://localhost:3000`
