#!/bin/bash
# ===========================================
# CivicPulse AI - Local Development Setup
# ===========================================

set -e

echo "🚀 Setting up CivicPulse AI development environment..."

# Check prerequisites
echo "📋 Checking prerequisites..."
command -v node >/dev/null 2>&1 || { echo "❌ Node.js is required but not installed."; exit 1; }
command -v npm >/dev/null 2>&1 || { echo "❌ npm is required but not installed."; exit 1; }
command -v python3 >/dev/null 2>&1 || command -v python >/dev/null 2>&1 || { echo "❌ Python is required but not installed."; exit 1; }
command -v docker >/dev/null 2>&1 || echo "⚠️  Docker is not installed. Infrastructure services will need to be started manually."

echo "✅ Prerequisites verified."

# Create .env from example if it doesn't exist
if [ ! -f .env ]; then
  echo "📄 Creating .env from .env.example..."
  cp .env.example .env
  echo "✅ .env created. Review and update values as needed."
else
  echo "📄 .env already exists, skipping."
fi

# Install frontend dependencies
echo "📦 Installing frontend dependencies..."
cd apps/web && npm install && cd ../..

# Install backend dependencies
echo "📦 Installing backend dependencies..."
cd services/api && npm install && cd ../..

# Install ML service dependencies
echo "📦 Installing ML service dependencies..."
cd services/ml && pip install -r requirements.txt && cd ../..

echo ""
echo "✅ Setup complete!"
echo ""
echo "To start infrastructure services:"
echo "  docker compose up -d db redis minio"
echo ""
echo "To start the application:"
echo "  Frontend:  cd apps/web && npm run dev"
echo "  Backend:   cd services/api && npm run dev"
echo "  ML:        cd services/ml && uvicorn app.main:app --reload --port 8000"
