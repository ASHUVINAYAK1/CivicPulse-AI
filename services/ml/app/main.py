"""
CivicPulse AI - ML Service Entry Point

Provides health checks, readiness probes, and ML inference endpoints.
"""

from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import AsyncGenerator

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config import get_settings
from app.api.health import router as health_router
from app.api.models import router as models_router


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Application lifespan manager."""
    settings = get_settings()
    print(f"""
    ╔══════════════════════════════════════════╗
    ║    CivicPulse AI - ML Service            ║
    ╠══════════════════════════════════════════╣
    ║  Status:  Running                        ║
    ║  Port:    {str(settings.port):<29}║
    ║  Docs:    http://localhost:{settings.port}/docs      ║
    ╚══════════════════════════════════════════╝
    """)
    yield
    print("ML Service shutting down...")


settings = get_settings()

app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    description="AI/ML inference and training service for road infrastructure analysis",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:5000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register routers
app.include_router(health_router, tags=["Health"])
app.include_router(models_router, prefix="/api/v1", tags=["Models"])
