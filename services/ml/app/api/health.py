"""
Health and readiness endpoints for the ML service.
"""

from datetime import datetime, timezone

from fastapi import APIRouter

from app.config import get_settings

router = APIRouter()


@router.get("/health")
async def health_check() -> dict:
    """Basic health check — confirms the ML service is running."""
    settings = get_settings()
    return {
        "success": True,
        "data": {
            "status": "healthy",
            "service": "civicpulse-ml",
            "version": settings.app_version,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        },
    }


@router.get("/ready")
async def readiness_check() -> dict:
    """
    Readiness check — verifies core ML dependencies are importable
    and service is ready to accept inference requests.
    """
    settings = get_settings()
    services: dict[str, dict] = {}

    # Check numpy
    try:
        import numpy as np
        services["numpy"] = {"status": "up", "version": np.__version__}
    except ImportError:
        services["numpy"] = {"status": "down"}

    # Check scikit-learn
    try:
        import sklearn
        services["scikit_learn"] = {"status": "up", "version": sklearn.__version__}
    except ImportError:
        services["scikit_learn"] = {"status": "down"}

    # Check OpenCV
    try:
        import cv2
        services["opencv"] = {"status": "up", "version": cv2.__version__}
    except ImportError:
        services["opencv"] = {"status": "down"}

    # Check Pillow
    try:
        import PIL
        services["pillow"] = {"status": "up", "version": PIL.__version__}
    except ImportError:
        services["pillow"] = {"status": "down"}

    all_up = all(s["status"] == "up" for s in services.values())

    return {
        "success": all_up,
        "data": {
            "status": "ready" if all_up else "degraded",
            "service": "civicpulse-ml",
            "version": settings.app_version,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "services": services,
        },
    }
