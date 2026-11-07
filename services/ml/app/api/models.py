"""
Model registry endpoints.

Provides information about available ML models.
"""

from datetime import datetime, timezone
from typing import Optional

from fastapi import APIRouter
from pydantic import BaseModel

router = APIRouter()


class ModelInfo(BaseModel):
    """Model information schema."""
    name: str
    model_type: str
    version: str
    status: str
    description: str
    framework: Optional[str] = None


# Placeholder model registry — will be replaced with database-backed registry
PLACEHOLDER_MODELS: list[ModelInfo] = [
    ModelInfo(
        name="road-damage-classifier",
        model_type="image_classification",
        version="0.0.0",
        status="not_trained",
        description="CNN-based road damage classifier (D00, D10, D20, D40)",
        framework="pytorch",
    ),
    ModelInfo(
        name="severity-predictor",
        model_type="classification",
        version="0.0.0",
        status="not_trained",
        description="Structured ML model for damage severity prediction",
        framework="scikit-learn",
    ),
    ModelInfo(
        name="maintenance-priority",
        model_type="classification",
        version="0.0.0",
        status="not_trained",
        description="Random Forest / Gradient Boosting priority predictor",
        framework="scikit-learn",
    ),
    ModelInfo(
        name="incident-forecaster",
        model_type="time_series",
        version="0.0.0",
        status="not_trained",
        description="LSTM-based incident volume forecaster",
        framework="pytorch",
    ),
]


@router.get("/ml/models")
async def list_models() -> dict:
    """List all registered ML models."""
    return {
        "success": True,
        "data": [m.model_dump() for m in PLACEHOLDER_MODELS],
        "meta": {
            "total": len(PLACEHOLDER_MODELS),
            "note": "These are placeholder entries. Models will be populated after training.",
        },
    }


@router.get("/ml/models/{model_name}")
async def get_model(model_name: str) -> dict:
    """Get information about a specific model."""
    model = next((m for m in PLACEHOLDER_MODELS if m.name == model_name), None)
    if model is None:
        return {
            "success": False,
            "error": {"code": "MODEL_NOT_FOUND", "message": f"Model '{model_name}' not found"},
        }
    return {
        "success": True,
        "data": model.model_dump(),
    }
