"""Tests for health endpoints."""

import pytest
from fastapi.testclient import TestClient
from app.main import app


client = TestClient(app)


def test_health_endpoint():
    """Test the health endpoint returns healthy status."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True
    assert data["data"]["status"] == "healthy"
    assert data["data"]["service"] == "civicpulse-ml"


def test_readiness_endpoint():
    """Test the readiness endpoint checks ML dependencies."""
    response = client.get("/ready")
    assert response.status_code == 200
    data = response.json()
    assert data["data"]["service"] == "civicpulse-ml"
    assert "services" in data["data"]


def test_list_models():
    """Test the model listing endpoint."""
    response = client.get("/api/v1/ml/models")
    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True
    assert isinstance(data["data"], list)
    assert len(data["data"]) > 0


def test_get_model_not_found():
    """Test getting a non-existent model."""
    response = client.get("/api/v1/ml/models/nonexistent")
    assert response.status_code == 200
    data = response.json()
    assert data["success"] is False
