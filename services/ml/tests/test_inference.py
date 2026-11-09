"""
CivicPulse AI — ML Inference API Integration Tests
Tests real image inference on GPU, schema validation, and error boundaries.
"""

from pathlib import Path
import io
import pytest
from fastapi.testclient import TestClient
from PIL import Image

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.main import app

client = TestClient(app)


def test_health_check():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    status_val = data.get("status") or data.get("data", {}).get("status")
    assert status_val in ["healthy", "degraded", "ok"]


def test_predict_real_image():
    # Pick a real road damage image from the test set
    test_img_dir = Path("datasets/raw/RDD2022_India/test/images")
    img_files = list(test_img_dir.glob("*.jpg"))
    if not img_files:
        pytest.skip("Test images not available")

    test_file = img_files[0]
    with open(test_file, "rb") as f:
        file_bytes = f.read()

    response = client.post(
        "/api/v1/ml/predict/road-damage",
        files={"image": (test_file.name, file_bytes, "image/jpeg")},
        data={"confidence_threshold": "0.35"},
    )

    assert response.status_code == 200
    data = response.json()

    assert data["success"] is True
    assert "road-damage-classifier-resnet18" in data["model"]["name"]
    assert len(data["predictions"]) == 4

    pred_codes = [p["damage_code"] for p in data["predictions"]]
    assert "D00" in pred_codes
    assert "D10" in pred_codes
    assert "D20" in pred_codes
    assert "D40" in pred_codes

    assert "total_latency_ms" in data["metrics"]
    assert data["metrics"]["total_latency_ms"] > 0
    print(f"\nInference verified on {test_file.name}: latency={data['metrics']['total_latency_ms']}ms, primary={data['summary']['primary_distress']}")


def test_predict_synthetic_fallback():
    # Create an in-memory test image
    img = Image.new("RGB", (640, 480), color=(100, 100, 100))
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    buf.seek(0)

    response = client.post(
        "/api/v1/predict/road-damage",
        files={"image": ("synthetic.jpg", buf.getvalue(), "image/jpeg")},
        data={"confidence_threshold": "0.40"},
    )

    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True
    assert isinstance(data["detections"], list)


def test_predict_invalid_mime():
    response = client.post(
        "/api/v1/ml/predict/road-damage",
        files={"image": ("test.txt", b"not an image", "text/plain")},
    )
    assert response.status_code == 400
    assert "Unsupported image type" in response.json()["detail"]


def test_predict_corrupt_image():
    response = client.post(
        "/api/v1/ml/predict/road-damage",
        files={"image": ("corrupt.jpg", b"bad image data here", "image/jpeg")},
    )
    assert response.status_code == 422
    assert "Corrupted or unreadable image" in response.json()["detail"]
