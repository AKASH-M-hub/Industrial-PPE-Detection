import io
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from src.api.main import app
from tests.test_detector import create_dummy_image_bytes

client = TestClient(app)


def test_health_endpoint():
    """Verify /health returns 200 with runtime diagnostics."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["unified_confidence_threshold"] == 0.45
    assert "version" in data


def test_detect_endpoint_success():
    """Verify /api/v1/detect processes image upload."""
    img_bytes = create_dummy_image_bytes()
    files = {"file": ("test.jpg", img_bytes, "image/jpeg")}

    response = client.post("/api/v1/detect", files=files)
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert "inference_time_ms" in data
    assert "image_metadata" in data
    assert data["image_metadata"]["width"] == 640


def test_detect_endpoint_rejects_empty_file():
    """Verify /api/v1/detect rejects empty file payloads."""
    files = {"file": ("test.jpg", b"", "image/jpeg")}
    response = client.post("/api/v1/detect", files=files)
    assert response.status_code == 400


def test_detect_endpoint_rejects_unsupported_format():
    """Verify /api/v1/detect rejects non-image extensions."""
    files = {"file": ("test.txt", b"some text", "text/plain")}
    response = client.post("/api/v1/detect", files=files)
    assert response.status_code == 400


def test_reason_endpoint_non_visual_question():
    """Verify /api/v1/reason routes non-visual questions without detector."""
    img_bytes = create_dummy_image_bytes()
    files = {"file": ("test.jpg", img_bytes, "image/jpeg")}
    data = {"question": "What is the capital of Canada?"}

    response = client.post("/api/v1/reason", files=files, data=data)
    assert response.status_code == 200
    res = response.json()
    assert res["detector_called"] is False
    assert res["intent_routing"]["needs_detection"] is False
    assert "answer" in res


def test_reason_endpoint_visual_question_guardrail():
    """Verify /api/v1/reason routes visual question through guardrail."""
    img_bytes = create_dummy_image_bytes()
    files = {"file": ("test.jpg", img_bytes, "image/jpeg")}
    data = {"question": "How many workers are wearing a helmet in this image?"}

    response = client.post("/api/v1/reason", files=files, data=data)
    assert response.status_code == 200
    res = response.json()
    assert res["detector_called"] is True
    assert res["guardrail_evaluated"] is True
    assert "answer" in res
