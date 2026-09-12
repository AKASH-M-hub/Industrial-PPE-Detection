import io
import pytest
import numpy as np
from PIL import Image

from src.detector.model import RTDETRDetector
from src.detector.schemas import DetectionResponse, BoundingBox
from src.config import settings


def create_dummy_image_bytes(width: int = 640, height: int = 640, color=(128, 128, 128)) -> bytes:
    """Generate a clean in-memory test image."""
    img = Image.new("RGB", (width, height), color=color)
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return buf.getvalue()


def test_detector_initialization():
    """Verify detector initialization and fallback handling."""
    detector = RTDETRDetector()
    assert detector is not None
    assert detector.classes == settings.TARGET_CLASSES


def test_detector_prediction_schema():
    """Verify prediction returns valid DetectionResponse schema."""
    detector = RTDETRDetector()
    img_bytes = create_dummy_image_bytes()
    response = detector.predict(img_bytes)

    assert isinstance(response, DetectionResponse)
    assert response.image_metadata.width == 640
    assert response.image_metadata.height == 640
    assert "person" in response.counts_by_class
    assert "hard-hat" in response.counts_by_class
    assert response.inference_time_ms >= 0.0


def test_detector_quality_blur_computation():
    """Verify Laplacian blur variance computation."""
    detector = RTDETRDetector()
    # Uniform gray has 0 variance (extremely blurry)
    blank_cv = np.zeros((300, 300, 3), dtype=np.uint8)
    var, is_blur = detector.compute_image_quality(blank_cv)
    assert var < settings.BLUR_LAPLACIAN_THRESHOLD
    assert is_blur is True


def test_corrupted_image_raises_value_error():
    """Verify detector rejects malformed payloads."""
    detector = RTDETRDetector()
    with pytest.raises(ValueError):
        detector.predict(b"corrupted_non_image_bytes")
