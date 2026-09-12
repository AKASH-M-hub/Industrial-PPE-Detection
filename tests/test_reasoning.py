import pytest
from src.reasoning.router import IntentRouter
from src.reasoning.guardrail import ConfidenceGuardrail
from src.reasoning.engine import ReasoningEngine
from src.detector.schemas import DetectionResponse, ImageMetadata, DetectionItem, BoundingBox
from src.detector.model import RTDETRDetector
from tests.test_detector import create_dummy_image_bytes


def test_intent_router_visual_detection():
    """Verify router identifies image-dependent visual questions."""
    router = IntentRouter()

    visual_queries = [
        "How many people are in this image?",
        "Is anyone not wearing a helmet?",
        "What is the most common object here?",
        "Count the workers wearing safety vests",
        "Are there any safety violations visible?"
    ]

    for q in visual_queries:
        decision = router.route(q)
        assert decision.needs_detection is True, f"Failed on query: {q}"
        assert decision.intent_category == "visual_inspection"


def test_intent_router_general_knowledge():
    """Verify router skips detector for non-visual / irrelevant queries."""
    router = IntentRouter()

    non_visual_queries = [
        "What is the capital of France?",
        "Who wrote Hamlet?",
        "What is the weather outside?",
        "Tell me a joke"
    ]

    for q in non_visual_queries:
        decision = router.route(q)
        assert decision.needs_detection is False, f"Failed on query: {q}"
        assert decision.intent_category == "general_knowledge"


def test_guardrail_triggers_on_severe_blur():
    """Verify guardrail rejects severely blurred images."""
    guardrail = ConfidenceGuardrail(confidence_threshold=0.45)

    meta = ImageMetadata(
        width=640,
        height=640,
        channels=3,
        blur_laplacian_variance=12.5,  # Below 60.0 threshold
        is_severely_blurred=True,
    )
    det_res = DetectionResponse(
        status="success",
        model_version="test",
        inference_time_ms=10.0,
        total_detections=1,
        counts_by_class={"person": 1},
        detections=[
            DetectionItem(
                class_id=0,
                class_name="person",
                confidence=0.88,
                box=BoundingBox(x1=10, y1=10, x2=100, y2=100, area=8100),
            )
        ],
        image_metadata=meta,
    )

    result = guardrail.evaluate("Is anyone wearing a helmet?", det_res)
    assert result.is_sufficient is False
    assert result.failure_mode == "severe_blur"
    assert "Insufficient information" in result.message


def test_guardrail_triggers_on_zero_detections():
    """Verify guardrail rejects empty detections on visual queries."""
    guardrail = ConfidenceGuardrail(confidence_threshold=0.45)

    meta = ImageMetadata(
        width=640,
        height=640,
        channels=3,
        blur_laplacian_variance=150.0,
        is_severely_blurred=False,
    )
    det_res = DetectionResponse(
        status="success",
        model_version="test",
        inference_time_ms=5.0,
        total_detections=0,
        counts_by_class={},
        detections=[],
        image_metadata=meta,
    )

    result = guardrail.evaluate("How many workers are present?", det_res)
    assert result.is_sufficient is False
    assert result.failure_mode == "no_objects_found"
    assert "Insufficient information" in result.message


def test_guardrail_passes_on_high_confidence_detections():
    """Verify guardrail approves high confidence sharp detections."""
    guardrail = ConfidenceGuardrail(confidence_threshold=0.45)

    meta = ImageMetadata(
        width=640,
        height=640,
        channels=3,
        blur_laplacian_variance=250.0,
        is_severely_blurred=False,
    )
    det_res = DetectionResponse(
        status="success",
        model_version="test",
        inference_time_ms=12.0,
        total_detections=2,
        counts_by_class={"person": 1, "hard-hat": 1},
        detections=[
            DetectionItem(
                class_id=0,
                class_name="person",
                confidence=0.85,
                box=BoundingBox(x1=50, y1=50, x2=300, y2=500, area=112500),
            ),
            DetectionItem(
                class_id=1,
                class_name="hard-hat",
                confidence=0.91,
                box=BoundingBox(x1=100, y1=60, x2=200, y2=140, area=8000),
            ),
        ],
        image_metadata=meta,
    )

    result = guardrail.evaluate("Is the worker wearing a hard-hat?", det_res)
    assert result.is_sufficient is True
    assert result.failure_mode == "none"
