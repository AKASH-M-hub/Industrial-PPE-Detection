from typing import Optional, List
from pydantic import BaseModel, Field
from loguru import logger
from src.config import settings
from src.detector.schemas import DetectionResponse, DetectionItem


class GuardrailResult(BaseModel):
    """Evaluation result from the Confidence & Quality Guardrail."""
    is_sufficient: bool = Field(
        ...,
        description="True if detection output and image quality allow confident reasoning"
    )
    failure_mode: Optional[str] = Field(
        None,
        description="Reason code: 'severe_blur', 'low_confidence', 'no_objects_found', 'extreme_distance_tiny_objects', 'none'"
    )
    message: str = Field(
        ...,
        description="Transparent statement on whether information is sufficient or why it cannot be answered confidently"
    )


class ConfidenceGuardrail:
    """Quality and confidence guardrail for visual inspection."""

    def __init__(self, confidence_threshold: float = settings.CONFIDENCE_THRESHOLD):
        self.min_confidence = confidence_threshold

    def evaluate(
        self,
        question: str,
        detection_res: DetectionResponse,
    ) -> GuardrailResult:
        meta = detection_res.image_metadata
        detections = detection_res.detections
        total = detection_res.total_detections

        # Image blur check
        if meta.is_severely_blurred:
            logger.warning(f"Guardrail triggered: blur variance={meta.blur_laplacian_variance}")
            return GuardrailResult(
                is_sufficient=False,
                failure_mode="severe_blur",
                message=(
                    f"Insufficient information: Image resolution or focus is too degraded "
                    f"(Laplacian sharpness variance: {meta.blur_laplacian_variance:.1f} < {settings.BLUR_LAPLACIAN_THRESHOLD}) "
                    f"to reliably verify PPE safety compliance."
                )
            )

        # Zero detections
        if total == 0:
            return GuardrailResult(
                is_sufficient=False,
                failure_mode="no_objects_found",
                message=(
                    f"Insufficient information: No workers or safety equipment could be detected "
                    f"with confidence above the {self.min_confidence:.2f} threshold."
                )
            )

        # Distant or tiny objects (< 0.05% of frame)
        total_img_area = meta.width * meta.height
        tiny_detections = [
            d for d in detections if (d.box.area or 0) / max(total_img_area, 1) < 0.0005
        ]
        if len(tiny_detections) == total and total > 0:
            return GuardrailResult(
                is_sufficient=False,
                failure_mode="extreme_distance_tiny_objects",
                message=(
                    f"Insufficient information: Detected subjects occupy less than 0.05% of the frame. "
                    f"Distinguishing equipment details is ambiguous at this distance."
                )
            )

        # Low confidence
        sub_threshold = [d for d in detections if d.confidence < self.min_confidence]
        if sub_threshold:
            return GuardrailResult(
                is_sufficient=False,
                failure_mode="low_confidence",
                message=(
                    f"Insufficient information: Detection confidence is below threshold ({self.min_confidence:.2f})."
                )
            )

        return GuardrailResult(
            is_sufficient=True,
            failure_mode="none",
            message="Detection confidence and image quality are sufficient."
        )
