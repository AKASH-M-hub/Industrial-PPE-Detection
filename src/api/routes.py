import os
from fastapi import APIRouter, File, UploadFile, Form, HTTPException, status, Depends
from loguru import logger

from src.config import settings
from src.detector.model import get_detector, RTDETRDetector
from src.detector.schemas import DetectionResponse
from src.reasoning.engine import get_reasoning_engine, ReasoningEngine, ReasoningResponse
from src.api.schemas import HealthResponse

router = APIRouter()

# Max image upload limit: 15 MB
MAX_FILE_SIZE = 15 * 1024 * 1024
ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def validate_image_file(file: UploadFile) -> None:
    """Validate uploaded file type and non-empty payload."""
    ext = os.path.splitext(file.filename or "")[1].lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported file extension '{ext}'. Allowed extensions: {list(ALLOWED_EXTENSIONS)}"
        )


@router.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check(
    detector: RTDETRDetector = Depends(get_detector),
) -> HealthResponse:
    """Health status and runtime configuration."""
    model_loaded = detector.model is not None and not detector._is_mock
    groq_ok = bool(settings.GROQ_API_KEY and not settings.GROQ_API_KEY.startswith("gsk_your_groq"))

    return HealthResponse(
        status="healthy",
        service=settings.PROJECT_NAME,
        version=settings.VERSION,
        model_loaded=model_loaded,
        model_path=settings.MODEL_PATH,
        device=settings.DEVICE,
        groq_configured=groq_ok,
        unified_confidence_threshold=settings.CONFIDENCE_THRESHOLD,
    )


@router.post(
    "/detect",
    response_model=DetectionResponse,
    status_code=status.HTTP_200_OK,
    summary="PPE object detection",
    tags=["Detection"],
)
async def detect_objects(
    file: UploadFile = File(..., description="Image file (JPG, PNG, WEBP)"),
    detector: RTDETRDetector = Depends(get_detector),
) -> DetectionResponse:
    """Detect workers, helmets, and safety vests in an uploaded image."""
    validate_image_file(file)

    try:
        image_bytes = await file.read()
        if len(image_bytes) == 0:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Empty image payload received."
            )
        if len(image_bytes) > MAX_FILE_SIZE:
            raise HTTPException(
                status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                detail="Uploaded image exceeds 15 MB limit."
            )

        result = detector.predict(image_bytes)
        return result
    except ValueError as val_err:
        logger.error(f"Image decode error: {val_err}")
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(val_err)
        )
    except HTTPException:
        raise
    except Exception as exc:
        logger.error(f"Detection inference failure: {exc}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Inference pipeline failure: {str(exc)}"
        )


@router.post(
    "/reason",
    response_model=ReasoningResponse,
    status_code=status.HTTP_200_OK,
    summary="Natural language query on image",
    tags=["Reasoning"],
)
async def reason_about_image(
    file: UploadFile = File(..., description="Image file"),
    question: str = Form(..., description="Question about the image"),
    detector: RTDETRDetector = Depends(get_detector),
    engine: ReasoningEngine = Depends(get_reasoning_engine),
) -> ReasoningResponse:
    """Answer natural language questions about PPE compliance in an image."""
    validate_image_file(file)

    if not question or not question.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Question string cannot be empty."
        )

    try:
        image_bytes = await file.read()
        if len(image_bytes) == 0:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Empty image payload received."
            )

        response = engine.reason(
            question=question.strip(),
            image_bytes=image_bytes,
            detector=detector,
        )
        return response
    except ValueError as val_err:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(val_err)
        )
    except HTTPException:
        raise
    except Exception as exc:
        logger.error(f"Reasoning pipeline failure: {exc}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Reasoning layer failure: {str(exc)}"
        )
