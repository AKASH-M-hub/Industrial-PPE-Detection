import os
import base64
from typing import Optional
from fastapi import APIRouter, File, UploadFile, Form, HTTPException, status, Depends, Request
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
    request: Request,
    file: Optional[UploadFile] = File(None, description="Image file (JPG, PNG, WEBP)"),
    detector: RTDETRDetector = Depends(get_detector),
) -> DetectionResponse:
    """Detect workers, helmets, and safety vests in an uploaded image (supports multipart or JSON base64)."""
    content_type = request.headers.get("content-type", "").lower()
    image_bytes = None

    if "application/json" in content_type:
        try:
            body = await request.json()
            raw_b64 = body.get("image") or body.get("file") or ""
            if "," in raw_b64:
                raw_b64 = raw_b64.split(",", 1)[1]
            image_bytes = base64.b64decode(raw_b64)
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid JSON base64 image payload: {e}"
            )
    else:
        if file is None:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="No file provided."
            )
        validate_image_file(file)
        image_bytes = await file.read()

    if not image_bytes or len(image_bytes) == 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Empty image payload received."
        )
    if len(image_bytes) > MAX_FILE_SIZE:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail="Uploaded image exceeds 15 MB limit."
        )

    try:
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
    request: Request,
    file: Optional[UploadFile] = File(None, description="Image file"),
    question: Optional[str] = Form(None, description="Question about the image"),
    detector: RTDETRDetector = Depends(get_detector),
    engine: ReasoningEngine = Depends(get_reasoning_engine),
) -> ReasoningResponse:
    """Answer natural language questions about PPE compliance in an image (supports multipart or JSON base64)."""
    content_type = request.headers.get("content-type", "").lower()
    image_bytes = None
    q_str = question

    if "application/json" in content_type:
        try:
            body = await request.json()
            raw_b64 = body.get("image") or body.get("file") or ""
            if "," in raw_b64:
                raw_b64 = raw_b64.split(",", 1)[1]
            image_bytes = base64.b64decode(raw_b64)
            q_str = body.get("question") or q_str
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid JSON base64 payload: {e}"
            )
    else:
        if file is None:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="No file provided."
            )
        validate_image_file(file)
        image_bytes = await file.read()

    if not q_str or not q_str.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Question string cannot be empty."
        )

    if not image_bytes or len(image_bytes) == 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Empty image payload received."
        )

    try:
        response = engine.reason(
            question=q_str.strip(),
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


@router.get("/dashboard", tags=["Dashboard"])
async def serve_dashboard():
    from fastapi.responses import FileResponse, HTMLResponse
    root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    dashboard_path = os.path.join(root_dir, "dashboard.html")
    if os.path.exists(dashboard_path):
        return FileResponse(dashboard_path)
    return HTMLResponse("<h3>Dashboard not found. Open dashboard.html directly.</h3>")


@router.get("/memo", tags=["Documentation"])
async def serve_memo():
    from fastapi.responses import FileResponse, HTMLResponse
    root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    memo_path = os.path.join(root_dir, "docs", "submission_memo.html")
    if os.path.exists(memo_path):
        return FileResponse(memo_path)
    return HTMLResponse("<h3>Memo not found. Open docs/submission_memo.html directly.</h3>")

