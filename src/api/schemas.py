from typing import Optional, Dict, Any
from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    """System health and runtime status."""
    status: str = "healthy"
    service: str
    version: str
    model_loaded: bool
    model_path: str
    device: str
    groq_configured: bool
    unified_confidence_threshold: float = Field(
        ...,
        description="Unified threshold (0.45) strictly audited across detection & guardrails"
    )


class ErrorDetail(BaseModel):
    """Standardized error response payload."""
    code: str
    message: str
    details: Optional[Any] = None
