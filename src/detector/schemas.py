from typing import List, Dict, Optional
from pydantic import BaseModel, Field


class BoundingBox(BaseModel):
    """Bounding box coordinates in absolute pixels and normalized format."""
    x1: float = Field(..., description="Top-left X coordinate in pixels")
    y1: float = Field(..., description="Top-left Y coordinate in pixels")
    x2: float = Field(..., description="Bottom-right X coordinate in pixels")
    y2: float = Field(..., description="Bottom-right Y coordinate in pixels")
    norm_x1: Optional[float] = Field(None, description="Normalized top-left X [0.0 - 1.0]")
    norm_y1: Optional[float] = Field(None, description="Normalized top-left Y [0.0 - 1.0]")
    norm_x2: Optional[float] = Field(None, description="Normalized bottom-right X [0.0 - 1.0]")
    norm_y2: Optional[float] = Field(None, description="Normalized bottom-right Y [0.0 - 1.0]")
    area: Optional[float] = Field(None, description="Area in pixel square units")


class DetectionItem(BaseModel):
    """Individual object detection item."""
    class_id: int = Field(..., description="Integer class identifier")
    class_name: str = Field(..., description="Readable class name (e.g. hard-hat, no-helmet)")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence score [0.0 - 1.0]")
    box: BoundingBox = Field(..., description="Bounding box coordinates")


class ImageMetadata(BaseModel):
    """Image physical dimensions and pre-computed visual quality metrics."""
    width: int
    height: int
    channels: int = 3
    blur_laplacian_variance: float = Field(..., description="Variance of Laplacian indicating image focus/blur")
    is_severely_blurred: bool = Field(False, description="True if blur variance is below strict threshold")


class DetectionResponse(BaseModel):
    """Part A API response payload."""
    status: str = "success"
    model_version: str = "RT-DETR-PPE-v1"
    inference_time_ms: float
    total_detections: int
    counts_by_class: Dict[str, int]
    detections: List[DetectionItem]
    image_metadata: ImageMetadata
