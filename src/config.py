import os
from typing import List
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application configuration loaded from environment or .env file."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )

    # API settings
    PROJECT_NAME: str = "PPE Safety Vision & Reasoning API"
    VERSION: str = "1.0.0"
    API_V1_STR: str = "/api/v1"
    DEBUG: bool = False
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    LOG_LEVEL: str = "INFO"

    # Detector settings
    MODEL_PATH: str = "weights/rtdetr_ppe_best.pt"
    FALLBACK_PRETRAINED: str = "rtdetr-l.pt"
    DEVICE: str = "cpu"
    IMAGE_SIZE: int = 640

    CONFIDENCE_THRESHOLD: float = Field(
        default=0.45,
        description="Confidence threshold for detections"
    )
    IOU_THRESHOLD: float = Field(
        default=0.50,
        description="IoU threshold for NMS"
    )

    # Class definitions
    TARGET_CLASSES: List[str] = [
        "hat",
        "person",
        "vest"
    ]

    # Reasoning settings
    GROQ_API_KEY: str = Field(default="", description="Groq API key")
    GROQ_MODEL: str = "llama-3.3-70b-versatile"
    LLM_TEMPERATURE: float = 0.1
    LLM_MAX_TOKENS: int = 512

    # Image quality check
    BLUR_LAPLACIAN_THRESHOLD: float = 60.0


settings = Settings()
