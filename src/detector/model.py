import io
import os
import time
from typing import List, Dict, Tuple, Optional
import cv2
import numpy as np
from PIL import Image
from loguru import logger

try:
    import spaces
except ImportError:
    class _MockSpaces:
        @staticmethod
        def GPU(fn=None, duration=60):
            if fn is not None:
                return fn
            def decorator(f):
                return f
            return decorator
    spaces = _MockSpaces()

from src.config import settings
from src.detector.schemas import (
    BoundingBox,
    DetectionItem,
    DetectionResponse,
    ImageMetadata,
)


class RTDETRDetector:
    """Inference wrapper for RT-DETR."""

    def __init__(self, model_path: Optional[str] = None):
        self.model_path = model_path or settings.MODEL_PATH
        self.device = settings.DEVICE
        self.classes = settings.TARGET_CLASSES
        self.model = None
        self._is_mock = False
        self._load_model()

    def _load_model(self) -> None:
        try:
            from ultralytics import RTDETR

            if not os.path.exists(self.model_path):
                try:
                    os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
                    hf_url = "https://huggingface.co/Akashhhhwqx/rtdetr-workplace-ppe-detection/resolve/main/rtdetr_ppe_best.pt"
                    logger.info(f"Downloading model weights from Hugging Face Model Hub: {hf_url}")
                    import urllib.request
                    urllib.request.urlretrieve(hf_url, self.model_path)
                    logger.info(f"Downloaded weights to {self.model_path}")
                except Exception as dl_err:
                    logger.warning(f"Could not download from Hugging Face Hub: {dl_err}")

            if os.path.exists(self.model_path):
                logger.info(f"Loading RT-DETR weights from {self.model_path}")
                self.model = RTDETR(self.model_path)
            else:
                logger.warning(f"Weights not found at '{self.model_path}', using fallback checkpoint.")
                self.model = RTDETR(settings.FALLBACK_PRETRAINED)

            logger.info("RT-DETR model loaded.")
        except Exception as exc:
            logger.error(f"Failed to load model: {exc}")
            self._is_mock = True

    def compute_image_quality(self, cv_image: np.ndarray) -> Tuple[float, bool]:
        """Compute variance of Laplacian for blur detection."""
        try:
            gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
            variance = float(cv2.Laplacian(gray, cv2.CV_64F).var())
            is_blurred = variance < settings.BLUR_LAPLACIAN_THRESHOLD
            return variance, is_blurred
        except Exception as e:
            logger.warning(f"Error computing blur score: {e}")
            return 100.0, False

    def predict(
        self,
        image_bytes: bytes,
        confidence_threshold: Optional[float] = None,
        iou_threshold: Optional[float] = None,
    ) -> DetectionResponse:
        conf_thresh = (
            confidence_threshold
            if confidence_threshold is not None
            else settings.CONFIDENCE_THRESHOLD
        )
        iou_thresh = (
            iou_threshold
            if iou_threshold is not None
            else settings.IOU_THRESHOLD
        )

        start_time = time.perf_counter()

        try:
            pil_img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
            cv_img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
        except Exception as e:
            raise ValueError(f"Invalid image format or corrupted payload: {e}")

        height, width = cv_img.shape[:2]
        blur_variance, is_blurred = self.compute_image_quality(cv_img)

        metadata = ImageMetadata(
            width=width,
            height=height,
            channels=cv_img.shape[2] if len(cv_img.shape) > 2 else 1,
            blur_laplacian_variance=round(blur_variance, 2),
            is_severely_blurred=is_blurred,
        )

        detections: List[DetectionItem] = []
        counts_by_class: Dict[str, int] = {c: 0 for c in self.classes}

        if self._is_mock or self.model is None:
            latency_ms = (time.perf_counter() - start_time) * 1000
            return DetectionResponse(
                status="mock_fallback",
                model_version="RT-DETR-MOCK",
                inference_time_ms=round(latency_ms, 2),
                total_detections=0,
                counts_by_class=counts_by_class,
                detections=[],
                image_metadata=metadata,
            )

        results = self.model.predict(
            source=pil_img,
            conf=conf_thresh,
            iou=iou_thresh,
            device=self.device,
            verbose=False,
            imgsz=settings.IMAGE_SIZE,
        )

        if results and len(results) > 0:
            res = results[0]
            boxes = res.boxes

            if boxes is not None and len(boxes) > 0:
                xyxy = boxes.xyxy.cpu().numpy()
                confs = boxes.conf.cpu().numpy()
                cls_ids = boxes.cls.cpu().numpy().astype(int)

                for i in range(len(confs)):
                    score = float(confs[i])
                    if score < conf_thresh:
                        continue

                    cid = int(cls_ids[i])
                    if hasattr(res, "names") and cid in res.names:
                        cname = str(res.names[cid]).lower().replace("_", "-")
                    elif cid < len(self.classes):
                        cname = self.classes[cid]
                    else:
                        cname = f"class_{cid}"

                    x1, y1, x2, y2 = xyxy[i]
                    area = float((x2 - x1) * (y2 - y1))
                    box_obj = BoundingBox(
                        x1=round(float(x1), 2),
                        y1=round(float(y1), 2),
                        x2=round(float(x2), 2),
                        y2=round(float(y2), 2),
                        norm_x1=round(float(x1 / width), 4),
                        norm_y1=round(float(y1 / height), 4),
                        norm_x2=round(float(x2 / width), 4),
                        norm_y2=round(float(y2 / height), 4),
                        area=round(area, 2),
                    )

                    detections.append(
                        DetectionItem(
                            class_id=cid,
                            class_name=cname,
                            confidence=round(score, 4),
                            box=box_obj,
                        )
                    )

                    if cname in counts_by_class:
                        counts_by_class[cname] += 1
                    else:
                        counts_by_class[cname] = 1

        latency_ms = (time.perf_counter() - start_time) * 1000

        return DetectionResponse(
            status="success",
            model_version="RT-DETR-PPE-v1",
            inference_time_ms=round(latency_ms, 2),
            total_detections=len(detections),
            counts_by_class=counts_by_class,
            detections=detections,
            image_metadata=metadata,
        )


# Singleton detector instance
_detector_instance: Optional[RTDETRDetector] = None


def get_detector() -> RTDETRDetector:
    """Provide singleton detector instance across requests."""
    global _detector_instance
    if _detector_instance is None:
        _detector_instance = RTDETRDetector()
    return _detector_instance
