# API Specification & Usage Guide

## Base URL
- Local: `http://localhost:8000`
- API Version Prefix: `/api/v1` (routes are also mirrored at root `/` for convenience)

---

## 1. GET `/health`
Returns system status, model load state, active device, and the unified confidence threshold (`0.45`).

### Request
```bash
curl -X GET "http://localhost:8000/health"
```

### Response (200 OK)
```json
{
  "status": "healthy",
  "service": "PPE Constrained Object Detection & Reasoning API",
  "version": "1.0.0",
  "model_loaded": true,
  "model_path": "weights/rtdetr_ppe_best.pt",
  "device": "cpu",
  "groq_configured": true,
  "unified_confidence_threshold": 0.45
}
```

---

## 2. POST `/api/v1/detect` (Part A: Object Detection)
Accepts an image file and executes fine-tuned RT-DETR object detection. Returns bounding boxes, class labels, and confidence scores (strictly filtered by `confidence >= 0.45`).

### Request
```bash
curl -X POST "http://localhost:8000/api/v1/detect" \
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@construction_site.jpg;type=image/jpeg"
```

### Python Requests Example
```python
import requests

url = "http://localhost:8000/api/v1/detect"
with open("construction_site.jpg", "rb") as f:
    files = {"file": ("construction_site.jpg", f, "image/jpeg")}
    response = requests.post(url, files=files)

print(response.json())
```

### Response (200 OK)
```json
{
  "status": "success",
  "model_version": "RT-DETR-PPE-v1",
  "inference_time_ms": 28.45,
  "total_detections": 3,
  "counts_by_class": {
    "person": 2,
    "hard-hat": 1,
    "no-helmet": 1,
    "safety-vest": 2,
    "no-vest": 0
  },
  "detections": [
    {
      "class_id": 0,
      "class_name": "person",
      "confidence": 0.8921,
      "box": {
        "x1": 120.4,
        "y1": 85.2,
        "x2": 310.8,
        "y2": 540.0,
        "norm_x1": 0.1881,
        "norm_y1": 0.1331,
        "norm_x2": 0.4856,
        "norm_y2": 0.8438,
        "area": 86600.64
      }
    },
    {
      "class_id": 1,
      "class_name": "hard-hat",
      "confidence": 0.8432,
      "box": {
        "x1": 175.2,
        "y1": 86.0,
        "x2": 255.4,
        "y2": 155.0,
        "norm_x1": 0.2738,
        "norm_y1": 0.1344,
        "norm_x2": 0.3991,
        "norm_y2": 0.2422,
        "area": 5533.8
      }
    },
    {
      "class_id": 2,
      "class_name": "no-helmet",
      "confidence": 0.7615,
      "box": {
        "x1": 420.0,
        "y1": 110.5,
        "x2": 490.2,
        "y2": 180.0,
        "norm_x1": 0.6563,
        "norm_y1": 0.1727,
        "norm_x2": 0.7659,
        "norm_y2": 0.2813,
        "area": 4878.9
      }
    }
  ],
  "image_metadata": {
    "width": 640,
    "height": 640,
    "channels": 3,
    "blur_laplacian_variance": 184.6,
    "is_severely_blurred": false
  }
}
```

---

## 3. POST `/api/v1/reason` (Part B: Minimal Reasoning Layer)
Accepts an image and a natural language question. Hand-written pipeline:
1. **Intent Router**: Evaluates whether detection is required.
2. **RT-DETR**: Runs detection if relevant.
3. **Confidence Guardrail**: Rejects ambiguous, low-confidence (< 0.45), or blurred images with explicit notice.
4. **Structured Reasoning**: Groq LLM synthesizes an auditable answer.

### Example A: Relevant Safety Question (Successful Reasoning)
#### Request
```bash
curl -X POST "http://localhost:8000/api/v1/reason" \
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@site_inspection.jpg;type=image/jpeg" \
  -F "question=Is anyone not wearing a helmet?"
```

#### Response (200 OK)
```json
{
  "status": "success",
  "question": "Is anyone not wearing a helmet?",
  "intent_routing": {
    "needs_detection": true,
    "intent_category": "visual_inspection",
    "confidence": 0.98,
    "reasoning": "Query matches visual inspection pattern: '\\bhelmet\\b'."
  },
  "detector_called": true,
  "guardrail_evaluated": true,
  "guardrail_result": {
    "is_sufficient": true,
    "failure_mode": "none",
    "message": "Sufficient information: Image quality and detection confidence meet strict standards."
  },
  "answer": "Yes, there is a safety violation. Two workers are detected in the image: one worker is compliant and wearing a hard-hat (confidence: 0.84), but a second worker is detected without a helmet ('no-helmet', confidence: 0.76). Immediate PPE compliance intervention is recommended.",
  "model_used": "llama-3.3-70b-versatile",
  "total_latency_ms": 312.4
}
```

---

### Example B: Non-Visual Question (Detector Skipped by Intent Router)
#### Request
```bash
curl -X POST "http://localhost:8000/api/v1/reason" \
  -F "file=@site_inspection.jpg;type=image/jpeg" \
  -F "question=What is the standard OSHA minimum penalty for a willful violation?"
```

#### Response (200 OK)
```json
{
  "status": "success",
  "question": "What is the standard OSHA minimum penalty for a willful violation?",
  "intent_routing": {
    "needs_detection": false,
    "intent_category": "general_knowledge",
    "confidence": 0.95,
    "reasoning": "Query is non-visual general knowledge."
  },
  "detector_called": false,
  "guardrail_evaluated": false,
  "guardrail_result": null,
  "answer": "Under OSHA guidelines, the statutory minimum penalty for a willful violation is currently $11,162, with maximum penalties exceeding $161,000 depending on inflation adjustments and the gravity of the hazard.",
  "model_used": "llama-3.3-70b-versatile",
  "total_latency_ms": 185.1
}
```

---

### Example C: Degraded / Ambiguous Image (Guardrail Triggered)
#### Request
```bash
curl -X POST "http://localhost:8000/api/v1/reason" \
  -F "file=@heavy_fog_blurry.jpg;type=image/jpeg" \
  -F "question=Are all workers wearing hard-hats?"
```

#### Response (200 OK with `insufficient_information` status)
```json
{
  "status": "insufficient_information",
  "question": "Are all workers wearing hard-hats?",
  "intent_routing": {
    "needs_detection": true,
    "intent_category": "visual_inspection",
    "confidence": 0.98,
    "reasoning": "Query matches visual inspection pattern: '\\bhard-?hats\\b'."
  },
  "detector_called": true,
  "guardrail_evaluated": true,
  "guardrail_result": {
    "is_sufficient": false,
    "failure_mode": "severe_blur",
    "message": "Insufficient information: Image resolution or focus is too degraded (Laplacian sharpness variance: 21.4 < 60.0) to reliably verify PPE safety compliance."
  },
  "answer": "Insufficient information: Image resolution or focus is too degraded (Laplacian sharpness variance: 21.4 < 60.0) to reliably verify PPE safety compliance.",
  "model_used": "confidence_guardrail_layer",
  "total_latency_ms": 35.8
}
```
