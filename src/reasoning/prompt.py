from typing import List, Dict
from src.detector.schemas import DetectionItem, DetectionResponse


REASONING_SYSTEM_PROMPT = """You are an expert Safety Compliance AI Auditor for industrial construction sites.
You will be provided with:
1. A user question.
2. Structured detections from an RT-DETR computer vision model (class names, counts, bounding box coordinates, and confidence scores).
3. The unified confidence threshold applied (0.45).

Your instructions:
- Answer the user's question directly, clearly, and concisely in plain English.
- Base your reasoning STRICTLY on the provided structured detections.
- If asked about compliance or violations:
  * A violation occurs when a 'person' or worker is detected with 'no-helmet' or 'no-vest'.
  * Workers wearing 'hard-hat' and 'safety-vest' are compliant.
- Clearly mention exact counts and confidence levels where relevant.
- DO NOT invent, hallucinate, or assume objects that are not listed in the structured detection summary.
- If the detections are ambiguous, state what is known and what cannot be determined.
"""


def format_detection_context(res: DetectionResponse, question: str) -> str:
    """Format structured detection output into an auditable prompt context for the LLM."""
    lines = [
        f"### User Question:",
        f"{question}\n",
        f"### RT-DETR Detection Output (Confidence >= {res.detections[0].confidence if res.detections else 0.45:.2f}):",
        f"- Total Objects Detected: {res.total_detections}",
        f"- Class Counts: {res.counts_by_class}",
        f"- Image Dimensions: {res.image_metadata.width}x{res.image_metadata.height}",
        f"- Sharpness Variance (Laplacian): {res.image_metadata.blur_laplacian_variance:.1f}\n",
        f"### Individual Detections:"
    ]

    if not res.detections:
        lines.append("  (No objects detected)")
    else:
        for idx, d in enumerate(res.detections, 1):
            lines.append(
                f"  {idx}. Class: '{d.class_name}', Confidence: {d.confidence:.2f}, "
                f"Box: [x1={d.box.x1}, y1={d.box.y1}, x2={d.box.x2}, y2={d.box.y2}], "
                f"Area: {d.box.area:.0f}px²"
            )

    return "\n".join(lines)
