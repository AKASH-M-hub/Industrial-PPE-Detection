try:
    import spaces
except ImportError:
    class _MockSpaces:
        @staticmethod
        def GPU(task=None, duration=60, **kwargs):
            if callable(task):
                return task
            def decorator(f):
                return f
            return decorator
    spaces = _MockSpaces()

import os
import io
import time
import json
import numpy as np
from PIL import Image
from loguru import logger
import gradio as gr

from src.api.routes import router as api_router
from src.config import settings
from src.detector.model import get_detector
from src.reasoning.engine import get_reasoning_engine


# ZeroGPU startup verification probe
@spaces.GPU
def _zerogpu_probe():
    """Startup probe registered for Hugging Face ZeroGPU runtime supervisor."""
    return True


def _to_pil_image(image):
    if image is None:
        return None
    if isinstance(image, np.ndarray):
        return Image.fromarray(image)
    if isinstance(image, Image.Image):
        return image
    if isinstance(image, str):
        return Image.open(image)
    if isinstance(image, bytes):
        return Image.open(io.BytesIO(image))
    return None


def _run_predict_pipeline(image):
    if image is None:
        return json.dumps({"error": "Please upload an image."})
    detector = get_detector()
    pil_img = _to_pil_image(image)
    if pil_img is None:
        return json.dumps({"error": "Invalid image payload."})
    buf = io.BytesIO()
    pil_img.save(buf, format="JPEG")
    response = detector.predict(buf.getvalue())
    return response.model_dump_json()


def _run_reason_pipeline(image, question):
    if image is None:
        return json.dumps({"error": "Please upload an image."})
    detector = get_detector()
    pil_img = _to_pil_image(image)
    if pil_img is None:
        return json.dumps({"error": "Invalid image payload."})
    buf = io.BytesIO()
    pil_img.save(buf, format="JPEG")
    engine = get_reasoning_engine()
    reason_res = engine.reason(
        question=question or "Is everyone wearing required PPE?",
        image_bytes=buf.getvalue(),
        detector=detector
    )
    return reason_res.model_dump_json()


@spaces.GPU(duration=60)
def predict_ppe_gpu(image):
    """ZeroGPU inference function for RT-DETR returning full DetectionResponse JSON."""
    try:
        return _run_predict_pipeline(image)
    except Exception as e:
        logger.error(f"Inference error on GPU, fallback to CPU: {e}")
        return _run_predict_pipeline(image)


def predict_ppe_cpu(image):
    """CPU fallback inference function with no ZeroGPU quota limits."""
    try:
        return _run_predict_pipeline(image)
    except Exception as e:
        logger.error(f"CPU Inference error: {e}")
        return json.dumps({"error": str(e)})


@spaces.GPU(duration=60)
def reason_ppe_gpu(image, question):
    """ZeroGPU natural language reasoning function."""
    try:
        return _run_reason_pipeline(image, question)
    except Exception as e:
        logger.error(f"Reasoning error on GPU, fallback to CPU: {e}")
        return _run_reason_pipeline(image, question)


def reason_ppe_cpu(image, question):
    """CPU fallback reasoning function with no ZeroGPU quota limits."""
    try:
        return _run_reason_pipeline(image, question)
    except Exception as e:
        logger.error(f"CPU Reasoning error: {e}")
        return json.dumps({"error": str(e)})


# Build Gradio Interface with named API endpoints
with gr.Blocks(title="PPE Safety Vision & Reasoning Console") as demo:
    gr.Markdown(
        """
        # 🦺 Industrial PPE Safety Vision & Reasoning API
        ### ZeroGPU Accelerated Backend with Automatic CPU Fallback
        
        - **Frontend App**: Connect with your Vercel deployment!
        """
    )
    with gr.Row():
        with gr.Column():
            input_img = gr.Image(type="numpy", label="Upload Inspection Image")
            question_input = gr.Textbox(label="Question", value="Is he wearing helmet or not?")
            run_btn = gr.Button("Run PPE Detection (GPU)", variant="primary")
            reason_btn = gr.Button("Run Safety Reasoning (GPU)", variant="secondary")
            cpu_detect_btn = gr.Button("Run PPE Detection (CPU Fallback)", visible=False)
            cpu_reason_btn = gr.Button("Run Safety Reasoning (CPU Fallback)", visible=False)
        with gr.Column():
            output_result = gr.Textbox(label="Detection Results (JSON)", lines=6)
            reason_result = gr.Textbox(label="Reasoning Results (JSON)", lines=6)

    run_btn.click(fn=predict_ppe_gpu, inputs=input_img, outputs=output_result, api_name="predict")
    reason_btn.click(fn=reason_ppe_gpu, inputs=[input_img, question_input], outputs=reason_result, api_name="reason")
    cpu_detect_btn.click(fn=predict_ppe_cpu, inputs=input_img, outputs=output_result, api_name="predict_cpu")
    cpu_reason_btn.click(fn=reason_ppe_cpu, inputs=[input_img, question_input], outputs=reason_result, api_name="reason_cpu")

# Attach API routes directly to demo.app before launch
if hasattr(demo, "app") and demo.app is not None:
    try:
        logger.info("Attaching FastAPI routes to demo.app before launch")
        demo.app.include_router(api_router, prefix=settings.API_V1_STR)
        demo.app.include_router(api_router)

        # Shift API routes to index 0 so they evaluate before Gradio catch-all
        api_routes = [
            r for r in demo.app.router.routes 
            if hasattr(r, "path") and (
                r.path.startswith("/api/v1") or 
                r.path in ("/health", "/detect", "/reason", "/dashboard", "/memo")
            )
        ]
        other_routes = [r for r in demo.app.router.routes if r not in api_routes]
        demo.app.router.routes = api_routes + other_routes
        logger.info(f"Successfully mounted {len(api_routes)} API routes at index 0 of demo.app!")
    except Exception as mount_err:
        logger.warning(f"Could not pre-mount FastAPI routes on demo.app: {mount_err}")


if __name__ == "__main__":
    demo.queue().launch()

