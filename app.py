import os
import io
import time
import json
import numpy as np
from PIL import Image
from loguru import logger
import gradio as gr

# ZeroGPU Compatibility Layer
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

from src.api.routes import router as api_router
from src.config import settings
from src.detector.model import get_detector
from src.reasoning.engine import get_reasoning_engine


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


@spaces.GPU(duration=60)
def predict_ppe_gpu(image):
    """ZeroGPU inference function for RT-DETR returning full DetectionResponse JSON."""
    if image is None:
        return json.dumps({"error": "Please upload an image."})
    try:
        detector = get_detector()
        pil_img = _to_pil_image(image)
        if pil_img is None:
            return json.dumps({"error": "Invalid image payload."})
        buf = io.BytesIO()
        pil_img.save(buf, format="JPEG")
        response = detector.predict(buf.getvalue())
        return response.model_dump_json()
    except Exception as e:
        logger.error(f"Inference error: {e}")
        return json.dumps({"error": str(e)})


@spaces.GPU(duration=60)
def reason_ppe_gpu(image, question):
    """ZeroGPU natural language reasoning function."""
    if image is None:
        return json.dumps({"error": "Please upload an image."})
    try:
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
    except Exception as e:
        logger.error(f"Reasoning error: {e}")
        return json.dumps({"error": str(e)})


# Build Gradio Interface with named API endpoints
with gr.Blocks(title="PPE Safety Vision & Reasoning Console") as demo:
    gr.Markdown(
        """
        # 🦺 Industrial PPE Safety Vision & Reasoning API
        ### ZeroGPU Accelerated Backend
        
        - **Frontend App**: Connect with your Vercel deployment!
        """
    )
    with gr.Row():
        with gr.Column():
            input_img = gr.Image(type="numpy", label="Upload Inspection Image")
            question_input = gr.Textbox(label="Question", value="Is he wearing helmet or not?")
            run_btn = gr.Button("Run PPE Detection (GPU)", variant="primary")
            reason_btn = gr.Button("Run Safety Reasoning (GPU)", variant="secondary")
        with gr.Column():
            output_result = gr.Textbox(label="Detection Results (JSON)", lines=6)
            reason_result = gr.Textbox(label="Reasoning Results (JSON)", lines=6)

    run_btn.click(fn=predict_ppe_gpu, inputs=input_img, outputs=output_result, api_name="predict")
    reason_btn.click(fn=reason_ppe_gpu, inputs=[input_img, question_input], outputs=reason_result, api_name="reason")


if __name__ == "__main__":
    demo.queue()

    # Launch Gradio server (ZeroGPU hooks into demo.launch)
    launch_res = None
    try:
        launch_res = demo.launch(prevent_thread_lock=True)
    except Exception as e:
        logger.warning(f"Standard launch fallback: {e}")
        launch_res = demo.launch(server_name="0.0.0.0", server_port=7860, prevent_thread_lock=True)

    # Extract FastAPI app from running Gradio instance
    active_app = None
    if isinstance(launch_res, tuple) and len(launch_res) > 0:
        active_app = launch_res[0]
    elif hasattr(demo, "server") and hasattr(demo.server, "app"):
        active_app = demo.server.app
    elif hasattr(demo, "app"):
        active_app = demo.app

    if active_app and hasattr(active_app, "include_router"):
        logger.info(f"Attaching API routes to active FastAPI app: {active_app}")
        active_app.include_router(api_router, prefix=settings.API_V1_STR)
        active_app.include_router(api_router)

        # Shift API routes to index 0 so they evaluate before Gradio catch-all
        api_routes = [
            r for r in active_app.router.routes 
            if hasattr(r, "path") and (
                r.path.startswith("/api/v1") or 
                r.path in ("/health", "/detect", "/reason", "/dashboard", "/memo")
            )
        ]
        other_routes = [r for r in active_app.router.routes if r not in api_routes]
        active_app.router.routes = api_routes + other_routes
        logger.info(f"Successfully mounted {len(api_routes)} API routes at index 0 of router.routes!")

    # Keep the server alive
    try:
        demo.block_thread()
    except Exception:
        while True:
            time.sleep(3600)
