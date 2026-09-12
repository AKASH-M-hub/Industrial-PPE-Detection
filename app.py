import os
import io
import time
import gradio as gr
from PIL import Image
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger

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


@spaces.GPU(duration=60)
def predict_ppe_gpu(image):
    """ZeroGPU inference function for RT-DETR."""
    if image is None:
        return "Please upload an image."
    try:
        detector = get_detector()
        pil_img = Image.fromarray(image)
        buf = io.BytesIO()
        pil_img.save(buf, format="JPEG")
        response = detector.predict(buf.getvalue())
        counts = response.counts_by_class
        return (
            f"✅ Detections: {response.total_detections} total\n"
            f"• Workers (person): {counts.get('person', 0)}\n"
            f"• Helmets (hat): {counts.get('hat', 0)}\n"
            f"• Safety Vests (vest): {counts.get('vest', 0)}\n"
            f"• Image Sharpness: {response.image_metadata.blur_laplacian_variance:.1f}"
        )
    except Exception as e:
        return f"Error during inference: {e}"


# Build Gradio Interface
with gr.Blocks(title="PPE Safety Vision & Reasoning Console") as demo:
    gr.Markdown(
        """
        # 🦺 Industrial PPE Safety Vision & Reasoning API
        ### ZeroGPU Accelerated Backend
        
        - **Swagger API Docs**: [Open `/docs`](/docs)
        - **Health Status**: [Check `/health`](/health)
        - **Frontend App**: Connect with your Vercel deployment!
        """
    )
    with gr.Row():
        with gr.Column():
            input_img = gr.Image(type="numpy", label="Upload Inspection Image")
            run_btn = gr.Button("Run PPE Detection (GPU)", variant="primary")
        with gr.Column():
            output_result = gr.Textbox(label="Detection Results", lines=6)

    run_btn.click(fn=predict_ppe_gpu, inputs=input_img, outputs=output_result)


# 1. Create root FastAPI application
app = FastAPI(
    title=settings.PROJECT_NAME,
    version=settings.VERSION,
    docs_url="/docs",
    redoc_url="/redoc",
)

# 2. Enable universal CORS for Vercel, localhost, and all external origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 3. Mount all API routes directly on FastAPI
app.include_router(api_router, prefix=settings.API_V1_STR)
app.include_router(api_router)

# 4. Mount Gradio demo onto FastAPI app
app = gr.mount_gradio_app(app, demo, path="/")


if __name__ == "__main__":
    import uvicorn
    server_port = int(os.environ.get("PORT", 7860))
    server_host = os.environ.get("HOST", "0.0.0.0")
    logger.info(f"Starting PPE Vision & Reasoning server on {server_host}:{server_port}")
    uvicorn.run(app, host=server_host, port=server_port)
