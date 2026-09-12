import os
import io
import uvicorn
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse
import gradio as gr
from PIL import Image

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


# 1. Create FastAPI application
app = FastAPI(
    title=settings.PROJECT_NAME,
    version=settings.VERSION,
    description="Workplace PPE safety detection and natural-language reasoning API.",
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
)

# 2. Add CORS middleware so Vercel deployment and any external origin can call the API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 3. Mount all FastAPI endpoints (both with /api/v1 prefix and root prefix)
app.include_router(api_router, prefix=settings.API_V1_STR)
app.include_router(api_router)


@app.get("/dashboard", response_class=FileResponse, tags=["Dashboard"])
async def serve_dashboard():
    root_dir = os.path.dirname(os.path.abspath(__file__))
    dashboard_path = os.path.join(root_dir, "dashboard.html")
    if os.path.exists(dashboard_path):
        return FileResponse(dashboard_path)
    return HTMLResponse("<h3>Dashboard not found.</h3>")


@app.get("/memo", response_class=FileResponse, tags=["Documentation"])
async def serve_memo():
    root_dir = os.path.dirname(os.path.abspath(__file__))
    memo_path = os.path.join(root_dir, "docs", "submission_memo.html")
    if os.path.exists(memo_path):
        return FileResponse(memo_path)
    return HTMLResponse("<h3>Memo not found.</h3>")


# 4. Mount Gradio interface onto FastAPI app at root "/"
# In Starlette, FastAPI routes defined above take priority over Gradio's catch-all,
# while the root "/" and Gradio assets are routed to the Gradio Blocks UI.
app = gr.mount_gradio_app(app, demo, path="/")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=7860)
