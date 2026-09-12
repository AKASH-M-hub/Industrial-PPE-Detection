import os
import io
import time
import gradio as gr
from PIL import Image
from fastapi.middleware.cors import CORSMiddleware

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


def configure_app(fastapi_app):
    """Attach API routes, CORS, and ensure API routes take precedence over Gradio catch-all."""
    if not fastapi_app or not hasattr(fastapi_app, "router"):
        return fastapi_app

    # 1. Enable CORS for all origins (so Vercel frontend can call the backend)
    fastapi_app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # 2. Prevent Gradio 403 on cross-site form/API requests
    class MaskSecFetchSiteMiddleware:
        def __init__(self, inner):
            self.inner = inner

        async def __call__(self, scope, receive, send):
            if scope.get("type") == "http":
                headers = []
                for k, v in scope.get("headers", []):
                    if k.lower() == b"sec-fetch-site":
                        headers.append((k, b"same-origin"))
                    else:
                        headers.append((k, v))
                scope["headers"] = headers
            await self.inner(scope, receive, send)

    fastapi_app.add_middleware(MaskSecFetchSiteMiddleware)

    # 3. Mount all API routes
    fastapi_app.include_router(api_router, prefix=settings.API_V1_STR)
    fastapi_app.include_router(api_router)

    # 4. Shift API routes to the FRONT of router.routes so they match before Gradio's catch-all
    api_routes = [
        r for r in fastapi_app.router.routes 
        if hasattr(r, "path") and (
            r.path.startswith("/api/v1") or 
            r.path in ("/health", "/detect", "/reason", "/dashboard", "/memo")
        )
    ]
    other_routes = [r for r in fastapi_app.router.routes if r not in api_routes]
    fastapi_app.router.routes = api_routes + other_routes
    return fastapi_app


if __name__ == "__main__":
    demo.queue()

    # Pre-configure demo.app
    try:
        configure_app(demo.app)
    except Exception:
        pass

    # Launch without hardcoding server_port to respect ZeroGPU environment
    app_instance = None
    try:
        launch_res = demo.launch(prevent_thread_lock=True)
        if isinstance(launch_res, tuple) and len(launch_res) > 0:
            app_instance = launch_res[0]
    except Exception:
        launch_res = demo.launch(server_name="0.0.0.0", server_port=7860, prevent_thread_lock=True)
        if isinstance(launch_res, tuple) and len(launch_res) > 0:
            app_instance = launch_res[0]

    # Post-configure the actual running app instance
    if app_instance:
        try:
            configure_app(app_instance)
        except Exception:
            pass

    # Keep server alive
    try:
        demo.block_thread()
    except Exception:
        while True:
            time.sleep(3600)
