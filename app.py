import os
import io
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


# Hook into Gradio's internal FastAPI app creation so all custom routes and CORS
# are properly registered and take precedence over Gradio's catch-all SPA routes
try:
    import gradio.routes as gr_routes

    _original_create_app = gr_routes.App.create_app

    def _custom_create_app(*args, **kwargs):
        fastapi_app = _original_create_app(*args, **kwargs)

        # 1. Enable CORS for all origins (so Vercel frontend can call the backend)
        fastapi_app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

        # 2. Prevent Gradio 403 Cross-site POST form block on API endpoints
        class MaskSecFetchSiteMiddleware:
            def __init__(self, app):
                self.app = app

            async def __call__(self, scope, receive, send):
                if scope["type"] == "http":
                    headers = []
                    for k, v in scope.get("headers", []):
                        if k.lower() == b"sec-fetch-site":
                            headers.append((k, b"same-origin"))
                        else:
                            headers.append((k, v))
                    scope["headers"] = headers
                await self.app(scope, receive, send)

        fastapi_app.add_middleware(MaskSecFetchSiteMiddleware)

        # 3. Add FastAPI routes (both with /api/v1 prefix and root)
        existing_routes = list(fastapi_app.router.routes)
        fastapi_app.include_router(api_router, prefix=settings.API_V1_STR)
        fastapi_app.include_router(api_router)

        # 4. Prioritize API routes over Gradio's catch-all SvelteKit route
        new_routes = [r for r in fastapi_app.router.routes if r not in existing_routes]
        fastapi_app.router.routes = new_routes + existing_routes

        return fastapi_app

    gr_routes.App.create_app = _custom_create_app
except Exception as patch_err:
    pass


if __name__ == "__main__":
    demo.queue()
    demo.launch(server_name="0.0.0.0", server_port=7860)
