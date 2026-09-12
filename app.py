import os
import io
import time
import gradio as gr
from PIL import Image
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


def configure_app(fastapi_app):
    """Attach API routes, CORS, and ensure API routes take precedence over Gradio catch-all."""
    if not fastapi_app or not hasattr(fastapi_app, "router"):
        return fastapi_app

    logger.info(f"Configuring routes on FastAPI instance: {fastapi_app}")

    # 1. Mount all API routes
    try:
        fastapi_app.include_router(api_router, prefix=settings.API_V1_STR)
        fastapi_app.include_router(api_router)
    except Exception as r_err:
        logger.warning(f"Router inclusion notice: {r_err}")

    # 2. Shift API routes to the FRONT of router.routes so they match before Gradio's catch-all
    api_routes = [
        r for r in fastapi_app.router.routes 
        if hasattr(r, "path") and (
            r.path.startswith("/api/v1") or 
            r.path in ("/health", "/detect", "/reason", "/dashboard", "/memo")
        )
    ]
    other_routes = [r for r in fastapi_app.router.routes if r not in api_routes]
    fastapi_app.router.routes = api_routes + other_routes
    logger.info(f"Successfully prioritized {len(api_routes)} API routes at index 0 of router.routes!")

    # 3. Direct ASGI Stack Wrapping (bypasses Gradio 403 CSRF and injects CORS even after startup)
    try:
        from starlette.middleware.cors import CORSMiddleware

        stack = getattr(fastapi_app, "middleware_stack", None)
        if stack is None and hasattr(fastapi_app, "build_middleware_stack"):
            stack = fastapi_app.build_middleware_stack()

        if stack is not None:
            cors_wrapped = CORSMiddleware(
                stack,
                allow_origins=["*"],
                allow_credentials=True,
                allow_methods=["*"],
                allow_headers=["*"],
            )

            class UnblockCSRFMiddleware:
                def __init__(self, inner):
                    self.inner = inner

                async def __call__(self, scope, receive, send):
                    if scope.get("type") == "http":
                        headers = []
                        for k, v in scope.get("headers", []):
                            if k.lower() == b"sec-fetch-site":
                                headers.append((b"sec-fetch-site", b"same-origin"))
                            else:
                                headers.append((k, v))
                        scope["headers"] = headers
                    await self.inner(scope, receive, send)

            fastapi_app.middleware_stack = UnblockCSRFMiddleware(cors_wrapped)
            logger.info("Wrapped middleware_stack with UnblockCSRFMiddleware & CORSMiddleware successfully!")
    except Exception as m_err:
        logger.warning(f"Middleware wrap notice: {m_err}")

    return fastapi_app


if __name__ == "__main__":
    demo.queue()

    # Step 1: Pre-configure on demo.app
    try:
        configure_app(demo.app)
        logger.info("Pre-configuration on demo.app complete.")
    except Exception as pre_err:
        logger.warning(f"demo.app pre-config notice: {pre_err}")

    # Step 2: Launch without hardcoding server_port to respect ZeroGPU environment
    launch_res = None
    try:
        launch_res = demo.launch(prevent_thread_lock=True)
    except Exception as e:
        logger.warning(f"Standard launch fallback: {e}")
        launch_res = demo.launch(server_name="0.0.0.0", server_port=7860, prevent_thread_lock=True)

    # Step 3: Find the active FastAPI app on the running server and configure it
    targets = []
    if isinstance(launch_res, tuple) and len(launch_res) > 0:
        targets.append(launch_res[0])
    if hasattr(demo, "server") and hasattr(demo.server, "app"):
        targets.append(demo.server.app)
    if hasattr(demo, "app"):
        targets.append(demo.app)
    if hasattr(launch_res, "app"):
        targets.append(launch_res.app)

    configured_any = False
    for target in targets:
        try:
            configure_app(target)
            configured_any = True
        except Exception as post_err:
            logger.warning(f"Post-config attempt notice: {post_err}")

    logger.info(f"Post-launch FastAPI configuration completed: {configured_any}")

    # Step 4: Keep server process alive
    try:
        demo.block_thread()
    except Exception:
        while True:
            time.sleep(3600)
