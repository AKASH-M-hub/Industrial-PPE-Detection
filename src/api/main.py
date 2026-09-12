import os
import time
import sys
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, HTMLResponse, FileResponse
from loguru import logger

from src.config import settings
from src.api.routes import router as api_router
from src.detector.model import get_detector
from src.reasoning.engine import get_reasoning_engine

# Configure Loguru logger
logger.remove()
logger.add(
    sys.stdout,
    colorize=True,
    format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
    level=settings.LOG_LEVEL,
)
logger.add(
    "runs/api.log",
    rotation="50 MB",
    retention="10 days",
    level="DEBUG",
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info(f"Starting {settings.PROJECT_NAME} v{settings.VERSION}")
    logger.info(f"Target classes: {settings.TARGET_CLASSES}")
    logger.info(f"Confidence threshold: {settings.CONFIDENCE_THRESHOLD}")
    logger.info(f"Model path: {settings.MODEL_PATH}")

    try:
        from scripts.generate_favicon import create_favicon
        create_favicon()
    except Exception:
        pass

    try:
        get_detector()
        get_reasoning_engine()
        logger.info("Detector and reasoning engine initialized.")
    except Exception as e:
        logger.warning(f"Engine warm-up warning: {e}")

    yield
    logger.info("Server stopped.")


app = FastAPI(
    title=settings.PROJECT_NAME,
    version=settings.VERSION,
    description="Workplace PPE safety detection and natural-language reasoning API.",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    start_time = time.perf_counter()
    response = await call_next(request)
    process_time = (time.perf_counter() - start_time) * 1000
    response.headers["X-Process-Time-Ms"] = f"{process_time:.2f}"
    return response


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled error on {request.url.path}: {exc}", exc_info=True)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "status": "error",
            "code": "INTERNAL_SERVER_ERROR",
            "message": "An internal server error occurred.",
            "path": request.url.path,
        },
    )


@app.get("/", response_class=FileResponse, tags=["Dashboard"])
@app.get("/dashboard", response_class=FileResponse, tags=["Dashboard"])
async def serve_dashboard():
    root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    dashboard_path = os.path.join(root_dir, "dashboard.html")
    if os.path.exists(dashboard_path):
        return FileResponse(dashboard_path)
    return HTMLResponse("<h3>Dashboard not found. Open dashboard.html directly.</h3>")


@app.get("/memo", response_class=FileResponse, tags=["Documentation"])
async def serve_memo():
    root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    memo_path = os.path.join(root_dir, "docs", "submission_memo.html")
    if os.path.exists(memo_path):
        return FileResponse(memo_path)
    return HTMLResponse("<h3>Memo not found. Open docs/submission_memo.html directly.</h3>")


@app.get("/favicon.ico", include_in_schema=False)
async def get_favicon_ico():
    root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    ico_path = os.path.join(root_dir, "favicon.ico")
    if os.path.exists(ico_path):
        return FileResponse(ico_path, media_type="image/x-icon")
    svg_path = os.path.join(root_dir, "favicon.svg")
    return FileResponse(svg_path, media_type="image/svg+xml")


@app.get("/favicon.svg", include_in_schema=False)
@app.get("/favicon.png", include_in_schema=False)
async def get_favicon_svg(request: Request):
    root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    if "png" in request.url.path and os.path.exists(os.path.join(root_dir, "favicon.png")):
        return FileResponse(os.path.join(root_dir, "favicon.png"), media_type="image/png")
    return FileResponse(os.path.join(root_dir, "favicon.svg"), media_type="image/svg+xml")


app.include_router(api_router, prefix=settings.API_V1_STR)
app.include_router(api_router)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "src.api.main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.DEBUG,
    )
