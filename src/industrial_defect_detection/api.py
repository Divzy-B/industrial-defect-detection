"""FastAPI application factory."""

from __future__ import annotations

from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from starlette.concurrency import run_in_threadpool

from industrial_defect_detection import __version__
from industrial_defect_detection.config import Settings
from industrial_defect_detection.inference import Detector, InferenceError, create_detector
from industrial_defect_detection.schemas import DetectionResponse
from industrial_defect_detection.service import DetectionService
from industrial_defect_detection.vision import ImageDecodeError


def create_app(detector_factory: Callable[[Settings], Detector] = create_detector) -> FastAPI:
    """Create an API whose model loads once, during application startup."""
    @asynccontextmanager
    async def lifespan(application: FastAPI) -> AsyncIterator[None]:
        application.state.load_model()
        yield

    app = FastAPI(
        title="Industrial Defect Detection API",
        version=__version__,
        description="YOLOv8 defect detection over PyTorch or ONNX Runtime.",
        lifespan=lifespan,
    )
    app.state.service = None
    app.state.settings = None
    app.state.model_error = None

    def load_model() -> None:
        try:
            settings = Settings.from_env()
            settings.validate()
            app.state.service = DetectionService(
                detector_factory(settings), max_image_pixels=settings.max_image_pixels
            )
            app.state.settings = settings
            app.state.model_error = None
        except (InferenceError, ValueError) as exc:
            # Keep /healthz available so orchestrators receive a useful error.
            app.state.service = None
            app.state.settings = None
            app.state.model_error = str(exc)

    app.state.load_model = load_model

    @app.get("/healthz", tags=["operations"])
    def healthz() -> dict[str, str | bool]:
        return {
            "status": "ok" if app.state.service else "degraded",
            "model_loaded": app.state.service is not None,
            "detail": app.state.model_error or "Model is ready.",
        }

    @app.get("/", tags=["operations"])
    def root() -> dict[str, str]:
        return {"service": "industrial-defect-detection", "docs": "/docs"}

    @app.get("/readyz", tags=["operations"])
    def readyz() -> dict[str, str]:
        """Report whether this process can accept inference requests."""
        settings = app.state.settings
        if app.state.service is None or settings is None:
            raise HTTPException(
                status_code=503, detail=app.state.model_error or "Model is unavailable."
            )
        return {"status": "ok"}

    @app.post("/v1/detect", response_model=DetectionResponse, tags=["inference"])
    async def detect(
        image: UploadFile = File(..., description="JPEG, PNG, or another OpenCV-decodable image."),
        annotate: bool = Query(False, description="Include an annotated JPEG as base64."),
        confidence: float | None = Query(
            None, ge=0.01, le=0.99, description="Reserved for request-scoped thresholding."
        ),
    ) -> DetectionResponse:
        del (
            confidence
        )  # Threshold is intentionally configured at process start for reproducibility.
        settings = app.state.settings
        if app.state.service is None or settings is None:
            raise HTTPException(
                status_code=503, detail=app.state.model_error or "Model is unavailable."
            )
        try:
            payload = await image.read(settings.max_upload_bytes + 1)
        finally:
            await image.close()
        if len(payload) > settings.max_upload_bytes:
            raise HTTPException(
                status_code=413,
                detail=f"Image exceeds {settings.max_upload_bytes} byte upload limit.",
            )
        try:
            return await run_in_threadpool(
                app.state.service.inspect_bytes, payload, include_annotation=annotate
            )
        except ImageDecodeError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        except InferenceError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc

    return app


app = create_app()
