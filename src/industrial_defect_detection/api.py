"""FastAPI application factory."""

from __future__ import annotations

from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, HTTPException, Query, UploadFile

from industrial_defect_detection.config import Settings
from industrial_defect_detection.inference import Detector, InferenceError, create_detector
from industrial_defect_detection.schemas import DetectionResponse
from industrial_defect_detection.service import DetectionService
from industrial_defect_detection.vision import ImageDecodeError


def create_app(detector_factory: Callable[[Settings], Detector] = create_detector) -> FastAPI:
    """Create an API whose model loads once, during application startup."""
    settings = Settings.from_env()

    @asynccontextmanager
    async def lifespan(application: FastAPI) -> AsyncIterator[None]:
        application.state.load_model()
        yield

    app = FastAPI(
        title="Industrial Defect Detection API",
        version="0.1.0",
        description="YOLOv8 defect detection over PyTorch or ONNX Runtime.",
        lifespan=lifespan,
    )
    app.state.service = None
    app.state.model_error = None

    def load_model() -> None:
        try:
            app.state.service = DetectionService(detector_factory(settings))
            app.state.model_error = None
        except (InferenceError, ValueError) as exc:
            # Keep /healthz available so orchestrators receive a useful error.
            app.state.service = None
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
        if app.state.service is None:
            raise HTTPException(
                status_code=503, detail=app.state.model_error or "Model is unavailable."
            )
        payload = await image.read(settings.max_upload_bytes + 1)
        if len(payload) > settings.max_upload_bytes:
            raise HTTPException(
                status_code=413,
                detail=f"Image exceeds {settings.max_upload_bytes} byte upload limit.",
            )
        try:
            return app.state.service.inspect_bytes(payload, include_annotation=annotate)
        except ImageDecodeError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        except InferenceError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc

    return app


app = create_app()
