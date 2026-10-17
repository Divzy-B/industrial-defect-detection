from __future__ import annotations

import cv2
import numpy as np
import pytest
from fastapi import HTTPException

import industrial_defect_detection.api as api_module
from industrial_defect_detection.api import create_app
from industrial_defect_detection.inference import InferenceError
from industrial_defect_detection.schemas import BoundingBox, Detection


class FakeDetector:
    backend = "fake"

    def detect(self, image: np.ndarray) -> list[Detection]:
        del image
        return [
            Detection(
                class_id=1,
                label="inclusion",
                confidence=0.88,
                box=BoundingBox(x1=2, y1=3, x2=20, y2=30),
            )
        ]


def _app_with_fake_detector():
    app = create_app(detector_factory=lambda _settings: FakeDetector())
    app.state.load_model()
    return app


def _endpoint(app, path: str):
    return next(route.endpoint for route in app.routes if route.path == path)


class FakeUpload:
    """A minimal async upload used when calling FastAPI handlers directly."""

    def __init__(self, payload: bytes) -> None:
        self.payload = payload

    async def read(self, size: int = -1) -> bytes:
        return self.payload if size < 0 else self.payload[:size]

    async def close(self) -> None:
        return None


def _upload(payload: bytes) -> FakeUpload:
    return FakeUpload(payload)


def _resolve_immediately(coroutine):
    """Drive a handler whose awaits complete in memory in these unit tests."""
    try:
        coroutine.send(None)
    except StopIteration as completed:
        return completed.value
    raise AssertionError("The fake upload unexpectedly suspended the handler.")


async def _run_inline(function, *args, **kwargs):
    return function(*args, **kwargs)


def test_health_and_detection_with_annotation(monkeypatch: pytest.MonkeyPatch) -> None:
    image = np.zeros((40, 50, 3), dtype=np.uint8)
    ok, encoded = cv2.imencode(".jpg", image)
    assert ok
    app = _app_with_fake_detector()
    monkeypatch.setattr(api_module, "run_in_threadpool", _run_inline)
    health = _endpoint(app, "/healthz")()
    response = _resolve_immediately(
        _endpoint(app, "/v1/detect")(_upload(encoded.tobytes()), annotate=True, confidence=None)
    )

    assert health["status"] == "ok"
    assert response.model_backend == "fake"
    assert response.detections[0].label == "inclusion"
    assert response.annotated_image_base64


def test_detection_rejects_malformed_image(monkeypatch: pytest.MonkeyPatch) -> None:
    app = _app_with_fake_detector()
    monkeypatch.setattr(api_module, "run_in_threadpool", _run_inline)
    with pytest.raises(HTTPException) as error:
        _resolve_immediately(
            _endpoint(app, "/v1/detect")(_upload(b"broken"), annotate=False, confidence=None)
        )

    assert error.value.status_code == 422
    assert "supported image" in error.value.detail


def test_detection_rejects_image_over_pixel_limit(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DEFECT_MAX_IMAGE_PIXELS", "100")
    app = _app_with_fake_detector()
    monkeypatch.setattr(api_module, "run_in_threadpool", _run_inline)
    image = np.zeros((10, 11, 3), dtype=np.uint8)
    ok, encoded = cv2.imencode(".png", image)
    assert ok

    with pytest.raises(HTTPException) as error:
        _resolve_immediately(
            _endpoint(app, "/v1/detect")(
                _upload(encoded.tobytes()), annotate=False, confidence=None
            )
        )

    assert error.value.status_code == 422
    assert "pixel limit" in error.value.detail


def test_readiness_is_only_ok_after_a_model_loads() -> None:
    app = _app_with_fake_detector()

    assert _endpoint(app, "/readyz")() == {"status": "ok"}


def test_readiness_returns_503_when_model_loading_fails() -> None:
    def unavailable_detector(_settings):
        raise InferenceError("model not found")

    app = create_app(detector_factory=unavailable_detector)
    app.state.load_model()

    assert _endpoint(app, "/healthz")()["status"] == "degraded"
    with pytest.raises(HTTPException) as error:
        _endpoint(app, "/readyz")()
    assert error.value.status_code == 503


def test_invalid_environment_keeps_health_endpoint_available(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("DEFECT_CONFIDENCE", "not-a-number")
    app = create_app(detector_factory=lambda _settings: FakeDetector())
    app.state.load_model()

    health = _endpoint(app, "/healthz")()
    assert health["status"] == "degraded"
    assert "could not convert string to float" in health["detail"]
