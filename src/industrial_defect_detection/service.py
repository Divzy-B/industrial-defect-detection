"""Application service used by both FastAPI and Streamlit."""

from __future__ import annotations

from threading import Lock

from industrial_defect_detection.inference import Detector
from industrial_defect_detection.schemas import DetectionResponse
from industrial_defect_detection.vision import annotate_image, decode_image, to_jpeg_base64


class DetectionService:
    """Runs one detector and formats a stable response."""

    def __init__(self, detector: Detector, *, max_image_pixels: int | None = None) -> None:
        self.detector = detector
        self.max_image_pixels = max_image_pixels
        self._inference_lock = Lock()

    def inspect_bytes(self, payload: bytes, include_annotation: bool = False) -> DetectionResponse:
        image = decode_image(payload, max_pixels=self.max_image_pixels)
        # Ultralytics/PyTorch and hardware-backed ONNX sessions are not a useful
        # concurrency boundary. Keep one model invocation at a time per process;
        # deploy more worker processes when throughput needs to scale.
        with self._inference_lock:
            detections = self.detector.detect(image)
        annotation = (
            to_jpeg_base64(annotate_image(image, detections)) if include_annotation else None
        )
        height, width = image.shape[:2]
        return DetectionResponse(
            image_width=width,
            image_height=height,
            detections=detections,
            model_backend=self.detector.backend,
            annotated_image_base64=annotation,
        )
