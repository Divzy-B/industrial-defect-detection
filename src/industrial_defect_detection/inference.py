"""Inference adapters for Ultralytics/PyTorch and ONNX Runtime."""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral
from pathlib import Path
from typing import Protocol

import cv2
import numpy as np

from industrial_defect_detection.config import Settings
from industrial_defect_detection.schemas import BoundingBox, Detection


class InferenceError(RuntimeError):
    """Raised when a configured inference backend cannot produce a result."""


class Detector(Protocol):
    """Minimal detector contract shared by the API and dashboard."""

    backend: str

    def detect(self, image: np.ndarray) -> list[Detection]:
        """Return detections with boxes in the input image coordinate system."""


@dataclass
class UltralyticsDetector:
    """YOLOv8 detector backed by the PyTorch Ultralytics runtime."""

    model: object
    settings: Settings
    names: dict[int, str]
    backend: str = "ultralytics"

    @classmethod
    def load(cls, settings: Settings) -> UltralyticsDetector:
        _ensure_model_exists(settings.model_path)
        try:
            from ultralytics import YOLO
        except ImportError as exc:  # pragma: no cover - environment-specific
            raise InferenceError(
                "Install the 'ultralytics' extra to use the PyTorch backend."
            ) from exc
        model = YOLO(str(settings.model_path))
        raw_names = getattr(model, "names", {})
        names = dict(raw_names) if isinstance(raw_names, dict) else dict(enumerate(raw_names))
        return cls(model=model, settings=settings, names=names)

    def detect(self, image: np.ndarray) -> list[Detection]:
        try:
            result = self.model.predict(
                source=image,
                conf=self.settings.confidence,
                iou=self.settings.iou,
                imgsz=self.settings.image_size,
                verbose=False,
            )[0]
        except Exception as exc:  # pragma: no cover - depends on runtime/model
            raise InferenceError(f"YOLOv8 inference failed: {exc}") from exc

        boxes = result.boxes
        if boxes is None:
            return []
        xyxy = boxes.xyxy.cpu().numpy()
        confidences = boxes.conf.cpu().numpy()
        class_ids = boxes.cls.cpu().numpy().astype(int)
        return [
            _make_detection(
                class_id=class_id,
                confidence=float(confidence),
                coords=coords,
                labels=self.names,
            )
            for coords, confidence, class_id in zip(xyxy, confidences, class_ids, strict=True)
        ]


@dataclass
class OnnxDetector:
    """YOLOv8 ONNX detector with OpenCV letterboxing and NMS."""

    session: object
    input_name: str
    settings: Settings
    backend: str = "onnx"

    @classmethod
    def load(cls, settings: Settings) -> OnnxDetector:
        _ensure_model_exists(settings.model_path)
        try:
            import onnxruntime as ort
        except ImportError as exc:  # pragma: no cover - environment-specific
            raise InferenceError("Install 'onnxruntime' to use the ONNX backend.") from exc
        try:
            session = ort.InferenceSession(
                str(settings.model_path), providers=["CPUExecutionProvider"]
            )
        except Exception as exc:  # pragma: no cover - environment-specific
            raise InferenceError(f"Could not load ONNX model: {exc}") from exc
        input_metadata = session.get_inputs()[0]
        _validate_onnx_input_shape(input_metadata.shape, settings.image_size)
        return cls(session=session, input_name=input_metadata.name, settings=settings)

    def detect(self, image: np.ndarray) -> list[Detection]:
        tensor, scale, pad_x, pad_y = _preprocess(image, self.settings.image_size)
        try:
            output = self.session.run(None, {self.input_name: tensor})[0]
        except Exception as exc:  # pragma: no cover - depends on runtime/model
            raise InferenceError(f"ONNX inference failed: {exc}") from exc
        return _decode_yolov8_output(
            output=output,
            scale=scale,
            pad_x=pad_x,
            pad_y=pad_y,
            image_shape=image.shape,
            confidence_threshold=self.settings.confidence,
            iou_threshold=self.settings.iou,
            labels=dict(enumerate(self.settings.class_names)),
        )


def create_detector(settings: Settings) -> Detector:
    """Construct the configured detector only after settings have been validated."""
    settings.validate()
    if settings.backend == "ultralytics":
        return UltralyticsDetector.load(settings)
    if settings.backend == "onnx":
        return OnnxDetector.load(settings)
    raise InferenceError(f"Unsupported backend: {settings.backend}")


def _ensure_model_exists(path: Path) -> None:
    if not path.is_file():
        raise InferenceError(
            f"Model file '{path}' was not found. Mount or copy trained weights and set "
            "DEFECT_MODEL_PATH."
        )


def _validate_onnx_input_shape(shape: object, image_size: int) -> None:
    """Fail early when a fixed-shape export disagrees with runtime preprocessing."""
    if not isinstance(shape, list | tuple) or len(shape) != 4:
        raise InferenceError(f"ONNX input must be NCHW with four dimensions, got {shape!r}")
    channels, height, width = shape[1:]
    if isinstance(channels, Integral) and channels != 3:
        raise InferenceError(f"ONNX input must have three color channels, got {channels}")
    if isinstance(height, Integral) and isinstance(width, Integral):
        if height != width:
            raise InferenceError(
                f"Only square YOLOv8 ONNX inputs are supported, got {height}x{width}."
            )
        if height != image_size:
            raise InferenceError(
                f"ONNX model expects {height}x{width}, but DEFECT_IMAGE_SIZE is {image_size}. "
                "Export with the configured size or update DEFECT_IMAGE_SIZE."
            )


def _make_detection(
    class_id: int, confidence: float, coords: np.ndarray, labels: dict[int, str]
) -> Detection:
    x1, y1, x2, y2 = (float(value) for value in coords[:4])
    return Detection(
        class_id=class_id,
        label=labels.get(class_id, f"class_{class_id}"),
        confidence=confidence,
        box=BoundingBox(x1=max(0, x1), y1=max(0, y1), x2=max(0, x2), y2=max(0, y2)),
    )


def _preprocess(image: np.ndarray, image_size: int) -> tuple[np.ndarray, float, int, int]:
    """Letterbox a BGR image and return an NCHW RGB float32 tensor."""
    height, width = image.shape[:2]
    scale = min(image_size / width, image_size / height)
    resized_width, resized_height = round(width * scale), round(height * scale)
    resized = cv2.resize(image, (resized_width, resized_height), interpolation=cv2.INTER_LINEAR)
    canvas = np.full((image_size, image_size, 3), 114, dtype=np.uint8)
    pad_x = (image_size - resized_width) // 2
    pad_y = (image_size - resized_height) // 2
    canvas[pad_y : pad_y + resized_height, pad_x : pad_x + resized_width] = resized
    tensor = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB).transpose(2, 0, 1)
    return np.ascontiguousarray(tensor, dtype=np.float32)[None] / 255.0, scale, pad_x, pad_y


def _decode_yolov8_output(
    *,
    output: np.ndarray,
    scale: float,
    pad_x: int,
    pad_y: int,
    image_shape: tuple[int, ...],
    confidence_threshold: float,
    iou_threshold: float,
    labels: dict[int, str],
) -> list[Detection]:
    """Decode common YOLOv8 detect-export layouts: (1, 4+C, N) or (1, N, 4+C)."""
    if output.ndim != 3 or output.shape[0] != 1:
        raise InferenceError(f"Unexpected ONNX output shape: {output.shape}")
    predictions = output[0]
    expected_features = 4 + len(labels)
    if predictions.shape[1] == expected_features:
        pass
    elif predictions.shape[0] == expected_features:
        predictions = predictions.T
    else:
        raise InferenceError(
            "ONNX output class count does not match DEFECT_CLASS_NAMES: "
            f"expected {expected_features} features (xywh + {len(labels)} classes), "
            f"got {predictions.shape}."
        )

    boxes_xywh: list[list[int]] = []
    scores: list[float] = []
    class_ids: list[int] = []
    raw_boxes: list[np.ndarray] = []
    for row in predictions:
        class_scores = row[4:]
        class_id = int(np.argmax(class_scores))
        confidence = float(class_scores[class_id])
        if confidence < confidence_threshold:
            continue
        cx, cy, width, height = row[:4]
        x1 = (float(cx) - float(width) / 2 - pad_x) / scale
        y1 = (float(cy) - float(height) / 2 - pad_y) / scale
        x2 = (float(cx) + float(width) / 2 - pad_x) / scale
        y2 = (float(cy) + float(height) / 2 - pad_y) / scale
        raw_boxes.append(np.array([x1, y1, x2, y2], dtype=float))
        boxes_xywh.append([int(x1), int(y1), max(1, int(x2 - x1)), max(1, int(y2 - y1))])
        scores.append(confidence)
        class_ids.append(class_id)

    if not boxes_xywh:
        return []
    # NMS independently per class avoids one defect type suppressing another
    # merely because their boxes overlap.
    kept_indexes: list[int] = []
    for class_id in sorted(set(class_ids)):
        member_indexes = [i for i, candidate_id in enumerate(class_ids) if candidate_id == class_id]
        class_boxes = [boxes_xywh[i] for i in member_indexes]
        class_scores = [scores[i] for i in member_indexes]
        kept = cv2.dnn.NMSBoxes(class_boxes, class_scores, confidence_threshold, iou_threshold)
        kept_indexes.extend(member_indexes[int(i)] for i in np.asarray(kept).reshape(-1))
    height, width = image_shape[:2]
    detections: list[Detection] = []
    for index in sorted(kept_indexes, key=lambda candidate: scores[candidate], reverse=True):
        coords = raw_boxes[index]
        coords[[0, 2]] = np.clip(coords[[0, 2]], 0, width - 1)
        coords[[1, 3]] = np.clip(coords[[1, 3]], 0, height - 1)
        detections.append(_make_detection(class_ids[index], scores[index], coords, labels))
    return detections
