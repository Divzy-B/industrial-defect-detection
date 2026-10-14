"""Image decoding, rendering, and encoding built on OpenCV."""

from __future__ import annotations

import base64

import cv2
import numpy as np

from industrial_defect_detection.schemas import Detection


class ImageDecodeError(ValueError):
    """Raised when a client payload is not a decodable raster image."""


def decode_image(payload: bytes) -> np.ndarray:
    """Decode image bytes to a non-empty BGR OpenCV array."""
    if not payload:
        raise ImageDecodeError("The uploaded file is empty.")
    image = cv2.imdecode(np.frombuffer(payload, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None or image.size == 0:
        raise ImageDecodeError("The uploaded file is not a supported image.")
    return image


def annotate_image(image: np.ndarray, detections: list[Detection]) -> np.ndarray:
    """Draw opaque high-contrast boxes and labels on a copy of an image."""
    annotated = image.copy()
    height, width = annotated.shape[:2]
    for detection in detections:
        box = detection.box
        start = (max(0, int(box.x1)), max(0, int(box.y1)))
        end = (min(width - 1, int(box.x2)), min(height - 1, int(box.y2)))
        color = _color_for_class(detection.class_id)
        cv2.rectangle(annotated, start, end, color, 2, lineType=cv2.LINE_AA)
        caption = f"{detection.label} {detection.confidence:.0%}"
        (caption_width, caption_height), baseline = cv2.getTextSize(
            caption, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
        )
        label_top = max(caption_height + baseline + 4, start[1])
        cv2.rectangle(
            annotated,
            (start[0], label_top - caption_height - baseline - 4),
            (start[0] + caption_width + 6, label_top),
            color,
            thickness=-1,
        )
        cv2.putText(
            annotated,
            caption,
            (start[0] + 3, label_top - baseline - 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 0),
            1,
            lineType=cv2.LINE_AA,
        )
    return annotated


def to_jpeg_base64(image: np.ndarray) -> str:
    """Encode a BGR image as a compact base64 JPEG string."""
    ok, encoded = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 90])
    if not ok:
        raise RuntimeError("Could not encode annotated image.")
    return base64.b64encode(encoded.tobytes()).decode("ascii")


def _color_for_class(class_id: int) -> tuple[int, int, int]:
    palette = ((82, 191, 255), (109, 220, 124), (245, 178, 76), (217, 91, 67))
    return palette[class_id % len(palette)]
