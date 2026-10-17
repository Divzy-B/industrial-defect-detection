from __future__ import annotations

import base64

import cv2
import numpy as np
import pytest

from industrial_defect_detection.schemas import BoundingBox, Detection
from industrial_defect_detection.vision import (
    ImageDecodeError,
    annotate_image,
    decode_image,
    to_jpeg_base64,
)


def test_decode_annotate_and_encode_round_trip() -> None:
    image = np.full((60, 80, 3), 255, dtype=np.uint8)
    ok, encoded = cv2.imencode(".png", image)
    assert ok
    decoded = decode_image(encoded.tobytes())
    detection = Detection(
        class_id=0,
        label="scratch",
        confidence=0.91,
        box=BoundingBox(x1=5, y1=5, x2=40, y2=30),
    )
    annotated = annotate_image(decoded, [detection])
    encoded_annotation = to_jpeg_base64(annotated)

    assert annotated.shape == image.shape
    assert base64.b64decode(encoded_annotation).startswith(b"\xff\xd8")


def test_decode_rejects_empty_and_invalid_payloads() -> None:
    with pytest.raises(ImageDecodeError):
        decode_image(b"")
    with pytest.raises(ImageDecodeError):
        decode_image(b"not an image")


def test_decode_rejects_an_image_over_the_pixel_limit() -> None:
    image = np.zeros((10, 11, 3), dtype=np.uint8)
    ok, encoded = cv2.imencode(".png", image)
    assert ok

    with pytest.raises(ImageDecodeError, match="100 pixel"):
        decode_image(encoded.tobytes(), max_pixels=100)
