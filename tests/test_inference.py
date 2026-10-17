from __future__ import annotations

import numpy as np
import pytest

from industrial_defect_detection.inference import (
    InferenceError,
    _decode_yolov8_output,
    _preprocess,
    _validate_onnx_input_shape,
)


def test_preprocess_letterboxes_to_nchw_rgb_float() -> None:
    image = np.zeros((100, 200, 3), dtype=np.uint8)
    tensor, scale, pad_x, pad_y = _preprocess(image, 640)

    assert tensor.shape == (1, 3, 640, 640)
    assert tensor.dtype == np.float32
    assert scale == 3.2
    assert pad_x == 0
    assert pad_y == 160


def test_decode_yolov8_output_filters_and_unletterboxes() -> None:
    # (1, 4 + 2 classes, 2 candidates); the second candidate is below confidence.
    output = np.array([[[320, 320], [320, 320], [128, 100], [64, 100], [0.9, 0.1], [0.1, 0.2]]])
    detections = _decode_yolov8_output(
        output=output,
        scale=2.0,
        pad_x=0,
        pad_y=100,
        image_shape=(270, 320, 3),
        confidence_threshold=0.5,
        iou_threshold=0.45,
        labels={0: "crack", 1: "scratch"},
    )

    assert len(detections) == 1
    assert detections[0].label == "crack"
    assert detections[0].confidence == 0.9
    assert detections[0].box.x1 == 128
    assert detections[0].box.y1 == 94


def test_decode_yolov8_output_accepts_candidate_first_and_single_candidate_layouts() -> None:
    candidate_first = np.array(
        [[[320, 320, 128, 64, 0.9, 0.1], [160, 160, 64, 64, 0.2, 0.8]]]
    )
    detections = _decode_yolov8_output(
        output=candidate_first,
        scale=1.0,
        pad_x=0,
        pad_y=0,
        image_shape=(640, 640, 3),
        confidence_threshold=0.5,
        iou_threshold=0.45,
        labels={0: "crack", 1: "scratch"},
    )

    assert [item.label for item in detections] == ["crack", "scratch"]

    single_candidate = candidate_first[:, :1, :].transpose(0, 2, 1)
    detections = _decode_yolov8_output(
        output=single_candidate,
        scale=1.0,
        pad_x=0,
        pad_y=0,
        image_shape=(640, 640, 3),
        confidence_threshold=0.5,
        iou_threshold=0.45,
        labels={0: "crack", 1: "scratch"},
    )

    assert len(detections) == 1
    assert detections[0].label == "crack"


def test_decode_yolov8_output_rejects_a_class_mapping_mismatch() -> None:
    output = np.array([[[320], [320], [128], [64], [0.9], [0.1]]])

    with pytest.raises(InferenceError, match="DEFECT_CLASS_NAMES"):
        _decode_yolov8_output(
            output=output,
            scale=1.0,
            pad_x=0,
            pad_y=0,
            image_shape=(640, 640, 3),
            confidence_threshold=0.5,
            iou_threshold=0.45,
            labels={0: "crack"},
        )


def test_onnx_input_shape_must_match_fixed_shape_export() -> None:
    _validate_onnx_input_shape([1, 3, 640, 640], 640)

    with pytest.raises(InferenceError, match="expects 512x512"):
        _validate_onnx_input_shape([1, 3, 512, 512], 640)
    with pytest.raises(InferenceError, match="square"):
        _validate_onnx_input_shape([1, 3, 512, 640], 640)
