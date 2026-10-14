"""Public API schemas."""

from __future__ import annotations

from pydantic import BaseModel, Field


class BoundingBox(BaseModel):
    """Pixel coordinates in the original image, using an xyxy convention."""

    x1: float = Field(ge=0)
    y1: float = Field(ge=0)
    x2: float = Field(ge=0)
    y2: float = Field(ge=0)


class Detection(BaseModel):
    """One localized defect."""

    class_id: int = Field(ge=0)
    label: str
    confidence: float = Field(ge=0, le=1)
    box: BoundingBox


class DetectionResponse(BaseModel):
    """Serialized inference result."""

    image_width: int = Field(gt=0)
    image_height: int = Field(gt=0)
    detections: list[Detection]
    model_backend: str
    annotated_image_base64: str | None = None
