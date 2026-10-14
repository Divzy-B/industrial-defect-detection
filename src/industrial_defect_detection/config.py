"""Configuration loaded from environment variables."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

DEFAULT_CLASS_NAMES = (
    "crazing",
    "inclusion",
    "patches",
    "pitted_surface",
    "rolled-in_scale",
    "scratches",
)


@dataclass(frozen=True)
class Settings:
    """Runtime settings for an inference process."""

    backend: str
    model_path: Path
    confidence: float
    iou: float
    image_size: int
    max_upload_bytes: int
    class_names: tuple[str, ...]

    @classmethod
    def from_env(cls) -> Settings:
        raw_names = os.getenv("DEFECT_CLASS_NAMES", "")
        names = tuple(name.strip() for name in raw_names.split(",") if name.strip())
        return cls(
            backend=os.getenv("DEFECT_BACKEND", "ultralytics").lower(),
            model_path=Path(os.getenv("DEFECT_MODEL_PATH", "models/best.pt")),
            confidence=float(os.getenv("DEFECT_CONFIDENCE", "0.35")),
            iou=float(os.getenv("DEFECT_IOU", "0.45")),
            image_size=int(os.getenv("DEFECT_IMAGE_SIZE", "640")),
            max_upload_bytes=int(os.getenv("DEFECT_MAX_UPLOAD_BYTES", str(10 * 1024 * 1024))),
            class_names=names or DEFAULT_CLASS_NAMES,
        )

    def validate(self) -> None:
        if self.backend not in {"ultralytics", "onnx"}:
            raise ValueError("DEFECT_BACKEND must be 'ultralytics' or 'onnx'")
        if not 0 < self.confidence < 1:
            raise ValueError("DEFECT_CONFIDENCE must be between 0 and 1")
        if not 0 < self.iou < 1:
            raise ValueError("DEFECT_IOU must be between 0 and 1")
        if self.image_size < 32:
            raise ValueError("DEFECT_IMAGE_SIZE must be at least 32")
        if self.max_upload_bytes < 1:
            raise ValueError("DEFECT_MAX_UPLOAD_BYTES must be positive")
