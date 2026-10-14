"""Streamlit dashboard entry point."""

from __future__ import annotations

import base64

import streamlit as st

from industrial_defect_detection.config import Settings
from industrial_defect_detection.inference import InferenceError, create_detector
from industrial_defect_detection.service import DetectionService


@st.cache_resource(show_spinner="Loading detector…")
def _service() -> DetectionService:
    return DetectionService(create_detector(Settings.from_env()))


def main() -> None:
    st.set_page_config(page_title="Industrial Defect Detection", page_icon="🔎", layout="wide")
    st.title("Industrial Defect Detection")
    st.caption("YOLOv8 visual inspection with PyTorch or ONNX Runtime")
    st.info(
        "Upload an image from a validated inspection workflow. Results are decision support, "
        "not a substitute for quality-release controls."
    )
    uploaded = st.file_uploader("Surface image", type=["jpg", "jpeg", "png", "bmp", "webp"])
    if not uploaded:
        st.stop()
    payload = uploaded.getvalue()
    try:
        result = _service().inspect_bytes(payload, include_annotation=True)
    except (InferenceError, ValueError) as exc:
        st.error(str(exc))
        st.stop()
    except Exception as exc:  # pragma: no cover - user UI defense
        st.error(f"Could not process image: {exc}")
        st.stop()

    left, right = st.columns((2, 1))
    with left:
        st.image(
            base64.b64decode(result.annotated_image_base64 or ""),
            caption=f"{len(result.detections)} detection(s) · {result.model_backend}",
            use_container_width=True,
        )
    with right:
        st.metric("Defects found", len(result.detections))
        st.dataframe(
            [
                {
                    "label": item.label,
                    "confidence": f"{item.confidence:.1%}",
                    "x1": round(item.box.x1),
                    "y1": round(item.box.y1),
                    "x2": round(item.box.x2),
                    "y2": round(item.box.y2),
                }
                for item in result.detections
            ],
            use_container_width=True,
            hide_index=True,
        )


if __name__ == "__main__":
    main()
