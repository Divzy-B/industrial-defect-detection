"""Export a trained YOLOv8 PyTorch checkpoint to ONNX."""

from __future__ import annotations

import argparse
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Export a YOLOv8 checkpoint to ONNX.")
    parser.add_argument("model", type=Path, help="Path to a trained .pt checkpoint.")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--opset", type=int, default=17)
    parser.add_argument("--simplify", action="store_true")
    args = parser.parse_args()
    if not args.model.is_file():
        raise SystemExit(f"Checkpoint not found: {args.model}")
    from ultralytics import YOLO

    exported = YOLO(str(args.model)).export(
        format="onnx", imgsz=args.imgsz, opset=args.opset, simplify=args.simplify
    )
    print(f"Exported ONNX model: {exported}")


if __name__ == "__main__":
    main()
