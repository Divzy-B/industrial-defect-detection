"""Train a YOLOv8 model from a local Ultralytics dataset definition."""

from __future__ import annotations

import argparse
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune YOLOv8 for industrial surface defects.")
    parser.add_argument(
        "--data", type=Path, required=True, help="Path to an Ultralytics data YAML file."
    )
    parser.add_argument(
        "--model", default="yolov8s.pt", help="Base checkpoint or model architecture."
    )
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--project", default="runs/train")
    parser.add_argument("--name", default="industrial-defects")
    parser.add_argument(
        "--device", default=None, help="CUDA device, 'cpu', or omit for Ultralytics default."
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.data.is_file():
        raise SystemExit(f"Dataset YAML not found: {args.data}")
    from ultralytics import YOLO

    model = YOLO(args.model)
    model.train(
        data=str(args.data),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        project=args.project,
        name=args.name,
        device=args.device,
        pretrained=True,
        patience=20,
    )


if __name__ == "__main__":
    main()
