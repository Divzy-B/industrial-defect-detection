"""Export a trained YOLOv8 PyTorch checkpoint to ONNX."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _package_version(name: str) -> str | None:
    try:
        return version(name)
    except PackageNotFoundError:
        return None


def _class_names(model: object) -> list[str]:
    raw_names = getattr(model, "names", {})
    if isinstance(raw_names, dict):
        return [str(raw_names[index]) for index in sorted(raw_names, key=int)]
    return [str(name) for name in raw_names]


def main() -> None:
    parser = argparse.ArgumentParser(description="Export a YOLOv8 checkpoint to ONNX.")
    parser.add_argument("model", type=Path, help="Path to a trained .pt checkpoint.")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--opset", type=int, default=17)
    parser.add_argument("--simplify", action="store_true")
    parser.add_argument(
        "--dynamic", action="store_true", help="Export dynamic square image dimensions."
    )
    args = parser.parse_args()
    if not args.model.is_file():
        raise SystemExit(f"Checkpoint not found: {args.model}")
    from ultralytics import YOLO

    model = YOLO(str(args.model))
    exported = model.export(
        format="onnx",
        imgsz=args.imgsz,
        opset=args.opset,
        simplify=args.simplify,
        dynamic=args.dynamic,
    )
    exported_path = Path(exported)
    manifest_path = exported_path.with_suffix(".manifest.json")
    manifest = {
        "source_checkpoint_filename": args.model.name,
        "source_checkpoint_sha256": _sha256(args.model),
        "onnx_filename": exported_path.name,
        "class_names": _class_names(model),
        "export": {
            "imgsz": args.imgsz,
            "opset": args.opset,
            "simplify": args.simplify,
            "dynamic": args.dynamic,
        },
        "runtime": {
            "python": sys.version,
            "platform": platform.platform(),
            "torch": _package_version("torch"),
            "ultralytics": _package_version("ultralytics"),
        },
    }
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"Exported ONNX model: {exported_path}")
    print(f"Export manifest: {manifest_path}")


if __name__ == "__main__":
    main()
