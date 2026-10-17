"""Train a YOLOv8 model from a local Ultralytics dataset definition."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import sys
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
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
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument(
        "--deterministic",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable deterministic operations where the selected runtime supports them.",
    )
    parser.add_argument(
        "--device", default=None, help="CUDA device, 'cpu', or omit for Ultralytics default."
    )
    return parser.parse_args(argv)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _git_commit() -> str | None:
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return completed.stdout.strip() or None


def _package_version(name: str) -> str | None:
    try:
        return version(name)
    except PackageNotFoundError:
        return None


def _write_manifest(save_dir: Path, args: argparse.Namespace) -> Path:
    """Persist the non-sensitive inputs needed to reproduce a training run."""
    manifest = {
        "data_yaml_filename": args.data.name,
        "data_yaml_sha256": _sha256(args.data),
        "git_commit": _git_commit(),
        "model": args.model,
        "training": {
            "epochs": args.epochs,
            "imgsz": args.imgsz,
            "batch": args.batch,
            "device": args.device,
            "seed": args.seed,
            "workers": args.workers,
            "patience": args.patience,
            "deterministic": args.deterministic,
        },
        "runtime": {
            "python": sys.version,
            "platform": platform.platform(),
            "torch": _package_version("torch"),
            "ultralytics": _package_version("ultralytics"),
        },
    }
    path = save_dir / "training_manifest.json"
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


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
        patience=args.patience,
        seed=args.seed,
        workers=args.workers,
        deterministic=args.deterministic,
        val=True,
    )
    manifest_path = _write_manifest(Path(model.trainer.save_dir), args)
    print(f"Training manifest: {manifest_path}")


if __name__ == "__main__":
    main()
