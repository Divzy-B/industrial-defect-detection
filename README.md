# Industrial Defect Detection

[![CI](https://github.com/Divzy-B/industrial-defect-detection/actions/workflows/ci.yml/badge.svg)](https://github.com/Divzy-B/industrial-defect-detection/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

An end-to-end, deployment-ready template for detecting industrial surface defects
with **YOLOv8**. It offers a FastAPI inference API, a Streamlit inspection
dashboard, PyTorch/Ultralytics and ONNX Runtime backends, OpenCV annotation,
Docker packaging, and GitHub Actions CI.

> **Safety note:** this is decision-support software. Validate it on your own
> production lines and retain human quality-release controls. Read the
> [model card](MODEL_CARD.md) before deployment.

## Highlights

- One detector contract with two engines: YOLOv8/PyTorch and CPU-friendly ONNX Runtime.
- Typed FastAPI endpoint with OpenAPI docs at `/docs`, image-size limits, and useful health status.
- Streamlit UI for operators: upload, annotate, and inspect structured detections.
- OpenCV image validation, letterboxing, rendering, and non-maximum suppression.
- Reproducible training and ONNX export commands; trained weights and data stay out of Git.
- Docker/Compose configuration and a Python 3.10–3.12 GitHub Actions matrix.

## Architecture

```text
Camera / uploaded image
          │
          ▼
 OpenCV decode + validation
          │
          ▼
 ┌───────────────────────────┐
 │ YOLOv8 PyTorch │ YOLOv8 ONNX │
 └──────────────┬────────────┘
                ▼
  boxes · labels · confidence
          │                 │
          ▼                 ▼
     FastAPI JSON      Streamlit + annotated image
```

## Supported classes

The defaults match a common NEU steel-surface taxonomy: `crazing`, `inclusion`,
`patches`, `pitted_surface`, `rolled-in_scale`, and `scratches`. They are only
defaults—set `DEFECT_CLASS_NAMES` when your training data differs.

## Quick start

Python 3.10+ is required. The inference dependencies include PyTorch and can be
large; use an environment appropriate for your CPU/GPU and operating system.

```bash
git clone https://github.com/Divzy-B/industrial-defect-detection.git
cd industrial-defect-detection
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
pip install -e ".[all,dev]"
copy .env.example .env  # macOS/Linux: cp .env.example .env
```

Place a model at `models/best.pt` (or change `DEFECT_MODEL_PATH` in `.env`), then:

```bash
uvicorn industrial_defect_detection.api:app --reload
# Separate terminal
streamlit run src/industrial_defect_detection/streamlit_app.py
```

Visit `http://localhost:8000/docs` for the API and `http://localhost:8501` for
the dashboard.

## API

```bash
curl -X POST "http://localhost:8000/v1/detect?annotate=true" \
  -F "image=@surface.jpg"
```

Example response (boxes are in the original image's pixels):

```json
{
  "image_width": 1920,
  "image_height": 1080,
  "model_backend": "ultralytics",
  "detections": [{
    "class_id": 5,
    "label": "scratches",
    "confidence": 0.91,
    "box": {"x1": 614.2, "y1": 418.7, "x2": 821.9, "y2": 495.4}
  }],
  "annotated_image_base64": "..."
}
```

`GET /healthz` remains available even if the model is missing, reporting a
`degraded` state and actionable configuration detail. Images default to a 10 MiB
limit; change `DEFECT_MAX_UPLOAD_BYTES` only after assessing operational risk.

## Backends

| Variable | PyTorch / Ultralytics | ONNX Runtime |
| --- | --- | --- |
| `DEFECT_BACKEND` | `ultralytics` | `onnx` |
| `DEFECT_MODEL_PATH` | `models/best.pt` | `models/best.onnx` |
| Runtime | PyTorch through Ultralytics | ONNX Runtime CPU provider |
| Best fit | training, GPU/cloud inference | portable CPU/edge inference |

For ONNX, export the same checkpoint:

```bash
python scripts/export_onnx.py models/best.pt --imgsz 640 --simplify
set DEFECT_BACKEND=onnx  # Windows PowerShell: $env:DEFECT_BACKEND = "onnx"
```

The ONNX adapter accepts standard YOLOv8 detection exports in `(1, 4+C, N)` or
`(1, N, 4+C)` layout, applies letterbox reversal, confidence filtering, and
class-aware rendering after OpenCV NMS.

## Train

Prepare an Ultralytics-format dataset outside the repository, review its license,
then adapt [`data/neu_steel.yaml`](data/neu_steel.yaml):

```bash
python scripts/train.py --data /absolute/path/to/data.yaml --epochs 80 --imgsz 640 --batch 16
```

Ultralytics writes results below `runs/train/`; promote only a reviewed `best.pt`
to your model registry or mounted `models/` directory. Never publish plant images
or model artifacts without checking confidentiality and data permissions.

## Docker

Create `.env` and place the selected model in `models/` first:

```bash
docker compose up --build
```

- API: `http://localhost:8000/docs`
- Dashboard: `http://localhost:8501`

The bundled image targets CPU inference. For NVIDIA deployment, use a CUDA base
image and validate the matching PyTorch, driver, and container-runtime versions.

## Quality checks

```bash
pip install -e ".[dev]"
ruff check .
pytest
```

CI runs these checks on Python 3.10, 3.11, and 3.12. Tests use a fake detector;
they verify request validation, response shape, OpenCV image handling, annotation,
and ONNX-output decoding without bundling a proprietary model.

## Repository layout

```text
src/industrial_defect_detection/  # API, inference adapters, service, UI, image utilities
scripts/                          # training and ONNX export entry points
tests/                            # fast unit/API tests with no trained weights
data/neu_steel.yaml               # dataset-config template
models/                           # intentionally empty; mount/copy approved models here
.github/workflows/ci.yml          # lint and test matrix
```

## Attribution and license

This project is MIT licensed; see [LICENSE](LICENSE). It was independently built
with the public [SatvikSPandey/defect-detection](https://github.com/SatvikSPandey/defect-detection)
repository as a product reference. That repository had no visible `LICENSE` file
at the time of review, so no implementation, weights, images, or text were
copied. The full attribution and dataset guidance is in [NOTICE.md](NOTICE.md).

The referenced project identifies the NEU Steel Surface Defect Dataset as
CC BY 4.0. This repository does not redistribute the dataset; if you use it,
preserve the attribution and verify the provider's current terms.

