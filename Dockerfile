FROM python:3.11-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

RUN apt-get update \
    && apt-get install --no-install-recommends -y libglib2.0-0 libgl1 \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml requirements.txt ./
COPY src ./src
RUN pip install --upgrade pip && pip install ".[all]"

COPY . .

EXPOSE 8000
CMD ["uvicorn", "industrial_defect_detection.api:app", "--host", "0.0.0.0", "--port", "8000"]

