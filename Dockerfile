FROM python:3.11-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

RUN apt-get update \
    && apt-get install --no-install-recommends -y libglib2.0-0 libgl1 \
    && rm -rf /var/lib/apt/lists/* \
    && groupadd --gid 10001 appuser \
    && useradd --uid 10001 --gid appuser --create-home --shell /usr/sbin/nologin appuser

COPY pyproject.toml README.md LICENSE NOTICE.md requirements.txt ./
COPY src ./src
RUN pip install --upgrade pip && pip install ".[all]"

COPY . .
RUN chown -R appuser:appuser /app

USER appuser

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --retries=3 CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/readyz', timeout=3)"]
CMD ["uvicorn", "industrial_defect_detection.api:app", "--host", "0.0.0.0", "--port", "8000"]

