# syntax=docker/dockerfile:1
FROM python:3.11-slim AS builder

WORKDIR /build
COPY requirements.txt requirements-train.txt ./
RUN pip install --no-cache-dir --user -r requirements-train.txt

FROM python:3.11-slim

RUN useradd --create-home --shell /bin/false appuser
WORKDIR /app

COPY --from=builder /root/.local /home/appuser/.local
COPY src/pdm ./pdm
COPY config ./config

ENV PATH=/home/appuser/.local/bin:$PATH \
    PYTHONPATH=/app \
    PYTHONUNBUFFERED=1

RUN mkdir -p /app/mlruns /app/data && chown -R appuser:appuser /app
USER appuser

ENTRYPOINT ["python", "-m", "pdm.training.train"]
CMD ["--raw-dir", "data/raw", "--config-name", "training.yaml", "--fail-on-gate"]
