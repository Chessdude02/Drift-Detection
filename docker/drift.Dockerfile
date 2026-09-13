# syntax=docker/dockerfile:1
FROM python:3.11-slim AS builder

WORKDIR /build
COPY requirements.txt requirements-drift.txt ./
RUN pip install --no-cache-dir --user -r requirements-drift.txt

FROM python:3.11-slim

RUN useradd --create-home --shell /bin/false appuser
WORKDIR /app

COPY --from=builder /root/.local /home/appuser/.local
COPY src/pdm ./pdm
COPY config ./config

ENV PATH=/home/appuser/.local/bin:$PATH \
    PYTHONPATH=/app \
    PYTHONUNBUFFERED=1

RUN mkdir -p /app/data && chown -R appuser:appuser /app
USER appuser

ENTRYPOINT ["python", "-m", "pdm.drift.run_drift_check"]
