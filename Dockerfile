# Multi-stage build: dependencies are assembled separately, then copied into a slim runtime.
FROM python:3.13-slim AS builder
ENV PIP_DISABLE_PIP_VERSION_CHECK=1 PYTHONDONTWRITEBYTECODE=1
WORKDIR /build
COPY requirements.txt .
RUN python -m pip install --no-cache-dir --prefix=/install -r requirements.txt

FROM python:3.13-slim AS runtime
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PORT=8000 MODEL_PATH=/app/artifacts/model_bundle.joblib
WORKDIR /app
RUN groupadd --system appuser && useradd --system --gid appuser --home-dir /app appuser
COPY --from=builder /install /usr/local
COPY app ./app
RUN mkdir -p /app/artifacts && chown appuser:appuser /app/artifacts
EXPOSE 8000
USER appuser
HEALTHCHECK --interval=30s --timeout=3s --start-period=15s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=2)" || exit 1
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--proxy-headers"]
