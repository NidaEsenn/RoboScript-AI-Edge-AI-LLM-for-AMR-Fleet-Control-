# Multi-stage image for the RoboScript FastAPI routing layer (PRD 6.9 / AC-09).
# Model weights are NOT baked in: the API talks to vLLM (ROBOSCRIPT_BACKEND=vllm) or
# loads a small edge model from a mounted volume / S3 sync at start-up.
FROM python:3.11-slim AS builder
WORKDIR /build
COPY requirements-serve.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements-serve.txt

FROM python:3.11-slim AS runtime
ENV PYTHONUNBUFFERED=1 ROBOSCRIPT_BACKEND=vllm PORT=8000
WORKDIR /app
COPY --from=builder /install /usr/local
COPY roboscript/ ./roboscript/
COPY data/seed/ ./data/seed/
RUN useradd -m app && chown -R app /app
USER app
EXPOSE 8000
HEALTHCHECK --interval=15s --timeout=3s CMD python -c "import urllib.request,sys; urllib.request.urlopen('http://localhost:8000/v1/health'); sys.exit(0)"
# SageMaker invokes the container with "serve"; both paths start uvicorn.
ENTRYPOINT ["python", "-m", "roboscript.deploy.entrypoint"]
