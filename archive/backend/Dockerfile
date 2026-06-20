FROM python:3.13-slim AS base

# libsndfile is required by librosa/soundfile for the post-call paralinguistic pass.
RUN apt-get update && apt-get install -y --no-install-recommends \
        libsndfile1 ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY pyproject.toml ./
# Install runtime deps only (dev extras run in CI).
RUN pip install --no-cache-dir --upgrade pip && pip install --no-cache-dir .

COPY . .

EXPOSE 8000 50051
# uvicorn serves FastAPI (/media WS); the gRPC server is started in the app lifespan.
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--no-access-log"]
