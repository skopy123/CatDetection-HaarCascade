# syntax=docker/dockerfile:1.7

# ---- stage 1: export ONNX models (torch lives only here, never in the final image) ----
FROM python:3.12-slim AS models
ARG MODELS="yolo11s yolo11n dinov2_vits14_reg dinov2_vitb14_reg"
ARG OPTIONAL_MODELS="megadescriptor_t_224 dinov3_vits16"
RUN apt-get update && apt-get install -y --no-install-recommends git libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*
COPY --from=ghcr.io/astral-sh/uv:0.12 /uv /bin/uv
ENV UV_SYSTEM_PYTHON=1 UV_NO_CACHE=1 UV_PYTHON_DOWNLOADS=never
# CPU wheels keep this stage small (the default Linux torch wheels bundle CUDA)
RUN uv pip install --index-url https://download.pytorch.org/whl/cpu torch torchvision \
    && uv pip install ultralytics onnx onnxslim onnxruntime timm transformers numpy opencv-python-headless pyyaml pydantic
WORKDIR /build
COPY catalarm ./catalarm
# DINOv3 is gated on Hugging Face: pass a token as a build secret to include it, e.g.
#   docker build --secret id=hf_token,env=HF_TOKEN .
RUN --mount=type=secret,id=hf_token,required=false \
    HF_TOKEN="$(cat /run/secrets/hf_token 2>/dev/null || true)" \
    python -m catalarm.tools.export_models --out /models ${MODELS} --optional ${OPTIONAL_MODELS} \
    && rm -f /models/*.pt && ls -la /models

# ---- stage 2: runtime ----
FROM python:3.12-slim AS runtime
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*
# uv must be at least the version that wrote uv.lock
COPY --from=ghcr.io/astral-sh/uv:0.12 /uv /bin/uv
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy UV_NO_CACHE=1 UV_PYTHON_DOWNLOADS=never \
    PYTHONUNBUFFERED=1 \
    VIRTUAL_ENV=/app/.venv PATH="/app/.venv/bin:$PATH" \
    CATALARM_DATA_DIR=/data CATALARM_MODELS_DIR=/models \
    CATALARM_FRONTEND_DIR=/app/frontend CATALARM_SEED_TEMPLATES=/app/templates \
    OPENCV_FFMPEG_CAPTURE_OPTIONS="rtsp_transport;tcp" \
    PORT=8080
WORKDIR /app
COPY pyproject.toml uv.lock README.md LICENSE ./
RUN uv sync --frozen --no-dev --no-install-project
COPY catalarm ./catalarm
RUN uv sync --frozen --no-dev
COPY frontend ./frontend
# old reference pictures, imported into the gallery on first start
COPY templates/DayTemplates ./templates/DayTemplates
COPY templates/NightTemplates ./templates/NightTemplates
COPY testImg ./testImg
COPY --from=models /models /models

RUN useradd --system --uid 1000 --home /data catalarm && mkdir -p /data && chown catalarm /data
USER catalarm
VOLUME /data
EXPOSE 8080
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s \
    CMD python -c "import urllib.request,os; urllib.request.urlopen(f'http://127.0.0.1:{os.environ[\"PORT\"]}/api/v1/health', timeout=4)"
CMD ["catalarm"]
