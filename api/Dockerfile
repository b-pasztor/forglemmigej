# Build context: repository root (the image also needs data/curation).
#   docker build -f api/Dockerfile -t forglemmigej-api .
#   docker build -f api/Dockerfile --target trainer -t forglemmigej-trainer .
FROM ghcr.io/astral-sh/uv:0.13.0 AS uv

# --- API: runtime dependencies only ------------------------------------------
FROM python:3.13-slim-trixie AS build
COPY --from=uv /uv /usr/local/bin/uv
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy UV_PYTHON_DOWNLOADS=never
WORKDIR /app
# Dependencies first, so code changes don't invalidate this layer.
COPY api/pyproject.toml api/uv.lock ./
RUN uv sync --locked --no-dev --no-install-project
COPY api/README.md ./
COPY api/src ./src
RUN uv sync --locked --no-dev --no-editable

# --- Trainer: plus the ml group (pandas, LightGBM, CPU-only PyTorch, MLflow) --
FROM python:3.13-slim-trixie AS build-trainer
COPY --from=uv /uv /usr/local/bin/uv
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy UV_PYTHON_DOWNLOADS=never
WORKDIR /app
COPY api/pyproject.toml api/uv.lock ./
RUN uv sync --locked --no-dev --group ml --no-install-project
COPY api/README.md ./
COPY api/src ./src
RUN uv sync --locked --no-dev --group ml --no-editable

FROM python:3.13-slim-trixie AS trainer
# LightGBM and PyTorch use OpenMP.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --system --uid 10001 --create-home app
WORKDIR /app
COPY --from=build-trainer /app/.venv /app/.venv
ENV PATH=/app/.venv/bin:$PATH PYTHONUNBUFFERED=1 MLFLOW_DISABLE_AGENT_HINT=1 HOME=/home/app
USER 10001
CMD ["python", "-m", "forglemmigej_api.training.nightly"]

# --- API runtime image (the default target) ------------------------------------
FROM python:3.13-slim-trixie AS api
RUN useradd --system --uid 10001 --no-create-home app
WORKDIR /app
COPY --from=build /app/.venv /app/.venv
COPY api/alembic.ini ./
COPY api/migrations ./migrations
COPY data/curation ./curation
ENV PATH=/app/.venv/bin:$PATH PYTHONUNBUFFERED=1
USER 10001
EXPOSE 8000
CMD ["uvicorn", "forglemmigej_api.app:app", "--host", "0.0.0.0", "--port", "8000", "--proxy-headers"]
