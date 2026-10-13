# Build context: repository root (the image also needs data/curation).
#   docker build -f api/Dockerfile -t forglemmigej-api .
FROM ghcr.io/astral-sh/uv:0.13.0 AS uv

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

FROM python:3.13-slim-trixie
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
