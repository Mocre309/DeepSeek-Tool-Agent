# syntax=docker/dockerfile:1
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy

# uv binary from the official image — no pip bootstrap, reproducible installs.
COPY --from=ghcr.io/astral-sh/uv:0.9 /uv /uvx /bin/

WORKDIR /app

# Dependency layer first so code changes do not invalidate it.
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project

COPY app ./app
COPY README.md ./

# Non-root runtime user.
RUN useradd --create-home --uid 10001 appuser \
    && mkdir -p /app/data \
    && chown -R appuser:appuser /app
USER appuser

ENV PATH="/app/.venv/bin:$PATH" \
    NOTES_DB_PATH=/app/data/notes.sqlite3 \
    APP_HOST=0.0.0.0 \
    APP_PORT=8000

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import os,httpx,sys; port=os.environ.get('APP_PORT','8000'); sys.exit(0 if httpx.get(f'http://127.0.0.1:{port}/api/health', timeout=3).status_code == 200 else 1)"

# Same entry point as local: host/port come from APP_HOST / APP_PORT.
CMD ["python", "-m", "app"]
