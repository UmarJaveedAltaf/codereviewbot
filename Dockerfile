# ── Stage 1: builder ──────────────────────────────────────────────────────────
# Installs all Python dependencies into a virtual-env so the runtime stage
# can simply copy the venv — no build tools needed in the final image.
FROM python:3.11-slim AS builder

# System packages needed to compile tree-sitter and chromadb C extensions.
RUN apt-get update \
    && apt-get install -y --no-install-recommends gcc g++ \
    && rm -rf /var/lib/apt/lists/*

# Isolated virtual-env — keeps site-packages clean and portable.
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

WORKDIR /build
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt


# ── Stage 2: runtime ──────────────────────────────────────────────────────────
# Slim final image — only the venv + application code, no compilers.
FROM python:3.11-slim AS runtime

# ── Non-root user (principle of least privilege) ──────────────────────────────
RUN groupadd -r botuser \
    && useradd  -r -g botuser -d /app -s /sbin/nologin -c "CodeReviewBot" botuser

# ── Copy installed packages from builder ──────────────────────────────────────
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# ── Python environment tunables ───────────────────────────────────────────────
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# ── Application working directory ─────────────────────────────────────────────
WORKDIR /app

# Copy application source — explicitly listed so .dockerignore is a backstop,
# not the primary guard. .env, .chroma, __pycache__ are never copied.
COPY --chown=botuser:botuser agent/          agent/
COPY --chown=botuser:botuser config.py       config.py
COPY --chown=botuser:botuser conventions.yaml conventions.yaml
COPY --chown=botuser:botuser dashboard/      dashboard/
COPY --chown=botuser:botuser frontend/       frontend/
COPY --chown=botuser:botuser integrations/   integrations/
COPY --chown=botuser:botuser main.py         main.py
COPY --chown=botuser:botuser memory/         memory/
COPY --chown=botuser:botuser parser/         parser/
COPY --chown=botuser:botuser scripts/        scripts/
COPY --chown=botuser:botuser webhook/        webhook/

# ChromaDB persistence directory (overridden to /data/chroma in docker-compose
# so the volume mount lands in the right place).
ENV CHROMA_PERSIST_DIR=/data/chroma
RUN mkdir -p /data/chroma && chown botuser:botuser /data/chroma

USER botuser

EXPOSE 8000

# Health check uses Python's built-in urllib — no curl required.
HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
    CMD python -c "\
import urllib.request, sys; \
try: \
    r = urllib.request.urlopen('http://localhost:8000/health', timeout=4); \
    sys.exit(0 if r.status == 200 else 1) \
except Exception: \
    sys.exit(1)"

# Default command — streamlit service overrides this in docker-compose.
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]
