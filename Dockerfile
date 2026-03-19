# ── Build stage ────────────────────────────────────────────────
FROM python:3.11-slim AS builder

WORKDIR /app

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

# ── Runtime stage ───────────────────────────────────────────────
FROM python:3.11-slim AS runtime

WORKDIR /app

# Runtime system dependencies only
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy installed packages from builder
COPY --from=builder /root/.local /root/.local

# Copy application source
COPY src/ ./src/
COPY config/ ./config/
COPY scripts/ ./scripts/
COPY data/samples/ ./data/samples/

# Create directories for runtime artefacts
RUN mkdir -p data/faiss_index data/eval_results

# Make scripts executable
RUN chmod +x scripts/*.py

# Non-root user for security
RUN useradd -m -u 1000 aiops && chown -R aiops:aiops /app
USER aiops

ENV PATH=/root/.local/bin:$PATH
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
  CMD python -c "import httpx; httpx.get('http://localhost:8000/v1/health').raise_for_status()" || exit 1

CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]
