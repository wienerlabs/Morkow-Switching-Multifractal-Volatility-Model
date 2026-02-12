# ---- Build stage: install dependencies ----
FROM python:3.11-slim AS builder

WORKDIR /build

COPY requirements.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt

# ---- Runtime stage: minimal image ----
FROM python:3.11-slim

LABEL maintainer="Cortex AI <dev@cortex-agent.xyz>"
LABEL org.opencontainers.image.source="https://github.com/cortex-agent/cortexagent"
LABEL org.opencontainers.image.description="CortexAgent Risk Engine — multi-model volatility and risk management API"

WORKDIR /app

# Copy installed packages from builder
COPY --from=builder /install /usr/local

# Copy application code
COPY api/ ./api/
COPY frontend/ ./frontend/
COPY tests/ ./tests/
COPY cortex/ ./cortex/
COPY pytest.ini .

ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]

