# Multi-stage Docker build for production-ready Clarity Loop Backend
# Optimized for Google Cloud Run deployment

# Build stage for dependencies
FROM python:3.11-slim AS builder

# Set environment variables for build
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install system dependencies for building
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Create virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy dependency files
COPY pyproject.toml ./
COPY README.md ./
COPY LICENSE ./

# Install Python dependencies
RUN pip install --upgrade pip setuptools wheel
RUN pip install -e .[dev]

# Production stage
FROM python:3.11-slim AS production

# Set production environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PATH="/opt/venv/bin:$PATH" \
    PORT=8080 \
    WORKERS=4 \
    MAX_WORKERS=8 \
    TIMEOUT=120 \
    KEEP_ALIVE=5

# Install runtime system dependencies
RUN apt-get update && apt-get install -y \
    ca-certificates \
    curl \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Copy virtual environment from builder
COPY --from=builder /opt/venv /opt/venv

# Create non-root user for security
RUN groupadd -r clarity && useradd -r -g clarity clarity

# Set working directory
WORKDIR /app

# Copy application code
COPY --chown=clarity:clarity . .

# Switch to non-root user
USER clarity

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:${PORT}/health || exit 1

# Expose port
EXPOSE ${PORT}

# Production startup command
CMD ["sh", "-c", "exec uvicorn clarity.main:app --host 0.0.0.0 --port ${PORT} --workers ${WORKERS} --loop uvloop --http httptools --access-log --log-level info --timeout-keep-alive ${KEEP_ALIVE} --timeout-graceful-shutdown ${TIMEOUT}"]
