# Multi-stage Docker build for production-ready Clarity Loop Backend
# Optimized for Google Cloud Run deployment

# Build stage for dependencies
FROM python:3.11-slim AS builder

WORKDIR /app

# Set environment variables for build
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install system dependencies for building
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential=12.9 \
    curl=7.88.1-10+deb12u12 \
    git=1:2.39.5-0+deb12u2 \
    && rm -rf /var/lib/apt/lists/*

# Create virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy package configuration first for better caching
COPY pyproject.toml README.md LICENSE ./

# Install Python dependencies first (without source code for better caching)
# Pinning pip, setuptools, and wheel to recent stable versions
RUN pip install --upgrade pip==24.0 setuptools==69.5.1 wheel==0.43.0

# Copy source code for installation
COPY src/ ./src/

# Install the package in development mode (dependencies are pinned in pyproject.toml)
RUN pip install -e .

# Production stage
FROM python:3.11-slim AS production

# Set production environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PATH="/opt/venv/bin:$PATH" \
    PYTHONPATH="/app/src" \
    PORT=8080 \
    WORKERS=4 \
    MAX_WORKERS=8 \
    TIMEOUT=120 \
    KEEP_ALIVE=5

# Install runtime system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates=20230311 \
    curl=7.88.1-10+deb12u12 \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Copy virtual environment from builder
COPY --from=builder /opt/venv /opt/venv

# Create non-root user for security
RUN groupadd -r clarity && useradd -r -g clarity clarity

# Set working directory
WORKDIR /app

# Create logs directory
RUN mkdir -p logs

# Copy application code with proper structure
COPY --chown=clarity:clarity src/ ./src/
COPY --chown=clarity:clarity main.py ./
COPY --chown=clarity:clarity pyproject.toml README.md LICENSE ./

# Fix logs directory ownership
RUN chown -R clarity:clarity /app/logs

# Switch to non-root user
USER clarity

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:${PORT}/health || exit 1

# Expose port
EXPOSE ${PORT}

# Production startup command - use the root main.py entry point
CMD ["sh", "-c", "exec uvicorn main:app --host 0.0.0.0 --port ${PORT} --workers ${WORKERS} --loop uvloop --http httptools --access-log --log-level info --timeout-keep-alive ${KEEP_ALIVE} --timeout-graceful-shutdown ${TIMEOUT}"]
