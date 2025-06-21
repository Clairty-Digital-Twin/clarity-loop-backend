# CLARITY Digital Twin - AWS Production Docker Image
# BULLETPROOF Multi-Architecture Build for ECS Fargate
# CRITICAL: Always build with --platform linux/amd64 for AWS ECS

# Stage 1: Builder
FROM --platform=$BUILDPLATFORM python:3.11-slim AS builder

ARG TARGETARCH
ARG BUILDPLATFORM

# Install build dependencies
RUN apt-get update && apt-get install -y \
    --no-install-recommends \
    gcc \
    python3-dev \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build

# Copy dependency files first for better caching
COPY pyproject.toml LICENSE README.md ./
COPY src/ ./src/

# Install build tools and create wheel
ENV PIP_DEFAULT_TIMEOUT=1000
RUN pip install --no-cache-dir --upgrade pip wheel setuptools build && \
    pip config set global.timeout 1000 && \
    python -m build --wheel --outdir /build/dist

# Stage 2: Runtime
FROM python:3.11-slim

ARG TARGETARCH

# Install runtime dependencies including AWS CLI (ARCHITECTURE-AWARE)
RUN apt-get update && apt-get install -y \
    --no-install-recommends \
    curl \
    unzip \
    && echo "Building for architecture: ${TARGETARCH}" \
    && if [ "${TARGETARCH}" = "amd64" ]; then \
         AWS_CLI_ARCH="x86_64"; \
       elif [ "${TARGETARCH}" = "arm64" ]; then \
         AWS_CLI_ARCH="aarch64"; \
       else \
         AWS_CLI_ARCH="x86_64"; \
       fi \
    && echo "Using AWS CLI architecture: ${AWS_CLI_ARCH}" \
    && curl -fsSL "https://awscli.amazonaws.com/awscli-exe-linux-${AWS_CLI_ARCH}.zip" -o "awscliv2.zip" \
    && unzip -q awscliv2.zip \
    && ./aws/install \
    && rm -rf awscliv2.zip aws \
    && apt-get remove -y unzip \
    && apt-get autoremove -y \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN groupadd -r clarity && useradd -r -g clarity -u 1000 clarity

WORKDIR /app

# Copy wheel from builder and install
COPY --from=builder /build/dist/*.whl /tmp/
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir /tmp/*.whl && \
    rm -rf /tmp/*.whl

# Copy application code and scripts
COPY --chown=clarity:clarity pyproject.toml ./
COPY --chown=clarity:clarity src/ ./src/
COPY --chown=clarity:clarity gunicorn.aws.conf.py ./
COPY --chown=clarity:clarity scripts/download_models.sh scripts/entrypoint.sh ./scripts/

# Create necessary directories
RUN mkdir -p /app/models/pat && \
    chmod +x ./scripts/download_models.sh ./scripts/entrypoint.sh && \
    chown -R clarity:clarity /app

# Switch to non-root user
USER clarity

# Set environment variables for production
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PORT=8000 \
    ENVIRONMENT=production \
    AWS_PAGER=""

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Expose port
EXPOSE 8000

# Use entrypoint script to download models before starting
ENTRYPOINT ["/app/scripts/entrypoint.sh"]