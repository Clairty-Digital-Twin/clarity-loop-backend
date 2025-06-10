# Multi-stage Dockerfile optimized for AWS ECS/Fargate deployment
# Fixes the module import issues and provides proper health checks

FROM python:3.11-slim as builder

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /build

# Copy package files first for better caching
COPY pyproject.toml README.md LICENSE ./
COPY src ./src

# Upgrade pip and install build tools
RUN pip install --upgrade pip setuptools wheel hatchling

# Build the wheel
RUN pip wheel --no-cache-dir --wheel-dir /wheels .

# Final stage - minimal runtime image
FROM python:3.11-slim

# Install runtime dependencies only
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user for security
RUN useradd -m -u 1000 clarity && \
    mkdir -p /app && \
    chown -R clarity:clarity /app

WORKDIR /app

# Copy wheels from builder
COPY --from=builder /wheels /wheels

# Install the application and dependencies
RUN pip install --upgrade pip && \
    pip install --no-cache-dir /wheels/*.whl && \
    rm -rf /wheels

# Copy runtime configuration files
COPY --chown=clarity:clarity gunicorn.conf.py entrypoint.sh ./

# Make entrypoint executable
RUN chmod +x entrypoint.sh

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PORT=8000 \
    ENVIRONMENT=production

# Switch to non-root user
USER clarity

# Health check for AWS ALB
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:${PORT}/health || exit 1

# Expose port
EXPOSE 8000

# Run the application
ENTRYPOINT ["./entrypoint.sh"]