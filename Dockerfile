# CLARITY Digital Twin - Enterprise ML Health Platform
# Production-ready Docker image with all ML dependencies
FROM python:3.11-slim

# Install system dependencies including AWS CLI
RUN apt-get update && apt-get install -y \
    --no-install-recommends \
    curl \
    gcc \
    python3-dev \
    unzip \
    && curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip" \
    && unzip awscliv2.zip \
    && ./aws/install \
    && rm -rf awscliv2.zip aws \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy the entire project for proper package installation
COPY pyproject.toml LICENSE README.md ./
COPY src/ ./src/

# Install dependencies and the package
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -e .

# Copy Gunicorn configuration and scripts
COPY gunicorn.aws.conf.py ./
COPY scripts/download_models.sh scripts/entrypoint.sh ./scripts/

# Create non-root user for security and prepare directories
RUN groupadd -r clarity && useradd -r -g clarity clarity && \
    mkdir -p /app/models/pat && \
    chmod +x ./scripts/download_models.sh ./scripts/entrypoint.sh && \
    chown -R clarity:clarity /app

# Switch to non-root user
USER clarity

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Expose port
EXPOSE 8000

# Use entrypoint script to download models before starting
ENTRYPOINT ["/app/scripts/entrypoint.sh"]