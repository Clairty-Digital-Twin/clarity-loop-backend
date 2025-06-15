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

# Copy only dependency files first for better caching
COPY pyproject.toml LICENSE README.md ./

# Install dependencies first (this layer will be cached if deps don't change)
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir toml && \
    python -c "import toml; deps = toml.load('pyproject.toml')['project']['dependencies']; print('\n'.join(deps))" > requirements.txt && \
    pip install --no-cache-dir -r requirements.txt

# Now copy source code and install in editable mode
COPY src/ ./src/
RUN pip install --no-cache-dir -e .

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