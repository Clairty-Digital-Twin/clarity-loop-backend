# CLARITY Digital Twin - Enterprise ML Health Platform
# Production-ready Docker image with all ML dependencies
FROM python:3.11-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    --no-install-recommends \
    curl \
    gcc \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements and install dependencies
COPY pyproject.toml LICENSE README.md ./
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -e .

# Copy source code
COPY src/ ./src/

# Create non-root user for security
RUN groupadd -r clarity && useradd -r -g clarity clarity && \
    chown -R clarity:clarity /app

# Switch to non-root user
USER clarity

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Expose port
EXPOSE 8000

# Production command - clean main.py
CMD ["uvicorn", "clarity.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]