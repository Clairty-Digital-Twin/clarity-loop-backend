# syntax=docker/dockerfile:1

# Multi-stage Docker build for production-ready Clarity Loop Backend
# Optimized for Google Cloud Run deployment

# --- Base Stage ---
FROM python:3.11-slim AS base

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Set work directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy the entire project (we need pyproject.toml for pip install)
COPY . /app/

# Install the package using pip directly with pyproject.toml
# This avoids Poetry's dependency resolution issues
RUN pip install --no-cache-dir -e .

# --- Final Stage ---
FROM python:3.11-slim AS final

# Set work directory
WORKDIR /app

# Copy Python binaries and packages from base stage
COPY --from=base /usr/local/bin /usr/local/bin
COPY --from=base /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages

# Copy application files
COPY pyproject.toml LICENSE README.md /app/
COPY src /app/src
COPY gunicorn.conf.py entrypoint.sh /app/

# Make entrypoint executable
RUN chmod +x /app/entrypoint.sh

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app/src:$PYTHONPATH

# Expose the port the app runs on
EXPOSE 8000

# Set the entrypoint to our script
ENTRYPOINT ["/app/entrypoint.sh"]