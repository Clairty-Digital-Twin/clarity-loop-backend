# syntax=docker/dockerfile:1

# Multi-stage Docker build for production-ready Clarity Loop Backend
# Optimized for Google Cloud Run deployment

# --- Base Stage ---
# This stage installs all dependencies, including the project itself in editable mode.
# This creates a complete environment with all necessary packages in site-packages.
FROM python:3.11-slim AS base

# Set environment variables for a clean Python environment
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Set the working directory
WORKDIR /app

# Install system dependencies required for building some Python packages
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy the entire project context
COPY . /app/

# Install the package and its dependencies.
# We install without -e to ensure the package is properly installed in site-packages.
RUN pip install --no-cache-dir .


# --- Final Stage ---
# This stage creates the final, lean image for production.
FROM python:3.11-slim AS final

# Set the working directory
WORKDIR /app

# IMPORTANT: Copy the fully installed site-packages from the base stage.
# This includes our 'clarity' application and all its dependencies, properly installed.
COPY --from=base /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages

# Copy Python binaries (including gunicorn and other installed scripts)
COPY --from=base /usr/local/bin /usr/local/bin

# Copy the gunicorn config and the entrypoint script.
# We do NOT copy the 'src' directory here to avoid path conflicts. The code is already
# in site-packages from the step above.
COPY gunicorn.conf.py entrypoint.sh /app/

# Make the entrypoint script executable
RUN chmod +x /app/entrypoint.sh

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
# PYTHONPATH is no longer needed as everything is in the standard site-packages directory.

# Expose the port the app runs on
EXPOSE 8000

# Set the entrypoint to our script to run the application.
ENTRYPOINT ["/app/entrypoint.sh"]