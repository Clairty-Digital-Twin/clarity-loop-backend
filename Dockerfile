# syntax=docker/dockerfile:1

# Multi-stage Docker build for production-ready Clarity Loop Backend
# Optimized for Google Cloud Run deployment

# --- Base Stage ---
FROM python:3.11-slim as base

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

# Set work directory
WORKDIR /app

# Install poetry
RUN pip install poetry

# Copy only the dependency definitions to leverage Docker cache
COPY poetry.lock pyproject.toml /app/

# Install dependencies
RUN poetry config virtualenvs.create false && poetry install --no-dev --no-interaction --no-ansi


# --- Final Stage ---
FROM python:3.11-slim as final

# Set work directory
WORKDIR /app

# Copy installed dependencies from base stage
COPY --from=base /app /app
COPY --from=base /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages

# Copy the application source
COPY src/clarity /app/src/clarity

# Expose the port the app runs on
EXPOSE 8000

# Command to run the application
CMD ["uvicorn", "src.clarity.main:app", "--host", "0.0.0.0", "--port", "8000"]
