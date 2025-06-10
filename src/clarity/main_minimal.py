"""Minimal CLARITY backend for AWS deployment - bypasses complex imports."""

from contextlib import asynccontextmanager
import logging
from typing import Any, Dict

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from prometheus_client import make_asgi_app

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan management."""
    logger.info("Starting minimal CLARITY backend...")

    # Startup
    yield

    # Shutdown
    logger.info("Shutting down minimal CLARITY backend...")


# Create FastAPI app
app = FastAPI(
    title="CLARITY Digital Twin Platform",
    description="Health AI platform with minimal AWS configuration",
    version="0.1.0",
    lifespan=lifespan,
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify your frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def root() -> dict[str, str]:
    """Root endpoint."""
    return {
        "message": "Welcome to CLARITY Digital Twin Platform",
        "version": "0.1.0",
        "mode": "minimal",
    }


@app.get("/health")
async def health_check() -> dict[str, Any]:
    """Health check endpoint."""
    return {
        "status": "ok",
        "environment": "aws-minimal",
        "services": {"auth": "disabled", "database": "disabled", "ai": "disabled"},
    }


@app.get("/api/v1/test")
async def test_endpoint() -> dict[str, str]:
    """Test API endpoint."""
    return {"status": "ok", "message": "API is working"}


# Prometheus metrics endpoint
metrics_app = make_asgi_app()
app.mount("/metrics", metrics_app)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
