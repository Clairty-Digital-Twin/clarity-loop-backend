"""Insight Service Entry Point.

Standalone FastAPI service for AI-powered health insight generation.
Handles Pub/Sub push subscriptions for async insight generation using Gemini.
"""

import asyncio
import logging
import os

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

from clarity.services.pubsub.insight_subscriber import insight_app

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title="CLARITY Insight Service",
    description="AI-powered health insight generation service",
    version="1.0.0"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount the insight app
app.mount("/", insight_app)


def main():
    """Run the insight service."""
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "8082"))

    logger.info("Starting CLARITY Insight Service")
    logger.info(f"Listening on {host}:{port}")

    # Run the service
    uvicorn.run(
        "clarity.entrypoints.insight_service:app",
        host=host,
        port=port,
        reload=os.getenv("ENVIRONMENT") == "development",
        log_level="info"
    )


if __name__ == "__main__":
    main()
