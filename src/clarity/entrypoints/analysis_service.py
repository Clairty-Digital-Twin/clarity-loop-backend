"""Analysis Service Entry Point.

Standalone FastAPI service for health data analysis processing.
Handles Pub/Sub push subscriptions for async health data processing.
"""

import asyncio
import logging
import os

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

from clarity.services.pubsub.analysis_subscriber import analysis_app

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title="CLARITY Analysis Service",
    description="Health data analysis processing service",
    version="1.0.0",
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount the analysis app
app.mount("/", analysis_app)


def main() -> None:
    """Run the analysis service."""
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "8081"))

    logger.info("Starting CLARITY Analysis Service")
    logger.info(f"Listening on {host}:{port}")

    # Run the service
    uvicorn.run(
        "clarity.entrypoints.analysis_service:app",
        host=host,
        port=port,
        reload=os.getenv("ENVIRONMENT") == "development",
        log_level="info",
    )


if __name__ == "__main__":
    main()
