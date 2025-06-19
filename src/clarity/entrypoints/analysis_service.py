"""Analysis Service Entry Point.

Standalone FastAPI service for health data analysis processing.
Handles Pub/Sub push subscriptions for async health data processing.
"""

# removed - breaks FastAPI

import logging
import os

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

from clarity.services.messaging.analysis_subscriber import analysis_app

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

# Configure CORS - HARDENED SECURITY (NO WILDCARDS!)
from clarity.core.config import get_settings  # noqa: E402

settings = get_settings()

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.get_cors_origins,  # ✅ EXPLICIT ORIGINS ONLY
    allow_credentials=True,  # ✅ SAFE WITH EXPLICIT ORIGINS
    allow_methods=["GET", "POST", "PUT", "DELETE"],  # ✅ SPECIFIC METHODS ONLY
    allow_headers=["Authorization", "Content-Type"],  # ✅ SPECIFIC HEADERS ONLY
    max_age=86400,  # ✅ 24hr PREFLIGHT CACHE
)

# Mount the analysis app
app.mount("/", analysis_app)


def main() -> None:
    """Run the analysis service."""
    host = os.getenv("HOST", "127.0.0.1")
    port = int(os.getenv("PORT", "8081"))

    logger.info("Starting CLARITY Analysis Service")
    logger.info("Listening on %s:%s", host, port)

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
