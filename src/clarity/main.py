"""CLARITY Digital Twin Platform - Main Application Entry Point.

FastAPI application setup with Clean Architecture dependency injection.
This module serves as the composition root for the entire application.
"""

# --- Modal-specific credential handling ---
# This block MUST run before any other application imports to ensure the
# environment is correctly configured before any settings are loaded.
import os
import json
from pathlib import Path

google_creds_json = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS_JSON")
if google_creds_json:
    temp_dir = Path("/tmp/clarity_creds")
    temp_dir.mkdir(parents=True, exist_ok=True)
    creds_path = temp_dir / "gcp_creds.json"
    
    try:
        # Verify it's valid JSON before writing
        json.loads(google_creds_json)
        with open(creds_path, "w") as f:
            f.write(google_creds_json)
        
        # Set the environment variable to the path of the new file
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = str(creds_path)
        print(f"✅ Successfully created and set GOOGLE_APPLICATION_CREDENTIALS to {creds_path}")
    except (json.JSONDecodeError, OSError) as e:
        print(f"🔥 ERROR: Failed to write credentials to temporary file: {e}")
# --- End Modal-specific credential handling ---


import logging

from dotenv import load_dotenv
from fastapi import FastAPI, WebSocket
import uvicorn

from clarity.core.config import get_settings
from clarity.core.container import create_application
from clarity.core.logging_config import setup_logging

# Load environment variables at the beginning
load_dotenv()

# Configure logging
logger = logging.getLogger(__name__)


# Application factory function
def create_app() -> FastAPI:
    """Create and configure FastAPI application.

    Factory function for creating the application instance.
    Uses Clean Architecture dependency injection container.
    """
    return create_application()


# For development and testing
def get_app() -> FastAPI:
    """Get application instance for development/testing.

    Creates a new instance each time to avoid global state issues.
    Uses Clean Architecture dependency injection container.
    """
    return create_application()


# For production deployment (uvicorn/gunicorn)
app = create_application()


# Add WebSocket route
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket) -> None:
    await websocket.accept()
    while True:
        data = await websocket.receive_text()
        await websocket.send_text(f"Message text was: {data}")


# For production deployment and direct execution
if __name__ == "__main__":
    # Setup logging first
    settings = get_settings()
    setup_logging()

    logger.info("Starting CLARITY Digital Twin Platform...")
    logger.info("Environment: %s", settings.environment)
    logger.info("Debug mode: %s", settings.debug)

    uvicorn.run(
        app,  # Pass app instance directly
        host=settings.host,
        port=settings.port,
        reload=settings.debug,
        log_level=settings.log_level.lower(),
    )
else:
    # Module import - lazy creation for better testability
    app = get_app()
