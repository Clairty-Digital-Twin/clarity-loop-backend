"""
CLARITY Digital Twin Platform - Modal Deployment

This file configures the deployment of the CLARITY backend to Modal's cloud platform.
"""

import sys
from pathlib import Path

# Add src directory to Python path
src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))

import modal

# Create a Modal app - this is the main entry point for Modal
app = modal.App("clarity-backend")

# Create a Modal Image with all dependencies
image = modal.Image.debian_slim().pip_install(
    "fastapi",
    "uvicorn",
    "python-dotenv",
    "google-cloud-firestore",
    "google-cloud-storage",
    "firebase-admin",
    "pydantic",
    "pydantic-settings",
    "python-jose[cryptography]",
    "passlib[bcrypt]",
    "httpx",
    "python-multipart",
    "websockets",
)

# Add the local code to the image
image = image.copy_local_dir("./src", "/app/src")
image = image.copy_local_file(".env.example", "/app/.env")

@app.function(image=image)
@modal.asgi_app()
def fastapi_app():
    """Deploy the FastAPI application to Modal."""
    # Add the app directory to the Python path
    sys.path.append("/app")
    
    # Import the FastAPI application
    from clarity.main import get_app
    
    # Return the FastAPI application
    return get_app()
