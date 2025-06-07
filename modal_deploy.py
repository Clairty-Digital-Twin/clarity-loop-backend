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
    "firebase-admin",
    "google-cloud-firestore",
    "google-cloud-storage",
    "google-cloud-aiplatform",
    "google-auth",
    "google-http",
    "google-api-python-client",
    "python-dotenv",
    "pydantic",
    "httpx",
)

# Add the project files to the image with copy=True so we can run pip install after
image = image.add_local_dir(".", "/app", copy=True)

# Install the package in editable mode so `import clarity` works properly
image = image.pip_install("/app", extra_options=["-e"])

@app.function(
    image=image,
    secrets=[
        modal.Secret.from_name("googlecloud-secret"),
        modal.Secret.from_name("gemini-secret")
    ]
)
@modal.asgi_app()
def fastapi_app():
    """Deploy the FastAPI application to Modal."""
    # Import the FastAPI application
    from clarity.main import get_app
    
    # Return the FastAPI application
    return get_app()


@app.function(
    image=image,
    secrets=[
        modal.Secret.from_name("googlecloud-secret"),
        modal.Secret.from_name("gemini-secret")
    ]
)
def ping_credentials():
    """Test function to verify credentials are working."""
    import json
    import os
    
    # Import the helper functions
    from clarity.core.cloud import firebase_credentials, gemini_api_key
    
    # Get the credentials
    creds = firebase_credentials()
    api_key = gemini_api_key()
    
    # Extract project_id from the service account info
    sa_info = json.loads(os.environ["SERVICE_ACCOUNT_JSON"])
    project_id = sa_info.get("project_id", "unknown")
    
    # Return basic info without exposing sensitive data
    return {
        "project_id": project_id,
        "api_key_length": len(api_key) if api_key else 0
    }
