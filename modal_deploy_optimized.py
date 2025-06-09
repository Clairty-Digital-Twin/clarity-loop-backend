"""Optimized Modal deployment with layered dependency installation for fast caching.

This approach uses Modal's image layering to cache dependencies separately,
making deployments much faster when only code changes.
"""

import json
import os
from pathlib import Path
import sys
from typing import Any, Dict

import modal

REPO_ROOT = Path(__file__).parent

google_secret = modal.Secret.from_name("googlecloud-secret")


def setup_firebase_credentials():
    """Write Firebase credentials from environment to file during image build"""
    import json
    import os
    from pathlib import Path

    creds_dir = Path("/workspace/creds")
    creds_dir.mkdir(parents=True, exist_ok=True)

    creds_json = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS_JSON")
    if creds_json:
        with open(creds_dir / "firebase.json", "w") as f:
            # Parse and re-write to ensure it's valid JSON
            creds_data = json.loads(creds_json)
            json.dump(creds_data, f, indent=2)
        print("✅ Firebase credentials written to /workspace/creds/firebase.json")
    else:
        print("⚠️ No GOOGLE_APPLICATION_CREDENTIALS_JSON found")


# Layer 1: Base Python dependencies (fast install, cache-friendly)
base_image = (
    modal.Image.debian_slim()
    .pip_install(
        # Core FastAPI and async framework
        "fastapi>=0.115.0,<0.116.0",
        "uvicorn[standard]>=0.32.0,<0.35.0",
        "pydantic>=2.9.0,<3.0.0",
        "pydantic-settings>=2.6.0,<3.0.0",
        # Basic HTTP and networking
        "httpx>=0.27.0,<0.28.0",
        "aiofiles>=24.1.0,<25.0.0",
        "websockets>=13.1,<14.0.0",
        # Security and validation basics
        "python-multipart>=0.0.18,<1.0.0",
        "email-validator>=2.2.0,<3.0.0",
        "cryptography>=44.0.1,<45.0.0",
        "bcrypt>=4.2.0,<5.0.0",
        # Configuration and environment
        "python-dotenv>=1.0.0,<2.0.0",
        "click>=8.1.0,<9.0.0",
        "typer>=0.12.0,<1.0.0",
        # Basic monitoring
        "prometheus-client>=0.21.0,<1.0.0",
        "structlog>=24.4.0,<25.0.0",
        "rich>=13.9.0,<14.0.0",
    )
    .run_function(
        setup_firebase_credentials,
        secrets=[google_secret]
    )
    .env(
        {
            "GOOGLE_APPLICATION_CREDENTIALS": "/workspace/creds/firebase.json"
        }
    )
)

# Layer 2: Google Cloud and Firebase dependencies (medium weight)
gcp_image = base_image.pip_install(
    # Google Cloud Platform
    "google-cloud-firestore>=2.19.0,<3.0.0",
    "google-cloud-storage>=2.18.0,<3.0.0",
    "google-cloud-pubsub>=2.25.0,<3.0.0",
    "google-cloud-secret-manager>=2.21.1,<3.0.0",
    "google-cloud-aiplatform>=1.70.0,<2.0.0",
    "google-cloud-run>=0.10.7,<1.0.0",
    "google-cloud-logging>=3.11.3,<4.0.0",
    "google-cloud-monitoring>=2.22.2,<3.0.0",
    # Firebase Authentication
    "firebase-admin>=6.5.0,<7.0.0",
    "PyJWT[crypto]>=2.10.0,<3.0.0",
    # Google AI/Gemini
    "google-generativeai>=0.8.3,<1.0.0",
    "langchain>=0.3.0,<0.4.0",
    "langchain-google-vertexai>=2.0.0,<3.0.0",
)

# Layer 3: Heavy ML dependencies (slowest install, cached separately)
ml_image = gcp_image.pip_install(
    # AI/ML Dependencies - these are the heavy ones
    "torch>=2.7.0,<3.0.0",
    "transformers>=4.52.4,<5.0.0",
    "numpy>=1.26.0,<2.0.0",
    "scikit-learn>=1.5.0,<2.0.0",
    "matplotlib>=3.9.0,<4.0.0",
    "seaborn>=0.13.0,<0.14.0",
    "plotly>=5.24.0,<6.0.0",
    "h5py>=3.11.0,<4.0.0",
    # Data processing
    "pandas>=2.2.0,<3.0.0",
    "scipy>=1.14.0,<2.0.0",
    "pytz>=2024.2,<2025.0",
    "python-dateutil>=2.9.0,<3.0.0",
    # Additional utilities
    "redis[hiredis]>=5.1.0,<6.0.0",
    "asyncpg>=0.29.0,<0.30.0",
    "alembic>=1.13.0,<2.0.0",
    "circuitbreaker>=2.0.0,<3.0.0",
)

# Final layer: Mount the source code (this changes most frequently)
# FORCE COPY to ensure latest code is in the image (fixes Modal caching issues)
# REBUILD TIMESTAMP: 2025-01-09-14:20:00-contextvars-fix
import time
final_image = ml_image.add_local_dir(REPO_ROOT, "/app", copy=True).run_commands(
    f"echo 'Build timestamp: {time.time()} - contextvars fix' > /app/build_timestamp.txt"
)

# Detect Modal environment (dev/prod)
modal_environment = os.getenv("MODAL_ENVIRONMENT", "main")
environment_name = "production" if modal_environment == "prod" else "development"

app = modal.App(
    "clarity-backend",
    secrets=[
        modal.Secret.from_name("googlecloud-secret"),
        modal.Secret.from_dict({
            "ENVIRONMENT": "production",
            "FIREBASE_PROJECT_ID": "clarity-loop-backend",
            "GCP_PROJECT_ID": "clarity-loop-backend"
        }),
    ],
)


def _set_environment():
    """Sets the application environment based on the Modal environment."""
    modal_env = os.getenv("MODAL_ENVIRONMENT", "main")
    app_env = "production" if modal_env == "prod" else "development"
    os.environ["ENVIRONMENT"] = app_env


@app.function(
    image=final_image,
    secrets=[
        modal.Secret.from_name("googlecloud-secret"),
        modal.Secret.from_dict({
            "ENVIRONMENT": "production",
            "FIREBASE_PROJECT_ID": "clarity-loop-backend",
            "GCP_PROJECT_ID": "clarity-loop-backend"
        }),
    ],
    cpu=2,  # More CPU for faster cold starts
    memory=2048,  # More memory for ML models
    timeout=300,  # 5 min timeout for ML operations
)
@modal.asgi_app()
def fastapi_app() -> Any:
    """Deploy the optimized FastAPI application with layered caching."""
    _set_environment()

    # Create logs directory if needed
    Path("/tmp/logs").mkdir(parents=True, exist_ok=True)

    # Add source path
    sys.path.append("/app/src")

    # Import and create the app using the factory method instead of global instance
    # CRITICAL: Use the factory method to ensure middleware is properly configured
    from clarity.core.container import create_application

    # Create fresh app instance with proper middleware
    app = create_application()

    # Log app info for debugging
    print(f"🔥🔥 MODAL: Created app with ID: {id(app)}")
    print(f"🔥🔥 MODAL: App title: {app.title}")
    print(f"🔥🔥 MODAL: App middleware attributes: {[attr for attr in dir(app) if 'middleware' in attr.lower()]}")

    return app


@app.function(image=final_image, secrets=[modal.Secret.from_name("googlecloud-secret")])
def health_check() -> dict[str, Any]:
    """Health check endpoint to verify deployment."""
    _set_environment()
    sys.path.append("/app/src")

    try:
        from clarity.core.config import get_settings
        from clarity.main import app as fastapi_app

        settings = get_settings()

        result = {
            "status": "healthy",
            "app_title": getattr(fastapi_app, "title", "CLARITY Backend"),
            "application_environment": settings.environment,
            "modal_environment": os.getenv("MODAL_ENVIRONMENT", "main"),
        }
    except Exception as e:
        result = {
            "status": "unhealthy",
            "error": str(e),
            "error_type": type(e).__name__,
        }
    print(json.dumps(result, indent=2))
    return result


@app.function(
    image=gcp_image,  # Only needs GCP layer, not ML layer
    secrets=[modal.Secret.from_name("googlecloud-secret")],
)
def credentials_test() -> dict[str, Any]:
    """Test credentials without heavy ML dependencies."""
    # Check for credentials
    has_service_account = "GOOGLE_APPLICATION_CREDENTIALS_JSON" in os.environ
    has_gemini_key = "GEMINI_API_KEY" in os.environ

    result = {
        "status": "success",
        "credentials": {
            "service_account": has_service_account,
            "gemini_api_key": has_gemini_key,
        },
    }

    if has_service_account:
        try:
            sa_info = json.loads(os.environ["GOOGLE_APPLICATION_CREDENTIALS_JSON"])
            result["project_id"] = sa_info.get("project_id", "unknown")
            result["service_account_email"] = sa_info.get("client_email", "unknown")
        except json.JSONDecodeError:
            result["credentials_error"] = "Failed to parse service account JSON"

    print(json.dumps(result, indent=2))
    return result


# Lightweight function for testing base dependencies only
@app.function(
    image=base_image,  # Only base layer - fastest
)
def base_test() -> dict[str, Any]:
    """Test that base dependencies are working."""
    try:
        import fastapi
        import pydantic
        from pydantic_settings import BaseSettings  # This was the missing import!

        return {
            "status": "success",
            "fastapi_version": fastapi.__version__,
            "pydantic_version": pydantic.__version__,
            "pydantic_settings": "available",
        }
    except ImportError as e:
        return {"status": "error", "missing_module": str(e)}


@app.local_entrypoint()
def main():
    """Local entrypoint for testing functions."""
    print("--- Running credentials_test ---")
    credentials_test.remote()
    print("\n--- Running health_check ---")
    health_check.remote()


if __name__ == "__main__":
    print("🚀 Optimized Modal deployment script.")
    print("\nUse the following commands from your terminal to deploy:")
    print("  modal deploy --env main modal_deploy_optimized.py")
    print("  modal deploy --env prod modal_deploy_optimized.py")
