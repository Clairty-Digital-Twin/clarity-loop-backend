import modal
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).parent

# Comprehensive dependency list matching pyproject.toml production dependencies
image = (
    modal.Image.debian_slim()
    .pip_install(
        # Core FastAPI and async framework
        "fastapi>=0.115.0,<0.116.0",
        "uvicorn[standard]>=0.32.0,<0.35.0", 
        "pydantic>=2.9.0,<3.0.0",
        "pydantic-settings>=2.6.0,<3.0.0",
        
        # Google Cloud Platform - constrained for compatibility
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
        
        # AI/ML Dependencies (PAT Implementation) - major version constraints
        "torch>=2.7.0,<3.0.0",
        "transformers>=4.52.4,<5.0.0",
        "numpy>=1.26.0,<2.0.0",
        "scikit-learn>=1.5.0,<2.0.0",
        "matplotlib>=3.9.0,<4.0.0",
        "seaborn>=0.13.0,<0.14.0",
        "plotly>=5.24.0,<6.0.0",
        "h5py>=3.11.0,<4.0.0",  # Required for PAT model weight loading
        
        # Google AI/Gemini Integration
        "google-generativeai>=0.8.3,<1.0.0",
        "langchain>=0.3.0,<0.4.0",
        "langchain-google-vertexai>=2.0.0,<3.0.0",
        
        # HTTP and networking
        "httpx>=0.27.0,<0.28.0",
        "aiofiles>=24.1.0,<25.0.0",
        "websockets>=13.1,<14.0.0",
        
        # Database and caching
        "redis[hiredis]>=5.1.0,<6.0.0",
        "asyncpg>=0.29.0,<0.30.0",
        "alembic>=1.13.0,<2.0.0",
        
        # Security and validation
        "cryptography>=44.0.1,<45.0.0",
        "bcrypt>=4.2.0,<5.0.0",
        "python-multipart>=0.0.18,<1.0.0",
        "email-validator>=2.2.0,<3.0.0",
        
        # Health data processing
        "pandas>=2.2.0,<3.0.0",
        "scipy>=1.14.0,<2.0.0",
        "pytz>=2024.2,<2025.0",
        "python-dateutil>=2.9.0,<3.0.0",
        
        # Monitoring and observability
        "prometheus-client>=0.21.0,<1.0.0",
        "structlog>=24.4.0,<25.0.0",
        "rich>=13.9.0,<14.0.0",
        "circuitbreaker>=2.0.0,<3.0.0",
        
        # Configuration and environment
        "python-dotenv>=1.0.0,<2.0.0",
        "click>=8.1.0,<9.0.0",
        "typer>=0.12.0,<1.0.0",
    )
    # Mount SOURCE after all build steps; .modalignore keeps .git out
    .add_local_dir(REPO_ROOT, "/app")
)

app = modal.App("clarity-backend")

@app.function(
    image=image, 
    secrets=[
        modal.Secret.from_name("googlecloud-secret")
    ]
)
@modal.asgi_app()
def fastapi_app():
    import os
    # Create logs directory if needed (but we're using console logging only)
    os.makedirs("/tmp/logs", exist_ok=True)
    
    sys.path.append("/app/src")            # make `import clarity` work
    from clarity.main import get_app       # noqa
    return get_app()

@app.function(
    image=image,
    secrets=[
        modal.Secret.from_name("googlecloud-secret")
    ]
)
def ping_credentials():
    import json
    import os
    sys.path.append("/app/src")
    
    # Check for specific keys we need
    has_service_account = "GOOGLE_APPLICATION_CREDENTIALS_JSON" in os.environ
    has_gemini_key = "GEMINI_API_KEY" in os.environ
    
    # Try to extract project_id if service account JSON is available
    project_id = "unknown"
    service_account_email = "unknown"
    if has_service_account:
        try:
            sa_info = json.loads(os.environ["GOOGLE_APPLICATION_CREDENTIALS_JSON"])
            project_id = sa_info.get("project_id", "unknown")
            service_account_email = sa_info.get("client_email", "unknown")
        except json.JSONDecodeError:
            project_id = "error-parsing-json"
            service_account_email = "error-parsing-json"
    
    # Log environment variables for debugging (safely)
    safe_env_vars = {}
    for key in os.environ.keys():
        if any(sensitive in key.lower() for sensitive in ['key', 'secret', 'password', 'token', 'json']):
            safe_env_vars[key] = f"[HIDDEN - length: {len(os.environ[key])}]"
        else:
            safe_env_vars[key] = os.environ[key]
    
    result = {
        "status": "SUCCESS",
        "credentials_found": {
            "service_account": has_service_account,
            "gemini_api_key": has_gemini_key
        },
        "service_account_info": {
            "project_id": project_id,
            "service_account_email": service_account_email
        },
        "gemini_key_info": {
            "length": len(os.environ.get("GEMINI_API_KEY", "")) if has_gemini_key else 0,
            "prefix": os.environ.get("GEMINI_API_KEY", "")[:10] + "..." if has_gemini_key else "not_found"
        },
        "environment_variables": safe_env_vars
    }
    
    # Print result for logging
    print("=== CREDENTIAL AUDIT RESULTS ===")
    print(json.dumps(result, indent=2))
    print("=== END CREDENTIAL AUDIT ===")
    
    return result

@app.function(
    image=image,
    secrets=[
        modal.Secret.from_name("googlecloud-secret")
    ]
)
def test_app_startup():
    """Test if the FastAPI app can start without errors"""
    import os
    import sys
    sys.path.append("/app/src")
    
    try:
        # Create logs directory if needed
        os.makedirs("/tmp/logs", exist_ok=True)
        
        # Import and test app initialization
        from clarity.main import get_app
        app = get_app()
        
        # Basic health check
        return {
            "status": "SUCCESS",
            "message": "FastAPI app initialized successfully",
            "app_title": getattr(app, 'title', 'Unknown'),
            "app_version": getattr(app, 'version', 'Unknown')
        }
    except Exception as e:
        return {
            "status": "ERROR", 
            "message": f"App initialization failed: {str(e)}",
            "error_type": type(e).__name__
        }