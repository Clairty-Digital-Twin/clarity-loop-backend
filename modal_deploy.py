import modal
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).parent

image = (
    modal.Image.debian_slim()
    .pip_install(
        "fastapi",
        "uvicorn",
        "firebase-admin",
        "google-cloud-firestore",
        "google-cloud-storage",
        "google-cloud-aiplatform",
        "google-auth",
        "google-api-python-client",
        "python-dotenv",
        "pydantic",
        "httpx",
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
    if has_service_account:
        try:
            sa_info = json.loads(os.environ["GOOGLE_APPLICATION_CREDENTIALS_JSON"])
            project_id = sa_info.get("project_id", "unknown")
        except json.JSONDecodeError:
            project_id = "error-parsing-json"
    
    return {
        "has_service_account": has_service_account,
        "has_gemini_key": has_gemini_key,
        "env_vars_keys": list(os.environ.keys()),
        "project_id": project_id,
        "gemini_key_length": len(os.environ.get("GEMINI_API_KEY", "")) if has_gemini_key else 0
    }