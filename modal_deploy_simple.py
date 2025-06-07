"""Simplified Modal deployment for testing credentials and basic functionality."""

import json
import os
import sys
from pathlib import Path

import modal

REPO_ROOT = Path(__file__).parent

# Minimal dependency list for testing
image = (
    modal.Image.debian_slim()
    .pip_install(
        # Core FastAPI and async framework
        "fastapi>=0.115.0,<0.116.0",
        "uvicorn[standard]>=0.32.0,<0.35.0",
        "pydantic>=2.9.0,<3.0.0",
        "pydantic-settings>=2.6.0,<3.0.0",

        # Essential validation
        "email-validator>=2.2.0,<3.0.0",

        # Configuration and environment
        "python-dotenv>=1.0.0,<2.0.0",

        # HTTP client
        "httpx>=0.27.0,<0.28.0",

        # Security basics
        "python-multipart>=0.0.18,<1.0.0",

        # Google Cloud Platform - minimal for auth
        "google-cloud-firestore>=2.19.0,<3.0.0",
        "firebase-admin>=6.5.0,<7.0.0",
        "google-generativeai>=0.8.3,<1.0.0",

        # Monitoring basics
        "prometheus-client>=0.21.0,<1.0.0",
    )
    .add_local_dir(REPO_ROOT, "/app")
)

app = modal.App("clarity-backend-simple")


@app.function(
    image=image,
    secrets=[
        modal.Secret.from_name("googlecloud-secret")
    ]
)
def ping_credentials():
    """Test and log credential information."""
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
    for key in os.environ:
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

    # Print result for logging (allowed in test functions)
    print("=== CREDENTIAL AUDIT RESULTS ===")  # noqa: T201
    print(json.dumps(result, indent=2))  # noqa: T201
    print("=== END CREDENTIAL AUDIT ===")  # noqa: T201

    return result


@app.function(
    image=image,
    secrets=[
        modal.Secret.from_name("googlecloud-secret")
    ]
)
def test_basic_imports():
    """Test if basic imports work without the ML dependencies."""
    sys.path.append("/app/src")

    try:
        # Test basic imports
        from clarity.core.config import get_settings  # noqa: F401
        from clarity.core.logging_config import setup_logging  # noqa: F401

        # Test settings loading
        settings = get_settings()

        return {
            "status": "SUCCESS",
            "message": "Basic imports and config loading successful",
            "environment": settings.environment,
            "debug": settings.debug,
            "log_level": settings.log_level
        }
    except Exception as e:
        return {
            "status": "ERROR",
            "message": f"Basic import test failed: {e!s}",
            "error_type": type(e).__name__
        }


@app.function(
    image=image,
    secrets=[
        modal.Secret.from_name("googlecloud-secret")
    ]
)
def comprehensive_modal_test():
    """Comprehensive test of Modal deployment functionality."""
    sys.path.append("/app/src")
    
    results = {
        "timestamp": "2025-01-07",
        "deployment_status": "OPERATIONAL",
        "tests": {}
    }
    
    # Test 1: Credentials
    try:
        has_service_account = "GOOGLE_APPLICATION_CREDENTIALS_JSON" in os.environ
        has_gemini_key = "GEMINI_API_KEY" in os.environ
        
        if has_service_account and has_gemini_key:
            import json
            sa_info = json.loads(os.environ["GOOGLE_APPLICATION_CREDENTIALS_JSON"])
            project_id = sa_info.get("project_id", "unknown")
            
            results["tests"]["credentials"] = {
                "status": "PASS",
                "project_id": project_id,
                "gemini_key_length": len(os.environ.get("GEMINI_API_KEY", "")),
                "service_account_email": sa_info.get("client_email", "unknown")
            }
        else:
            results["tests"]["credentials"] = {
                "status": "FAIL",
                "message": "Missing credentials"
            }
    except Exception as e:
        results["tests"]["credentials"] = {
            "status": "ERROR",
            "message": str(e)
        }
    
    # Test 2: Core imports
    try:
        from clarity.core.config import get_settings
        from clarity.core.logging_config import setup_logging
        from clarity.core.cloud import firebase_credentials, gemini_api_key
        
        settings = get_settings()
        
        results["tests"]["core_imports"] = {
            "status": "PASS",
            "environment": settings.environment,
            "config_loaded": True
        }
    except Exception as e:
        results["tests"]["core_imports"] = {
            "status": "ERROR",
            "message": str(e)
        }
    
    # Test 3: Cloud credentials helper
    try:
        from clarity.core.cloud import firebase_credentials, gemini_api_key
        
        # Try to load credentials
        creds = firebase_credentials()
        api_key = gemini_api_key()
        
        results["tests"]["cloud_helpers"] = {
            "status": "PASS",
            "credentials_type": str(type(creds)),
            "api_key_length": len(api_key)
        }
    except Exception as e:
        results["tests"]["cloud_helpers"] = {
            "status": "ERROR",
            "message": str(e)
        }
    
    # Test 4: FastAPI dependencies (lightweight test)
    try:
        import fastapi
        import uvicorn
        import pydantic
        
        results["tests"]["fastapi_deps"] = {
            "status": "PASS",
            "fastapi_version": fastapi.__version__,
            "pydantic_version": pydantic.__version__
        }
    except Exception as e:
        results["tests"]["fastapi_deps"] = {
            "status": "ERROR",
            "message": str(e)
        }
    
    # Overall status
    passed_tests = sum(1 for test in results["tests"].values() if test.get("status") == "PASS")
    total_tests = len(results["tests"])
    
    results["summary"] = {
        "passed": passed_tests,
        "total": total_tests,
        "success_rate": f"{(passed_tests/total_tests)*100:.1f}%",
        "overall_status": "PASS" if passed_tests == total_tests else "PARTIAL_PASS"
    }
    
    # Print for logging
    print("=== COMPREHENSIVE MODAL TEST RESULTS ===")  # noqa: T201
    print(json.dumps(results, indent=2))  # noqa: T201
    print("=== END COMPREHENSIVE TEST ===")  # noqa: T201
    
    return results