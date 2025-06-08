"""CLARITY Digital Twin Platform - Cloud Credentials Helper.

This module provides centralized access to cloud credentials and API keys.
"""

import json
import os
from typing import Any

from google.oauth2 import service_account


def firebase_credentials() -> Any:
    """Get Firebase/GCP credentials from environment variables.

    Returns:
        google.oauth2.service_account.Credentials: The service account credentials
    """
    sa_info = json.loads(os.environ["GOOGLE_APPLICATION_CREDENTIALS_JSON"])
    return service_account.Credentials.from_service_account_info(sa_info)


def gemini_api_key() -> str:
    """Get Gemini API key from environment variables.

    Returns:
        str: The Gemini API key
    """
    return os.environ["GEMINI_API_KEY"]
