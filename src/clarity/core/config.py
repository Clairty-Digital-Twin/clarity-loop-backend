"""CLARITY Digital Twin Platform - Configuration Management.

Environment-based configuration using Pydantic settings for secure,
production-ready deployment across development, staging, and production.
"""

from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application configuration with environment variable support."""

    # Application settings
    app_name: str = "CLARITY Digital Twin Platform"
    app_version: str = "1.0.0"
    environment: str = Field(default="development", alias="ENVIRONMENT")
    debug: bool = Field(default=False, alias="DEBUG")

    # Server settings
    host: str = Field(default="0.0.0.0", alias="HOST")
    port: int = Field(default=8080, alias="PORT")
    log_level: str = Field(default="INFO", alias="LOG_LEVEL")

    # Security settings
    allowed_hosts: list[str] = Field(
        default=["localhost", "127.0.0.1", "*.run.app"], alias="ALLOWED_HOSTS"
    )
    cors_origins: list[str] = Field(
        default=["http://localhost:3000", "http://localhost:8080"], alias="CORS_ORIGINS"
    )

    # Firebase settings
    firebase_project_id: str = Field(default="", alias="FIREBASE_PROJECT_ID")
    firebase_credentials_path: str = Field(
        default="", alias="FIREBASE_CREDENTIALS_PATH"
    )

    # Google Cloud settings
    gcp_project_id: str = Field(default="", alias="GCP_PROJECT_ID")
    firestore_database: str = Field(default="(default)", alias="FIRESTORE_DATABASE")

    # Vertex AI settings
    vertex_ai_project_id: str = Field(default="", alias="VERTEX_AI_PROJECT_ID")
    vertex_ai_location: str = Field(default="us-central1", alias="VERTEX_AI_LOCATION")
    vertex_ai_model_id: str = Field(
        default="gemini-2.5-pro-preview-05-06", alias="VERTEX_AI_MODEL_ID"
    )

    # Storage settings
    cloud_storage_bucket: str = Field(default="", alias="CLOUD_STORAGE_BUCKET")

    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "case_sensitive": False,
    }


@lru_cache
def get_settings() -> Settings:
    """Get cached application settings."""
    return Settings()
