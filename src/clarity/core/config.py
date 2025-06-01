"""
CLARITY Digital Twin Platform - Configuration Management.

Environment-based configuration using Pydantic settings for secure,
production-ready deployment across development, staging, and production.
"""

import os
from functools import lru_cache
from typing import List

from pydantic import Field
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application configuration with environment variable support."""
    
    # Application settings
    app_name: str = "CLARITY Digital Twin Platform"
    app_version: str = "1.0.0"
    environment: str = Field(default="development", env="ENVIRONMENT")
    debug: bool = Field(default=False, env="DEBUG")
    
    # Server settings
    host: str = Field(default="0.0.0.0", env="HOST")
    port: int = Field(default=8080, env="PORT")
    log_level: str = Field(default="INFO", env="LOG_LEVEL")
    
    # Security settings
    allowed_hosts: List[str] = Field(
        default=["localhost", "127.0.0.1", "*.run.app"],
        env="ALLOWED_HOSTS"
    )
    cors_origins: List[str] = Field(
        default=["http://localhost:3000", "http://localhost:8080"],
        env="CORS_ORIGINS"
    )
    
    # Firebase settings
    firebase_project_id: str = Field(default="", env="FIREBASE_PROJECT_ID")
    firebase_credentials_path: str = Field(default="", env="FIREBASE_CREDENTIALS_PATH")
    
    # Google Cloud settings
    gcp_project_id: str = Field(default="", env="GCP_PROJECT_ID")
    firestore_database: str = Field(default="(default)", env="FIRESTORE_DATABASE")
    
    # Vertex AI settings
    vertex_ai_project_id: str = Field(default="", env="VERTEX_AI_PROJECT_ID")
    vertex_ai_location: str = Field(default="us-central1", env="VERTEX_AI_LOCATION")
    vertex_ai_model_id: str = Field(
        default="gemini-2.5-pro-preview-05-06",
        env="VERTEX_AI_MODEL_ID"
    )
    
    # Storage settings
    cloud_storage_bucket: str = Field(default="", env="CLOUD_STORAGE_BUCKET")
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False


@lru_cache()
def get_settings() -> Settings:
    """Get cached application settings."""
    return Settings()
