"""
CivicPulse AI ML Service - Configuration Module

Loads configuration from environment variables with validation.
"""

from pydantic_settings import BaseSettings
from pydantic import Field
from functools import lru_cache


class Settings(BaseSettings):
    """ML Service configuration."""

    # Service
    app_name: str = "CivicPulse AI - ML Service"
    app_version: str = "0.1.0"
    debug: bool = False
    host: str = Field(default="0.0.0.0", alias="ML_SERVICE_HOST")
    port: int = Field(default=8000, alias="ML_SERVICE_PORT")

    # Database (read-only access for fetching data)
    postgres_host: str = "localhost"
    postgres_port: int = 5432
    postgres_db: str = "civicpulse"
    postgres_user: str = "civicpulse"
    postgres_password: str = "civicpulse_dev_password"

    # Redis
    redis_host: str = "localhost"
    redis_port: int = 6379
    redis_password: str = ""

    # MinIO
    minio_endpoint: str = "localhost"
    minio_port: int = 9000
    minio_root_user: str = "civicpulse_minio"
    minio_root_password: str = ""
    minio_bucket: str = "civicpulse-images"
    minio_use_ssl: bool = False

    # ML
    model_artifacts_dir: str = "./model_artifacts"
    max_image_size_mb: int = 10

    # MLflow
    mlflow_tracking_uri: str = "http://localhost:5001"

    model_config = {
        "env_file": "../../.env",
        "env_file_encoding": "utf-8",
        "extra": "ignore",
    }


@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()
