"""Application settings using Pydantic Settings."""

import logging
from enum import Enum
from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Environment(str, Enum):
    """Application environment."""

    DEVELOPMENT = "development"
    TESTING = "testing"
    PRODUCTION = "production"


class Settings(BaseSettings):
    """Application settings loaded from environment variables.

    Settings are loaded from .env files and environment variables.
    Environment variables take precedence over .env file values.

    Attributes:
        environment: Current environment (development, testing, production).
        log_level: Logging level as string (DEBUG, INFO, WARNING, ERROR).
        api_url: Base URL for the pricing prediction API.
        api_host: Host for the FastAPI server.
        api_port: Port for the FastAPI server.
        dashboard_port: Port for the Streamlit dashboard.
        mlflow_tracking_uri: URI for MLflow tracking server.
        data_dir: Directory for data files.
        models_dir: Directory for serialized models.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )

    # Environment
    environment: Environment = Environment.DEVELOPMENT

    # Logging
    log_level: str = "DEBUG"

    # API
    api_url: str = "http://localhost:8000"
    api_host: str = "0.0.0.0"
    api_port: int = 8000

    # Dashboard
    dashboard_port: int = 8501

    # MLflow
    mlflow_tracking_uri: str = "./mlruns"

    # Paths
    data_dir: Path = Path("data")
    models_dir: Path = Path("models")

    @property
    def log_level_int(self) -> int:
        """Get logging level as integer.

        Returns:
            Logging level constant (e.g., logging.DEBUG, logging.INFO).
        """
        return getattr(logging, self.log_level.upper(), logging.INFO)


# Logging config by environment (from ORCHESTRATION.md)
LOGGING_CONFIG: dict[Environment, dict[str, int | str]] = {
    Environment.DEVELOPMENT: {
        "level": logging.DEBUG,
        "format": "%(asctime)s - %(name)s:%(lineno)d - %(levelname)s - %(message)s",
    },
    Environment.TESTING: {
        "level": logging.WARNING,
        "format": "%(levelname)s - %(message)s",
    },
    Environment.PRODUCTION: {
        "level": logging.INFO,
        "format": "%(asctime)s - %(levelname)s - %(message)s",
    },
}


def configure_logging(settings: Settings) -> None:
    """Configure logging based on environment.

    Sets up the root logger with appropriate level and format based on
    the current environment.

    Args:
        settings: Application settings instance.
    """
    config = LOGGING_CONFIG.get(
        settings.environment, LOGGING_CONFIG[Environment.DEVELOPMENT]
    )
    logging.basicConfig(
        level=config["level"],
        format=config["format"],
    )


@lru_cache
def get_settings() -> Settings:
    """Get cached settings instance.

    Uses lru_cache to ensure settings are loaded only once and reused
    across the application.

    Returns:
        Cached Settings instance.
    """
    return Settings()
