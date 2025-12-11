import os
from typing import Literal
from dotenv import load_dotenv
from pydantic_settings import BaseSettings, SettingsConfigDict
import logging

logger = logging.getLogger("scene_search")


class Settings(BaseSettings):
    """Application configuration with environment variable support."""

    app_name: str = "Scene Search API"
    version: str = "1.0.0"
    environment: Literal["development", "staging", "production"] = "development"

    enable_metrics: bool = True
    enable_docs: bool = True

    max_batch_size: int = 32

    allowed_hosts: list[str] = ["*"]
    cors_origins: list[str] = ["*"]
    api_key: str | None = None

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )


def load_settings() -> Settings:
    """Load settings from environment variables and .env file."""

    env_files = [".env", ".env.development", ".env.local"]
    env_file_used = next((f for f in env_files if os.path.exists(f)), None)

    if env_file_used:
        logger.info(f"Loading configuration from: {env_file_used}")
        load_dotenv(env_file_used, override=True)
    else:
        logger.warning("No .env file found, using default settings")

    return Settings()


settings = load_settings()
