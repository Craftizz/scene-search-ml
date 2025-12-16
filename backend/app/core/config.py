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

    # Upload / image safety limits
    # Maximum raw upload size in bytes for single image uploads / ws frames
    max_upload_size_bytes: int = 5 * 1024 * 1024  # 5 MB
    # Maximum allowed image dimensions (width / height)
    max_image_width: int = 4096
    max_image_height: int = 4096
    # Maximum allowed total pixels (width * height)
    max_image_pixels: int = 50_000_000

    # Rate limiting configuration (simple in-memory limiter)
    rate_limit_requests: int = 60
    rate_limit_window_seconds: int = 60
    # Maximum concurrent websocket connections per api_key/ip
    max_ws_connections_per_key: int = 4

    allowed_hosts: list[str] = ["*"]
    cors_origins: list[str] = ["*"]
    # API key used to authenticate requests. Default is empty string.
    # Do NOT leave empty in production; `load_settings()` enforces this.
    api_key: str = ""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )


def load_settings() -> Settings:
    """Load settings from environment variables and .env file."""

    # Determine preferred env file ordering. If the calling process has
    # ENVIRONMENT set to "production" we prioritize `.env.production` so
    # that production-specific values override generic `.env` files.
    preferred_env = os.environ.get("ENVIRONMENT", "").strip().lower()

    env_files = [".env", ".env.development", ".env.local", ".env.production"]
    if preferred_env:
        pref_file = f".env.{preferred_env}"
        # move preferred file to the front if present or add it
        if pref_file in env_files:
            env_files.remove(pref_file)
        env_files.insert(0, pref_file)

    env_file_used = next((f for f in env_files if os.path.exists(f)), None)

    if env_file_used:
        logger.info(f"Loading configuration from: {env_file_used}")
        load_dotenv(env_file_used, override=True)
    else:
        logger.warning("No .env file found, using default settings")

    s = Settings()

    # Enforce secure defaults for API key in production.
    try:
        if s.environment == "production" and not s.api_key:
            logger.error("API key must be set in production environment")
            raise RuntimeError("API key must be set in production environment")
        if not s.api_key:
            logger.warning("No API key configured; authentication will be skipped")
    except Exception:
        # If logger or settings access fails, re-raise to fail fast on startup
        raise

    return s


settings = load_settings()
