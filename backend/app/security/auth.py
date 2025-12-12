from fastapi import Security, HTTPException
from fastapi.security.api_key import APIKeyHeader

from app.core.config import settings


api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


def verify_api_key(api_key: str = Security(api_key_header)):
    """Verify API key against configured setting.

    This module is placed under `app.security` so it can be imported by
    both `app.main` and route modules without creating circular imports.
    """

    expected = settings.api_key
    if not expected:
        return api_key

    if api_key != expected:
        raise HTTPException(status_code=401, detail="Unauthorized")

    return api_key
