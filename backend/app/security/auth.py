from fastapi import Security, HTTPException
from fastapi.security.api_key import APIKeyHeader
import secrets

from app.core.config import settings


api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


def verify_api_key(api_key: str = Security(api_key_header)) -> str | None:
    """Verify API key against configured setting using constant-time compare.

    If no `settings.api_key` is configured the check is bypassed (useful for
    local development). When an API key is configured, we use
    `secrets.compare_digest` to avoid timing-attack vulnerabilities.
    """

    expected = settings.api_key
    # If no expected API key configured, skip verification
    if not expected:
        return api_key

    # Require a presented key and compare in constant time
    if not api_key or not secrets.compare_digest(str(api_key), str(expected)):
        raise HTTPException(status_code=401, detail="Unauthorized")

    return api_key
