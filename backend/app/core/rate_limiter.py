import time
import asyncio
from collections import deque
from typing import Dict

from fastapi import HTTPException

from app.core.config import settings


class RateLimiter:
    """Simple sliding-window rate limiter and websocket connection tracker.

    This in-memory implementation is suitable for single-process deployments.
    For multi-instance deployments use Redis or another centralized store.
    """

    def __init__(self, requests: int, window_seconds: int, max_ws_per_key: int):
        self.requests = int(requests)
        self.window = int(window_seconds)
        self.max_ws = int(max_ws_per_key)

        # key -> deque[timestamp]
        self._hits: Dict[str, deque] = {}
        # key -> count
        self._ws_counts: Dict[str, int] = {}
        self._lock = asyncio.Lock()

    async def allow(self, key: str) -> bool:
        """Return True if request is allowed for `key`, else False."""
        now = time.time()
        async with self._lock:
            q = self._hits.get(key)
            if q is None:
                q = deque()
                self._hits[key] = q

            # prune old hits
            cutoff = now - self.window
            while q and q[0] < cutoff:
                q.popleft()

            if len(q) >= self.requests:
                return False

            q.append(now)
            return True

    async def check_or_raise(self, key: str) -> None:
        ok = await self.allow(key)
        if not ok:
            raise HTTPException(status_code=429, detail="Too Many Requests")

    async def acquire_ws(self, key: str) -> bool:
        """Attempt to acquire a websocket slot for `key`.

        Returns True if acquired, False if limit reached.
        """
        async with self._lock:
            cur = self._ws_counts.get(key, 0)
            if cur >= self.max_ws:
                return False
            self._ws_counts[key] = cur + 1
            return True

    async def release_ws(self, key: str) -> None:
        async with self._lock:
            cur = self._ws_counts.get(key, 0)
            if cur <= 1:
                self._ws_counts.pop(key, None)
            else:
                self._ws_counts[key] = cur - 1


# Single shared limiter instance for the app
limiter = RateLimiter(
    requests=getattr(settings, "rate_limit_requests", 60),
    window_seconds=getattr(settings, "rate_limit_window_seconds", 60),
    max_ws_per_key=getattr(settings, "max_ws_connections_per_key", 4),
)
