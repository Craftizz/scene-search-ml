from __future__ import annotations
import logging
import secrets
from fastapi import APIRouter, WebSocket

from app.core.config import settings
from app.core.rate_limiter import limiter
from .analyze_session import AnalyzeSession, SessionConfig

"""Websocket route for analysis.

This module exposes the `/v1/ws/analyze` websocket route and delegates
per-connection handling to `AnalyzeSession` defined in
`analyze_session.py`.
"""

logger = logging.getLogger("scene_search")


router = APIRouter(prefix="/v1/ws", tags=["ws"])


@router.websocket("/analyze")
async def analyze(websocket: WebSocket):
    api_key = websocket.query_params.get("api_key")
    # also accept API key via WebSocket handshake header `X-API-Key`
    if not api_key:
        api_key = websocket.headers.get("x-api-key") or websocket.headers.get("X-API-Key")
    # also accept API key sent as WebSocket subprotocol (Sec-WebSocket-Protocol)
    if not api_key:
        api_key = websocket.headers.get("sec-websocket-protocol") or websocket.headers.get("Sec-WebSocket-Protocol")

    try:
        hdrs = {k: websocket.headers.get(k) for k in ["origin", "sec-websocket-protocol", "x-api-key", "sec-websocket-extensions"]}
    except Exception:
        logger.exception("Failed to log ws handshake headers")

    expected = settings.api_key
    if expected:
        # Compare using constant-time comparison to avoid timing attacks
        if not api_key or not secrets.compare_digest(str(api_key), str(expected)):
            # In development allow mismatched keys (convenience) but close in prod
            if settings.environment == "development":
                pass
            else:
                await websocket.close(code=1008)
                return

    # Enforce websocket concurrency limits per key/ip
    ws_key = api_key if api_key else (websocket.client.host if websocket.client else "anon")
    try:
        ok = await limiter.acquire_ws(str(ws_key))
    except Exception:
        ok = False
    if not ok:
        # 1013 = Try again later
        await websocket.close(code=1013)
        return

    # parse optional session config from query params (including detection tuning)
    batch_size = websocket.query_params.get("batch_size")
    batch_timeout = websocket.query_params.get("batch_timeout")
    max_pending = websocket.query_params.get("max_pending")
    threshold = websocket.query_params.get("threshold")
    persistence = websocket.query_params.get("persistence")
    smoothing_window = websocket.query_params.get("smoothing_window")
    min_scene_gap_sec = websocket.query_params.get("min_scene_gap_sec")

    cfg = SessionConfig()
    try:
        if batch_size is not None:
            cfg.batch_size = int(batch_size)
        if batch_timeout is not None:
            cfg.batch_timeout = float(batch_timeout)
        if max_pending is not None:
            cfg.max_pending = int(max_pending)
        if threshold is not None:
            cfg.threshold = float(threshold)
        if persistence is not None:
            cfg.persistence = int(persistence)
        if smoothing_window is not None:
            cfg.smoothing_window = int(smoothing_window)
        if min_scene_gap_sec is not None:
            cfg.min_scene_gap_sec = float(min_scene_gap_sec)
    except Exception:
        pass

    session = AnalyzeSession(websocket, config=cfg, ws_key=str(ws_key))
    await session.run()
