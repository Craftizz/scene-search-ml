"""WebSocket session management service."""

import asyncio
import json
import logging
from typing import Optional, Dict, Any, List
from dataclasses import dataclass

from fastapi import WebSocket, WebSocketDisconnect
from app.models import WebSocketConfig

logger = logging.getLogger("scene_search")


@dataclass
class SessionState:
    """Track session state and metadata."""
    is_connected: bool = True
    ws_key: Optional[str] = None
    closed: bool = False


class WebSocketSessionManager:
    """Manage WebSocket communication and session lifecycle."""
    
    def __init__(self, websocket: WebSocket, config: Optional[WebSocketConfig] = None):
        """Initialize WebSocket session manager.
        
        Args:
            websocket: FastAPI WebSocket connection
            config: Optional WebSocket configuration
        """
        self.websocket = websocket
        self.config = config or WebSocketConfig()
        self.state = SessionState()
    
    async def accept_connection(self, ws_key: Optional[str] = None) -> None:
        """Accept the WebSocket connection and initialize session.
        
        Args:
            ws_key: Optional key for rate limiting
        """
        try:
            await self.websocket.accept()
            self.state.ws_key = ws_key
            self.state.is_connected = True
        except Exception as e:
            logger.debug("Failed to accept WebSocket connection: %s", e)
            self.state.closed = True
    
    async def send_message(self, message: Dict[str, Any]) -> None:
        """Send a JSON message over the websocket.
        
        Args:
            message: Message dictionary to send
        """
        if self.state.closed:
            return
            
        try:
            await self.websocket.send_json(message)
        except Exception as e:
            logger.debug("Failed to send websocket message: %s", e)
            self.state.closed = True
    
    async def receive_frame_meta(self) -> Optional[Dict[str, Any]]:
        """Receive and parse frame metadata message.
        
        Returns:
            Parsed metadata or None if error/disconnection
        """
        if self.state.closed:
            return None
            
        try:
            msg = await self.websocket.receive()
            
            if msg is None or msg.get("type") == "websocket.disconnect":
                self.state.closed = True
                return None
                
            if msg.get("text") is not None:
                try:
                    return json.loads(msg["text"]) if isinstance(msg["text"], str) else None
                except Exception as e:
                    logger.debug("Failed to parse JSON message: %s", e)
                    return None
                    
        except WebSocketDisconnect:
            logger.debug("WebSocket disconnected by client")
            self.state.closed = True
        except Exception as e:
            logger.debug("WebSocket receive error: %s", e)
            self.state.closed = True
            
        return None
    
    async def receive_frame_bytes(self) -> Optional[bytes]:
        """Receive binary frame data.
        
        Returns:
            Frame bytes or None if error/disconnection
        """
        if self.state.closed:
            return None
            
        try:
            msg = await self.websocket.receive()
            
            if msg is None or msg.get("type") == "websocket.disconnect":
                self.state.closed = True
                return None
                
            img_bytes = msg.get("bytes")
            if img_bytes is not None and isinstance(img_bytes, (bytes, bytearray)):
                return bytes(img_bytes)
                
        except WebSocketDisconnect:
            logger.debug("WebSocket disconnected by client")
            self.state.closed = True
        except Exception as e:
            logger.debug("Failed to receive binary frame: %s", e)
            self.state.closed = True
            
        return None
    
    async def send_error(self, message: str, request_id: Optional[str] = None) -> None:
        """Send an error message to the client.
        
        Args:
            message: Error message
            request_id: Optional request ID for context
        """
        error_msg = {"type": "error", "message": message}
        if request_id:
            error_msg["request_id"] = request_id
        await self.send_message(error_msg)
    
    async def send_embedded_batch(
        self, 
        timestamps: List[float], 
        embeddings: List[List[float]]
    ) -> None:
        """Send embedded batch message to client.
        
        Args:
            timestamps: Frame timestamps
            embeddings: Frame embeddings
        """
        await self.send_message({
            "type": "embedded_batch",
            "timestamps": timestamps,
            "embeddings": embeddings
        })
    
    async def send_scene(
        self,
        scene_id: int,
        timestamp: float,
        end_timestamp: Optional[float] = None,
        request_id: Optional[str] = None
    ) -> None:
        """Send scene detection message to client.
        
        Args:
            scene_id: Scene identifier
            timestamp: Scene start timestamp
            end_timestamp: Optional scene end timestamp
            request_id: Optional request ID for captions
        """
        scene_msg = {
            "type": "scene",
            "id": scene_id,
            "timestamp": timestamp
        }
        
        if end_timestamp is not None:
            scene_msg["end_timestamp"] = end_timestamp
        if request_id is not None:
            scene_msg["request_id"] = request_id
            
        await self.send_message(scene_msg)
    
    async def send_caption_started(self, request_id: str) -> None:
        """Send caption processing started message.
        
        Args:
            request_id: Caption request ID
        """
        await self.send_message({
            "type": "caption_started",
            "request_id": request_id
        })
    
    async def send_caption_result(self, request_id: str, caption: str) -> None:
        """Send caption result message.
        
        Args:
            request_id: Caption request ID
            caption: Generated caption text
        """
        await self.send_message({
            "type": "caption_result",
            "request_id": request_id,
            "caption": caption
        })
    
    def is_connected(self) -> bool:
        """Check if WebSocket is still connected."""
        return self.state.is_connected and not self.state.closed
    
    def close(self) -> None:
        """Close the WebSocket session."""
        self.state.closed = True
        self.state.is_connected = False
        
        # Release websocket slot if acquired
        if self.state.ws_key:
            try:
                from app.core.rate_limiter import limiter
                asyncio.create_task(limiter.release_ws(self.state.ws_key))
            except Exception:
                pass