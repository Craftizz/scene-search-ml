"""Frame processing service for WebSocket video analysis."""

import asyncio
from io import BytesIO
from typing import List, Optional, Tuple
import logging

from PIL import Image
from app.core.config import settings
from app.core.image_limits import validate_image_dimensions
from app.models import FrameProcessingConfig, ProcessedFrame, FrameBatch

logger = logging.getLogger("scene_search")


class FrameProcessor:
    """Handle frame decoding, validation, and batching."""
    
    def __init__(self, config: FrameProcessingConfig):
        """Initialize frame processor with configuration.
        
        Args:
            config: Frame processing configuration
        """
        self.config = config
        self.batch_size = min(config.batch_size, getattr(settings, "max_batch_size", 16))
        
        # Buffering state
        self._bytes_buf: List[bytes] = []
        self._ts_buf: List[float] = []
        self._lock = asyncio.Lock()
        self._timer: Optional[asyncio.Task] = None
    
    async def enqueue_frame(
        self, 
        img_bytes: bytes, 
        timestamp: float,
        total_pending: int,
        on_batch_ready
    ) -> bool:
        """Queue a frame for processing.
        
        Args:
            img_bytes: Raw image bytes
            timestamp: Frame timestamp
            total_pending: Current total pending frames
            on_batch_ready: Callback when batch is ready
            
        Returns:
            True if frame was queued, False if dropped
        """
        # Check size limits
        max_size = getattr(settings, "max_upload_size_bytes", 0)
        if max_size and len(img_bytes) > max_size:
            return False
            
        async with self._lock:
            # Check pending limits
            if total_pending + len(self._bytes_buf) >= self.config.max_pending:
                return False
                
            self._bytes_buf.append(img_bytes)
            self._ts_buf.append(timestamp)
            
            # Schedule timeout if this is first frame
            if self._timer is None:
                self._timer = asyncio.create_task(
                    self._schedule_timeout(on_batch_ready)
                )
                
            # Trigger immediate batch if full
            if len(self._bytes_buf) >= self.batch_size:
                if self._timer:
                    self._timer.cancel()
                    self._timer = None
                asyncio.create_task(on_batch_ready())
                
        return True
    
    async def _schedule_timeout(self, callback) -> None:
        """Schedule timeout-based batch processing."""
        try:
            await asyncio.sleep(self.config.batch_timeout)
            asyncio.create_task(callback())
        except asyncio.CancelledError:
            pass
    
    async def get_batch(self) -> Optional[FrameBatch]:
        """Get current batch and clear buffers.
        
        Returns:
            FrameBatch or None if empty
        """
        async with self._lock:
            if self._timer:
                self._timer.cancel()
                self._timer = None
                
            if not self._bytes_buf:
                return None
                
            # Snapshot and clear
            bytes_batch = self._bytes_buf.copy()
            ts_batch = self._ts_buf.copy()
            self._bytes_buf.clear()
            self._ts_buf.clear()
            
        # Process batch
        processed_frames = []
        for img_bytes, timestamp in zip(bytes_batch, ts_batch):
            frame = self._decode_frame(img_bytes, timestamp)
            if frame:
                processed_frames.append(frame)
        
        return FrameBatch(
            frames=processed_frames,
            raw_count=len(bytes_batch)
        )
    
    def _decode_frame(self, img_bytes: bytes, timestamp: float) -> Optional[ProcessedFrame]:
        """Decode and validate a single frame.
        
        Args:
            img_bytes: Raw image bytes
            timestamp: Frame timestamp
            
        Returns:
            ProcessedFrame or None if invalid
        """
        try:
            image = Image.open(BytesIO(img_bytes))
            validate_image_dimensions(image)
            return ProcessedFrame(
                timestamp=timestamp,
                image=image.convert("RGB")
            )
        except Exception as e:
            logger.debug("Failed to decode/validate frame: %s", e)
            return None
    
    def close(self) -> None:
        """Close processor and cleanup."""
        if self._timer:
            self._timer.cancel()
            self._timer = None