"""Video analysis session manager using service-oriented architecture."""

import asyncio
from uuid import uuid4
from typing import List, Optional
import numpy as np
import logging

from fastapi import WebSocket

from app.manager.model_manager import ModelManager
from app.services.frame_processing_service import FrameProcessor
from app.services.websocket_session_service import WebSocketSessionManager
from app.services.scene_detection_service import detect_boundaries
from app.services.scene_emission_service import SceneEmitter
from app.services.embedder_service import EmbeddingSmoother
from app.models import (
    SessionConfig,
    FrameProcessingConfig,
    SceneEmissionConfig,
    EmbeddingSmoothingConfig,
    PendingFrame
)

logger = logging.getLogger("scene_search")



class AnalyzeSession:
    """Orchestrate video analysis using service-oriented architecture."""
    
    def __init__(self, websocket: WebSocket, config: Optional[SessionConfig] = None, ws_key: Optional[str] = None):
        """Initialize session with services.
        
        Args:
            websocket: WebSocket connection
            config: Session configuration 
            ws_key: Optional WebSocket key for rate limiting
        """
        self.config = config or SessionConfig()
        
        # Initialize services
        self._websocket_mgr = WebSocketSessionManager(websocket)
        self._frame_processor = FrameProcessor(FrameProcessingConfig(
            batch_size=self.config.batch_size,
            batch_timeout=self.config.batch_timeout,
            max_pending=self.config.max_pending
        ))
        self._embedder = ModelManager.get_embedder()
        self._embedding_smoother = EmbeddingSmoother(EmbeddingSmoothingConfig(
            window_size=self.config.smoothing_window
        ))
        self._scene_emitter = SceneEmitter(SceneEmissionConfig(
            min_scene_gap_sec=self.config.min_scene_gap_sec
        ))
        
        # Session state
        self._pending_frames: List[PendingFrame] = []
        self._pending_lock = asyncio.Lock()
        self.last_scene_baseline: Optional[np.ndarray] = None
        self.last_seen_embedding: Optional[np.ndarray] = None
        self._ws_key = ws_key

    async def enqueue(self, img_bytes: bytes, timestamp: float) -> None:
        """Queue frame for processing."""
        total_pending = len(self._pending_frames)
        
        success = await self._frame_processor.enqueue_frame(
            img_bytes=img_bytes,
            timestamp=timestamp,
            total_pending=total_pending,
            on_batch_ready=self._process_batch
        )
        
        if not success:
            await self._websocket_mgr.send_error("Frame dropped due to limits")

    async def _process_batch(self) -> None:
        """Process a batch through the full pipeline."""
        # Get processed batch
        batch = await self._frame_processor.get_batch()
        if not batch or not batch.frames:
            return
            
        # Extract images for embedding
        images = [frame.image for frame in batch.frames]
        timestamps = [frame.timestamp for frame in batch.frames]
        
        # Generate embeddings
        try:
            results = await self._embedder.embed_images(images)
            embeddings = [np.array(r.vector, dtype=float) for r in results]
        except Exception as e:
            await self._websocket_mgr.send_error(f"Embedding failed: {e}")
            return
        
        # Apply smoothing
        smoothed = self._embedding_smoother.smooth_embeddings(embeddings)
        
        # Send embeddings to client
        embeddings_list = [emb.tolist() for emb in smoothed]
        await self._websocket_mgr.send_embedded_batch(timestamps, embeddings_list)
        
        # Buffer frames for scene detection
        pending_items = [
            PendingFrame(timestamp=ts, embedding=emb.tolist(), image=img)
            for ts, emb, img in zip(timestamps, smoothed, images)
        ]
        
        async with self._pending_lock:
            self._pending_frames.extend(pending_items)
        
        # Detect scenes
        scenes = detect_boundaries(
            embeddings=smoothed,
            prev_embedding=self.last_scene_baseline,
            threshold=self.config.threshold,
            min_scene_length=self.config.persistence
        )
        
        # Update baseline
        self._update_baselines(scenes, smoothed)
        
        # Emit scene messages
        if scenes:
            await self._emit_scenes(scenes, len(smoothed))

    def _update_baselines(self, scenes: List[dict], smoothed: List[np.ndarray]) -> None:
        """Update scene baseline and last seen embedding."""
        if scenes:
            try:
                self.last_scene_baseline = np.array(scenes[-1]["embedding"], dtype=float)
            except Exception:
                self.last_scene_baseline = smoothed[-1] if smoothed else None
        elif smoothed:
            self.last_scene_baseline = smoothed[-1]
            
        self.last_seen_embedding = smoothed[-1] if smoothed else None

    async def _emit_scenes(self, scenes: List[dict], batch_size: int) -> None:
        """Emit scene messages with captions."""
        scene_indices = sorted(int(d.get("index", 0)) for d in scenes)
        
        async with self._pending_lock:
            pending_snapshot = self._pending_frames.copy()
            batch_start = len(pending_snapshot) - batch_size
            
            messages = self._scene_emitter.compute_scene_messages(
                scene_indices, pending_snapshot, batch_start, batch_size
            )
            
            # Send scene messages with caption requests
            for i, msg in enumerate(messages):
                req_id = f"cap-{uuid4().hex}"
                
                await self._websocket_mgr.send_scene(
                    scene_id=msg["id"],
                    timestamp=msg["timestamp"],
                    end_timestamp=msg.get("end_timestamp"),
                    request_id=req_id
                )
                
                # Schedule caption task
                try:
                    rel_idx = scene_indices[i] if i < len(scene_indices) else None
                    if rel_idx is not None:
                        global_idx = batch_start + rel_idx
                        if 0 <= global_idx < len(pending_snapshot):
                            frame = pending_snapshot[global_idx]
                            if frame and frame.image:
                                asyncio.create_task(self._caption_task(req_id, frame.image))
                except Exception as e:
                    logger.debug("Failed to schedule caption: %s", e)
            
            # Update baseline from last scene
            last_emb = self._scene_emitter.get_last_scene_embedding(
                pending_snapshot, batch_start, scene_indices
            )
            if last_emb is not None:
                self.last_scene_baseline = last_emb
            
            # Trim pending frames
            if scene_indices:
                last_rel_idx = scene_indices[-1]
                last_global = batch_start + last_rel_idx + 1
                if last_global >= len(self._pending_frames):
                    self._pending_frames = []
                elif last_global > 0:
                    self._pending_frames = self._pending_frames[last_global:]

    async def _caption_task(self, request_id: str, image) -> None:
        """Generate caption for scene."""
        try:
            await self._websocket_mgr.send_caption_started(request_id)
            
            captioner = ModelManager.get_captioner()
            results = await captioner.caption([image])
            caption = results[0].text if results else ""
            
            await self._websocket_mgr.send_caption_result(request_id, caption)
            
        except Exception as e:
            logger.exception("Caption task failed: %s", e)
            await self._websocket_mgr.send_error(f"Caption error: {e}", request_id)

    async def run(self) -> None:
        """Main session run loop."""
        await self._websocket_mgr.accept_connection(self._ws_key)
        
        try:
            while self._websocket_mgr.is_connected():
                # Receive frame metadata
                meta = await self._websocket_mgr.receive_frame_meta()
                if meta is None:
                    break
                    
                if meta.get("type") == "frame_meta":
                    # Receive frame bytes
                    img_bytes = await self._websocket_mgr.receive_frame_bytes()
                    if img_bytes is None:
                        continue
                        
                    timestamp = float(meta.get("timestamp", 0))
                    await self.enqueue(img_bytes, timestamp)
                    
                elif meta.get("type") == "close":
                    await self._process_batch()  # Final flush
                    break
                    
        except Exception as e:
            logger.exception("Session error: %s", e)
        finally:
            self._websocket_mgr.close()
            self._frame_processor.close()