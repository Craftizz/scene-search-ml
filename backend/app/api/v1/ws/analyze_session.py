from __future__ import annotations
import asyncio
from io import BytesIO
from typing import List, Optional
from dataclasses import dataclass

from fastapi import WebSocket, WebSocketDisconnect
from PIL import Image
import numpy as np

from app.core.config import settings
from app.manager.model_manager import ModelManager
from app.services.scene_detection_service import detect_boundaries

"""analyze_session.py

Session-level logic for the `/v1/ws/analyze` websocket endpoint.

This module encapsulates the `AnalyzeSession` class and small data
structures used to buffer frames prior to embedding and scene
detection. The routing layer (`analyze.py`) delegates websocket
connections to `AnalyzeSession` instances so the session logic can be
tested and maintained independently of the router.

Protocol summary:
- Client sends a JSON text message: {"type": "frame_meta", "timestamp": <float>}.
- Client then sends a binary message containing the encoded image bytes.
- Server batches frames, runs the embedder, emits `embedded_batch`, and
    runs scene detection; scene boundaries are emitted as `scene` messages
    with numeric `id` and start/end timestamps.
"""

@dataclass
class PendingFrame:
    timestamp: float
    embedding: Optional[List[float]] = None
    image: Optional[Image.Image] = None


@dataclass
class SessionConfig:
    batch_size: int = 8
    batch_timeout: float = 1.2
    max_pending: int = 1000

    # detection tuning
    threshold: float = 0.9
    persistence: int = 1
    
    # smoothing: running-average (exponential moving average) window size in frames
    smoothing_window: int = 1
    # minimum time gap between emitted scenes (seconds)
    min_scene_gap_sec: float = 1.0


class FrameBatch:
    """Decode and validate raw image bytes into PIL Images."""
    
    @staticmethod
    def decode(raw_bytes: List[bytes]) -> List[Optional[Image.Image]]:
        """Decode a list of image bytes to PIL Images.
        
        Returns a list with the same length as input; failed decodes are None.
        """
        imgs = []
        for b in raw_bytes:
            try:
                imgs.append(Image.open(BytesIO(b)).convert("RGB"))
            except Exception:
                imgs.append(None)
        return imgs
    
    @staticmethod
    def filter_valid(
        images: List[Optional[Image.Image]], 
        timestamps: List[float]
    ) -> tuple[List[Image.Image], List[float]]:
        """Filter out None images and return parallel valid images/timestamps."""
        valid_imgs = []
        valid_ts = []
        for img, ts in zip(images, timestamps):
            if img is not None:
                valid_imgs.append(img)
                valid_ts.append(ts)
        return valid_imgs, valid_ts


class EmbeddingSmoothing:
    """Apply exponential moving average smoothing to embeddings."""
    
    def __init__(self, window: int):
        """Initialize smoother with a window size.
        
        Args:
            window: EMA window size in frames (1 = no smoothing).
        """
        self.window = window
        self._ema: Optional[np.ndarray] = None
    
    def smooth(self, embeddings: List[np.ndarray]) -> List[np.ndarray]:
        """Apply EMA smoothing to a batch of embeddings.
        
        Returns smoothed embeddings (or original if window <= 1).
        """
        if self.window <= 1:
            return embeddings
        
        alpha = 2.0 / (self.window + 1)
        smoothed = []
        for e in embeddings:
            if self._ema is None:
                self._ema = e.copy()
            else:
                self._ema = alpha * e + (1.0 - alpha) * self._ema
            smoothed.append(self._ema.copy())
        return smoothed


class SceneEmitter:
    """Compute scene boundaries and format scene messages."""
    
    def __init__(self, config: SessionConfig):
        """Initialize the emitter with session config."""
        self.config = config
        self._scene_counter: int = 0
        self._last_emitted_ts: Optional[float] = None
    
    def compute_scene_messages(
        self,
        scene_indices: List[int],
        pending: List[PendingFrame],
        batch_start: int,
        batch_size: int
    ) -> List[dict]:
        """Compute scene message payloads from detected boundaries.
        
        Returns a list of scene message dicts ready to send over websocket.
        """
        messages = []
        last_sent = 0
        
        for seq_i, rel_idx in enumerate(scene_indices):
            global_idx = batch_start + rel_idx
            if global_idx < last_sent:
                continue
            
            # extract the pending frames that belong to this scene chunk
            chunk = pending[last_sent : global_idx + 1]
            if not chunk:
                continue
            
            # use the last frame of the chunk as the scene start
            scene_frame = chunk[-1]
            scene_ts = scene_frame.timestamp
            
            # enforce a minimum gap between emitted scenes
            if self._last_emitted_ts is not None:
                try:
                    if (scene_ts - self._last_emitted_ts) < float(self.config.min_scene_gap_sec):
                        last_sent = global_idx + 1
                        continue
                except Exception:
                    pass
            
            # compute a best-effort end timestamp for the scene
            end_ts = self._compute_end_timestamp(
                seq_i, scene_indices, batch_start, batch_size, pending
            )
            
            # sequential numeric scene id (session-scoped)
            try:
                self._scene_counter += 1
                scene_idx = int(self._scene_counter)
            except Exception:
                scene_idx = None
            
            # build message payload
            payload = {"type": "scene", "id": scene_idx, "timestamp": scene_ts}
            if end_ts is not None:
                payload["end_timestamp"] = end_ts
            
            messages.append(payload)
            self._last_emitted_ts = scene_ts
            last_sent = global_idx + 1
        
        return messages
    
    def _compute_end_timestamp(
        self,
        seq_i: int,
        scene_indices: List[int],
        batch_start: int,
        batch_size: int,
        pending: List[PendingFrame]
    ) -> Optional[float]:
        """Compute end timestamp for a scene (next scene start or batch end)."""
        try:
            if seq_i + 1 < len(scene_indices):
                next_rel = scene_indices[seq_i + 1]
                next_global = batch_start + next_rel
            else:
                next_global = batch_start + batch_size - 1
            
            if 0 <= next_global < len(pending):
                return float(pending[next_global].timestamp)
            return None
        except Exception:
            return None
    
    def get_last_scene_embedding(
        self, 
        pending: List[PendingFrame], 
        batch_start: int, 
        scene_indices: List[int]
    ) -> Optional[np.ndarray]:
        """Extract the embedding of the last detected scene for baseline update."""
        if not scene_indices:
            return None
        try:
            last_rel_idx = scene_indices[-1]
            global_idx = batch_start + last_rel_idx
            if 0 <= global_idx < len(pending):
                emb = pending[global_idx].embedding
                if emb is not None:
                    return np.array(emb, dtype=float)
        except Exception:
            pass
        return None


class AnalyzeSession:
    """Manage a single websocket analyze session.

    The session batches incoming frames, runs the embedder on each batch,
    immediately replies with `embedded_batch` and runs scene detection on
    the embeddings. Detected scenes are emitted as `scene` messages.

    The class is self-contained and stores all per-connection state so it
    can be instantiated concurrently for multiple websocket clients.
    """

    def __init__(self, websocket: WebSocket, config: Optional[SessionConfig] = None):
        """Create a new session bound to `websocket`.

        Args:
            websocket: FastAPI `WebSocket` connection for this session.
            config: Optional `SessionConfig` to override defaults.
        """
        self.websocket = websocket
        self.embedder = ModelManager.get_embedder()
        self.config = config or SessionConfig()

        # Helper classes for focused responsibilities
        self._smoother = EmbeddingSmoothing(self.config.smoothing_window)
        self._scene_emitter = SceneEmitter(self.config)

        # Last seen embedding (most recent frame embedding processed)
        self.last_seen_embedding: Optional[np.ndarray] = None
        # Baseline embedding for comparing scenes / continuity across chunks.
        self.last_scene_baseline: Optional[np.ndarray] = None

        # Buffers for incoming binary frames before embedding
        self._bytes_buf: List[bytes] = []
        self._ts_buf: List[float] = []
        self._lock = asyncio.Lock()
        self._timer: Optional[asyncio.Task] = None
        self._flush_task: Optional[asyncio.Task] = None
        self._pending: List[PendingFrame] = []

        self.BATCH_SIZE = min(self.config.batch_size, settings.max_batch_size)
        self.BATCH_TIMEOUT = self.config.batch_timeout
        self._closed = False


    async def _send(self, obj: dict) -> None:
        """Send a JSON serializable object over the websocket.

        The helper intentionally ignores send errors because the client may
        have disconnected; callers should not need to handle send failure.
        """
        try:
            await self.websocket.send_json(obj)
        except Exception:
            # ignore send errors — connection may already be closed
            return


    async def enqueue(self, img_bytes: bytes, timestamp: float) -> None:
        """Queue an incoming image for batching.

        Frames are buffered until either `BATCH_SIZE` is reached or the
        batch timeout fires. If the total queued frames exceed
        `config.max_pending` the frame is dropped and the client is
        notified with an `error` message.

        Args:
            img_bytes: Raw image bytes (JPEG/PNG) sent by the client.
            timestamp: Float timestamp associated with the frame.
        """
        async with self._lock:
            total_queued = len(self._pending) + len(self._bytes_buf)
            if total_queued >= self.config.max_pending:
                try:
                    await self._send({"type": "error", "message": "max_pending exceeded, frame dropped"})
                except Exception:
                    pass
                return

            self._bytes_buf.append(img_bytes)
            self._ts_buf.append(timestamp)
            if self._timer is None:
                self._timer = asyncio.create_task(self._schedule_flush())
            if len(self._bytes_buf) >= self.BATCH_SIZE:
                if self._timer:
                    self._timer.cancel()
                    self._timer = None
                if self._flush_task is None or self._flush_task.done():
                    self._flush_task = asyncio.create_task(self._flush())


    async def _schedule_flush(self) -> None:
        """Sleep for the configured batch timeout and schedule a flush.

        The scheduled flush is cancellable if the batch fills before the
        timeout elapses.
        """
        await asyncio.sleep(self.BATCH_TIMEOUT)
        if self._flush_task is None or self._flush_task.done():
            self._flush_task = asyncio.create_task(self._flush())


    async def _flush(self) -> None:
        """Decode pending bytes, run embedder, and perform scene detection.

        This orchestrates the main processing pipeline using helper classes.
        """
        if self._closed:
            return

        # Step 1: Prepare batch for processing
        work = await self._prepare_batch()
        if work is None:
            return
        work_bytes, work_ts = work

        # Step 2: Decode and filter images
        valid_imgs, valid_ts = self._decode_and_filter_images(work_bytes, work_ts)
        if not valid_imgs:
            return

        # Step 3: Run embedder
        embeddings_np = await self._embed_images(valid_imgs)
        if embeddings_np is None:
            return

        # Step 4: Send embeddings to client and buffer frames
        await self._send_embeddings_and_buffer(embeddings_np, valid_ts, valid_imgs)

        # Step 5: Apply smoothing and detect scenes
        smoothed = self._smoother.smooth(embeddings_np)
        scenes = self._detect_scenes(smoothed)

        # Step 6: Update baselines
        self._update_baselines(scenes, smoothed)

        # Step 7: Emit scene messages
        if scenes:
            await self._emit_scene_messages(scenes, len(embeddings_np))

        # Clear flush task marker
        self._flush_task = None

    async def _prepare_batch(self) -> Optional[tuple[List[bytes], List[float]]]:
        """Snapshot and clear the pending byte/timestamp buffers.
        
        Returns None if no work to do.
        """
        async with self._lock:
            if self._timer:
                self._timer.cancel()
                self._timer = None
            if not self._bytes_buf:
                return None
            work_bytes = self._bytes_buf.copy()
            work_ts = self._ts_buf.copy()
            self._bytes_buf.clear()
            self._ts_buf.clear()
        return work_bytes, work_ts

    def _decode_and_filter_images(
        self, 
        raw_bytes: List[bytes], 
        timestamps: List[float]
    ) -> tuple[List[Image.Image], List[float]]:
        """Decode images and filter out failed decodes."""
        imgs = FrameBatch.decode(raw_bytes)
        return FrameBatch.filter_valid(imgs, timestamps)

    async def _embed_images(self, images: List[Image.Image]) -> Optional[List[np.ndarray]]:
        """Run the embedder on a batch of images.
        
        Returns None on error after notifying the client.
        """
        try:
            if self._closed:
                return None
            results = await self.embedder.embed_images(images)
        except asyncio.CancelledError:
            return None
        except Exception as e:
            await self._send({"type": "error", "message": f"embedding failed: {e}"})
            return None

        embeddings = [r.vector for r in results]
        return [np.array(v, dtype=float) for v in embeddings]

    async def _send_embeddings_and_buffer(
        self,
        embeddings: List[np.ndarray],
        timestamps: List[float],
        images: List[Image.Image]
    ) -> None:
        """Send embedded_batch message and append frames to pending buffer."""
        # Convert to list for JSON serialization
        embeddings_list = [emb.tolist() for emb in embeddings]
        await self._send({
            "type": "embedded_batch",
            "timestamps": timestamps,
            "embeddings": embeddings_list
        })

        # Buffer frames for scene grouping
        for ts, emb, img in zip(timestamps, embeddings_list, images):
            self._pending.append(PendingFrame(timestamp=ts, embedding=emb, image=img))

    def _detect_scenes(self, embeddings: List[np.ndarray]) -> List[dict]:
        """Run scene boundary detection on embeddings."""
        return detect_boundaries(
            embeddings=embeddings,
            prev_embedding=self.last_scene_baseline,
            threshold=self.config.threshold,
            min_scene_length=self.config.persistence,
        )

    def _update_baselines(
        self,
        scenes: List[dict],
        smoothed: List[np.ndarray]
    ) -> None:
        """Update scene baseline and last seen embedding."""
        # Update baseline for continuity: prefer last detected scene embedding
        if scenes:
            try:
                self.last_scene_baseline = np.array(scenes[-1]["embedding"], dtype=float)
            except Exception:
                self.last_scene_baseline = None
        elif smoothed:
            self.last_scene_baseline = smoothed[-1]

        # Record the most recent embedding seen
        self.last_seen_embedding = smoothed[-1] if smoothed else None

    async def _emit_scene_messages(self, scenes: List[dict], batch_size: int) -> None:
        """Compute and send scene messages to the client."""
        batch_start = len(self._pending) - batch_size
        scene_indices = sorted(int(d.get("index", 0)) for d in scenes)

        messages = self._scene_emitter.compute_scene_messages(
            scene_indices, self._pending, batch_start, batch_size
        )

        for msg in messages:
            await self._send(msg)

        # Update baseline to last scene embedding if available
        last_emb = self._scene_emitter.get_last_scene_embedding(
            self._pending, batch_start, scene_indices
        )
        if last_emb is not None:
            self.last_scene_baseline = last_emb

        # Trim pending buffer (keep frames after last emitted scene)
        if scene_indices:
            last_rel_idx = scene_indices[-1]
            last_global = batch_start + last_rel_idx + 1
            self._pending = self._pending[last_global:]


    async def run(self) -> None:
        """Main session run loop.

        Accepts websocket messages until the client disconnects. The
        expected pattern is a JSON ``frame_meta`` text message followed by
        a binary frame containing the image bytes. A ``close`` meta will
        cause a final flush and the session to exit.
        """
        await self.websocket.accept()
        try:
            while True:
                try:
                    msg = await self.websocket.receive()
                except Exception:
                    break

                if msg is None:
                    break

                if msg.get("type") == "websocket.disconnect":
                    break

                if msg.get("text") is not None:
                    try:
                        import json as _json

                        meta = _json.loads(msg["text"]) if isinstance(msg["text"], str) else None
                    except Exception:
                        meta = None

                    if meta and meta.get("type") == "frame_meta":
                        try:
                            bin_msg = await self.websocket.receive()
                        except Exception:
                            continue
                        if bin_msg.get("bytes") is None:
                            continue
                        img_bytes = bin_msg.get("bytes")
                        if not isinstance(img_bytes, (bytes, bytearray)):
                            continue
                        img_bytes = bytes(img_bytes)
                        ts = float(meta.get("timestamp", 0))
                        await self.enqueue(img_bytes, ts)
                    elif meta and meta.get("type") == "close":
                        await self._flush()
                        break
                    else:
                        continue

        except WebSocketDisconnect:
            pass
        finally:
            # mark closed and cancel outstanding tasks
            self._closed = True
            if self._timer:
                try:
                    self._timer.cancel()
                except Exception:
                    pass
                self._timer = None

            if self._flush_task and not self._flush_task.done():
                try:
                    self._flush_task.cancel()
                except Exception:
                    pass

            # attempt a final flush if no active flush task
            if not (self._flush_task and not self._flush_task.done()):
                try:
                    await self._flush()
                except Exception:
                    pass
