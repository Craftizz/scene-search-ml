"""Scene emission service for formatting and sending scene messages."""

from typing import List, Optional, Dict, Any
import logging

from app.models import PendingFrame, SceneEmissionConfig
import numpy as np

logger = logging.getLogger("scene_search")


class SceneEmitter:
    """Handle scene message emission and formatting."""
    
    def __init__(self, config: SceneEmissionConfig):
        """Initialize scene emitter.
        
        Args:
            config: Scene emission configuration
        """
        self.config = config
        self._scene_counter = 0
        self._last_emitted_ts: Optional[float] = None
    
    def compute_scene_messages(
        self,
        scene_indices: List[int],
        pending_frames: List[PendingFrame], 
        batch_start: int,
        batch_size: int
    ) -> List[Dict[str, Any]]:
        """Compute scene message payloads from detected boundaries.
        
        Args:
            scene_indices: Detected scene boundary indices
            pending_frames: All pending frames
            batch_start: Start index for current batch
            batch_size: Size of current batch
            
        Returns:
            List of scene messages ready for emission
        """
        messages = []
        last_sent = 0
        
        for seq_i, rel_idx in enumerate(scene_indices):
            global_idx = batch_start + rel_idx
            if global_idx < last_sent:
                continue
                
            # Get scene chunk
            chunk = pending_frames[last_sent : global_idx + 1]
            if not chunk:
                continue
                
            scene_frame = chunk[-1]
            scene_ts = scene_frame.timestamp
            
            # Enforce minimum gap between scenes
            if (self._last_emitted_ts is not None and 
                (scene_ts - self._last_emitted_ts) < self.config.min_scene_gap_sec):
                last_sent = global_idx + 1
                continue
            
            # Compute end timestamp
            end_ts = self._compute_end_timestamp(
                seq_i, scene_indices, batch_start, batch_size, pending_frames
            )
            
            # Create scene message
            self._scene_counter += 1
            scene_msg = {
                "id": self._scene_counter,
                "timestamp": scene_ts,
                "global_index": global_idx
            }
            
            if end_ts is not None:
                scene_msg["end_timestamp"] = end_ts
                
            messages.append(scene_msg)
            self._last_emitted_ts = scene_ts
            last_sent = global_idx + 1
            
        return messages
    
    def _compute_end_timestamp(
        self,
        seq_i: int,
        scene_indices: List[int], 
        batch_start: int,
        batch_size: int,
        pending_frames: List[PendingFrame]
    ) -> Optional[float]:
        """Compute end timestamp for a scene."""
        try:
            if seq_i + 1 < len(scene_indices):
                next_rel = scene_indices[seq_i + 1]
                next_global = batch_start + next_rel
            else:
                next_global = batch_start + batch_size - 1
                
            if 0 <= next_global < len(pending_frames):
                return pending_frames[next_global].timestamp
        except Exception:
            pass
        return None
    
    def get_last_scene_embedding(
        self,
        pending_frames: List[PendingFrame],
        batch_start: int,
        scene_indices: List[int]
    ) -> Optional[np.ndarray]:
        """Get embedding of last detected scene."""
        if not scene_indices:
            return None
            
        try:
            last_rel_idx = scene_indices[-1]
            global_idx = batch_start + last_rel_idx
            
            if 0 <= global_idx < len(pending_frames):
                emb = pending_frames[global_idx].embedding
                if emb is not None:
                    return np.array(emb, dtype=float)
        except Exception:
            pass
        return None